from turtle import st
import torchvision
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch.nn.functional import cosine_similarity
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as D
from torch import linalg as LA

import numpy as np
import pandas as pd
from collections import deque
from copy import deepcopy

from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape,numpyarray_to_tensorarray,load_teset,get_balanced_testset
from utils.buffermemory import memory_update,retrieve_replaysamples,retrieve_MIR_replaysamples,random_memory_update
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model,obtain_grad_vector,check_grad_exist
from utils.config.configurations import cfg
from utils.config.configurations import mnist_cifar10 as ds
from utils.metadata import initialize_metadata


import time
import random
from collections import Counter
import math
import os
from fvcore.nn import FlopCountAnalysis


import tempfile
import json



memory_population_time=0
MIR_time = 0
Regular_sgd_operations = 0
VPU_sgd_operations=0
flops_count = 0
global_priority_list,local_priority_list=dict(),dict()
classes_so_far,full= set(),set()
local_store = {}
global_count, local_count, replay_count,replay_individual_count,local_count  = Counter(), Counter(),Counter(),Counter(),Counter()
task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,is_lazy_training,task_num,pth_testset,testset_class_ids = None,None,None,None,None,None,None,None,None,None,None
replay_size,memory_size,minority_allocation,epochs,batch_size,device = None,None,None,None,None,None
memory_X, memory_y, memory_y_name = None,None,None
model,opt,loss_fn,train_acc_metric,learning_rate = None,None,None,None,None
temp_model,opt_MIR = None,None
nc,total_interfered_samples = 0,0
X_test,y_test = None,None
param_weights_queue = None 
image_resolution = None
no_tasks = 0
param_weights_container_Reg_SGD,param_weights_container_vpu_SGD=None,None
bool_store_weights,bool_store_grads = None,None
load_whole_dataset = False
gauss_noise_arary = None
gauss_noise_iter,gauss_noise_threshold = 0,3
gmm,gmm_train_time = None,0
dist_list = list()




def load_metadata(): 
    log('loading meta data')   
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,no_tasks,param_weights_queue,is_lazy_training ,load_whole_dataset,bool_store_grads,trigger_MIR_Queue
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,learning_rate,input_shape,pth_testset,testset_class_ids,bool_store_weights,image_resolution
    # set_seed(125)
    # set_seed(125)
    #get_gpu()
    label = ds.label
    cfg.avalanche_dir = False
    set_cl_strategy_name("MIR-taskagnostic")
    no_tasks = ds.no_tasks
    metadata_dict = initialize_metadata(label)
    temp_dict = metadata_dict[no_tasks]
    task_order = temp_dict['task_order']
    class_ids = temp_dict['class_ids']
    minorityclass_ids = temp_dict['minorityclass_ids']
    pth = temp_dict['path']
    if 'path_testset' in temp_dict:
        pth_testset = temp_dict['path_testset']
        testset_class_ids = temp_dict['testset_class_ids']
        load_whole_dataset = True
    tasks_list = temp_dict['tasks_list']
    task2_list = temp_dict['task2_list']
    replay_size = ds.replay_size
    memory_size = ds.mem_size
    minority_allocation = ds.minority_allocation
    epochs = ds.n_epochs
    batch_size = ds.batch_size
    device = cfg.device
    learning_rate = ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    bool_store_weights = ds.store_weights
    input_shape = get_inputshape(pth,class_ids)
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    param_weights_queue = deque([],maxlen=ds.param_weights_queue_length)
    trigger_MIR_Queue = deque([],maxlen=3)
    load_model_metadata()
    create_directories(label)
    trigger_logging(label=label)
    image_resolution = ds.image_resolution
    bool_store_grads = ds.store_grads
   
    

def load_model_metadata():
    global param_weights_queue
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric,input_shape,param_weights_container_Reg_SGD,param_weights_container_vpu_SGD,gauss_noise_arary,opt_MIR,temp_model
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    model = model.to(device)
    # model.train()
    # opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    loss_fn = torch.nn.BCELoss()
    train_acc_metric = Accuracy().to(device)

    # temp_model = deepcopy(model).to(device)
    # opt_MIR = torch.optim.SGD(temp_model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    # opt_MIR = torch.optim.RMSprop(temp_model.parameters(), lr=learning_rate)
    param_vector = torch.nn.utils.parameters_to_vector(model.parameters())
    
    param_weights_queue.append(param_vector)
    param_weights_container_Reg_SGD= np.zeros((1,torch.numel(torch.nn.utils.parameters_to_vector(model.parameters()))))
    param_weights_container_vpu_SGD= np.zeros((1,torch.numel(torch.nn.utils.parameters_to_vector(model.parameters()))))
    


def get_flopscount(input):
    return FlopCountAnalysis(model, input).total()     


def set_cl_strategy_name(strategy):
    cfg.clstrategy = strategy    


def convertto_3channels(Xt):
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToPILImage(),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor()]
    )
    transform2=torchvision.transforms.Compose(
        [torchvision.transforms.Resize([28,28])]
    )
    x_temp = np.zeros((Xt.shape[0],3,28,28))
    for i in range(0,Xt.shape[0]):  
        if  Xt[i,:].shape[0] == 784:
            x_temp[i,:,:,:] = transform((Xt[i,:].reshape(28,28,1))).numpy()#.squeeze().reshape(1,2352)
        else:
            x_temp[i,:,:,:] = transform2(torch.from_numpy(Xt[i,:].reshape(3,32,32))).numpy()#.squeeze().reshape(1,2352)
    
    return x_temp


def initialize_buffermemory(tasks,mem_size):
    log("initialising the buffer memory")
    global memory_X, memory_y, memory_y_name
    initial_X, initial_y, initial_yname = tasks[0]
    mem = min(mem_size, initial_yname.shape[0])    
    memory_X, memory_y, memory_y_name = convertto_3channels(initial_X[:mem,:]), initial_y[:mem], initial_yname[:mem]

def update_buffermemory_counter(memorysamples):
    global local_count
    for class_ in memorysamples:
        local_count[class_]+=1

def update_exemplars_global_counter(samples):
    global global_count,classes_so_far,nc
    for j in range(len(samples)):
      global_count[samples[j]]+=1# global_count stores "class_name : no. of class_name instances in the stream so far"
      if samples[j] not in classes_so_far:
        classes_so_far.add(samples[j])
        nc += 1  


def update_replay_counter(binarymemorysamples,classwisememorysamples):
    global replay_count,replay_individual_count
    for b_class_,class_ in zip(binarymemorysamples,classwisememorysamples):
        replay_count[b_class_]+=1
        replay_individual_count[class_]+=1


def update_mem_samples_indexdict(memorysamples):
    global local_store
    for idx,class_ in enumerate(memorysamples):
        if class_ in local_store :
            local_store[class_].append(idx)
        else:
            local_store[class_] = [idx]

def compute_variance():
    global dist_list   
    l1_norm_dist = list()
    for idx in range(len(dist_list)):
        l1_norm_dist.append(torch.norm(torch.abs(dist_list[idx]), p=2))
        
    print("mean value is",torch.mean(torch.Tensor(l1_norm_dist)))
    print("variance is:",torch.var(torch.Tensor(l1_norm_dist)))
    print("standard deviation",torch.std(torch.Tensor(l1_norm_dist)))
    print("coefficient of variation is",torch.std(torch.Tensor(l1_norm_dist))/torch.mean(torch.Tensor(l1_norm_dist))) 
             



def fit_gmm():
    global dist_list,gmm
    weights = torch.ones(2,device=device,requires_grad=True)
    # print(weights)
    no_of_ele = torch.numel(dist_list[0])
    means = torch.tensor(np.random.randn(2,no_of_ele),device=device,requires_grad=True)
    stdevs = torch.tensor(np.abs(np.random.randn(2,no_of_ele)),device=device,requires_grad=True)
    # means = torch.tensor(np.zeros((2,no_of_ele)),device=device,requires_grad=True)
    # stdevs = torch.tensor(np.abs(np.ones((2,no_of_ele))),device=device,requires_grad=True)
    parameters = [weights, means, stdevs]
    optimizer1 = torch.optim.SGD(parameters, lr=0.001, momentum=0.9)
    soft_max=torch.nn.Softmax(dim=0)
    for idx in range(0,len(dist_list)):
        for i in range(0,5):
            # print(soft_max(weights))
            mix = D.Categorical(soft_max(weights))
            std_weight = 1e-4
            comp = D.Independent(D.Normal(means,std_weight*stdevs.abs()), 1)
            gmm = D.MixtureSameFamily(mix, comp)

            optimizer1.zero_grad()
            # print(dist_list[idx])
            loss2 = -gmm.log_prob(dist_list[idx].squeeze()).mean()#-densityflow.log_prob(inputs=x).mean()
            loss2.backward()
            optimizer1.step()
        # print("loss values is :",loss2.detach().cpu().float().numpy())
    #print(weights)
    #print(means)
    #print(parameters[0])
    #exit()



def retrieve_MIR_samples(Xt, yt):
    global batch_size,total_interfered_samples,VPU_sgd_operations,param_weights_container_Reg_SGD,param_weights_container_vpu_SGD,bool_store_weights,temp_model,gauss_noise_arary,gmm,opt_MIR,dist_list
    stream_dataset = Tempdataset(Xt, yt)
    
    # print(Xt.shape)
    # stream_dataset = TensorDataset(Xt, yt)
    stream_train_dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True) 
    epochs = 5
    temp_model = deepcopy(model).to(device)
    opt_MIR = torch.optim.SGD(temp_model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    # opt_MIR = torch.optim.RMSprop(temp_model.parameters(), lr=learning_rate)
    
    
    
    for step,stream in enumerate(stream_train_dataloader):    
        for epoch in range(0,epochs):
            x_stream_train, y_stream_train = stream
            x_stream_train,y_stream_train = x_stream_train.float() ,y_stream_train.float() 
            x_stream_train = x_stream_train
            x_stream_train = x_stream_train.to(device)
            y_stream_train = y_stream_train.to(device)
            #temp_model = temp_model.to(device)
            # x_stream_train = x_stream_train.reshape(-1,3,224,224)
            
            # print(x_stream_train.shape)
            #if image_resolution is not None:
             #   x_stream_train = x_stream_train.reshape(image_resolution)
            y_hat = temp_model(x_stream_train).reshape(y_stream_train.shape)
            loss = loss_fn(y_hat,y_stream_train) 
            opt_MIR.zero_grad()
            loss.backward()
            opt_MIR.step()
            VPU_sgd_operations+=1

        param_vector = torch.nn.utils.parameters_to_vector(model.parameters()).reshape(1,-1)#.detach().cpu().numpy().reshape(1,-1)
        param_vector2 = torch.nn.utils.parameters_to_vector(temp_model.parameters()).reshape(1,-1)#.detach().cpu().numpy().reshape(1,-1)
        # diff =torch.cdist(torch.from_numpy(param_vector).reshape(1,-1),torch.from_numpy(param_vector2).reshape(1,-1),p=1).detach().cpu()
        
        diff = torch.sub(param_vector2,param_vector)        
        dist_list.append(diff)       
    
    

    if bool_store_weights:
        
        # param_vector = np.concatenate((param_vector,param_vector2),axis=1)
        param_weights_container_Reg_SGD= np.concatenate((param_weights_container_Reg_SGD,param_vector.detach().cpu().numpy().reshape(1,-1)), axis=0)  
        param_weights_container_vpu_SGD= np.concatenate((param_weights_container_vpu_SGD,param_vector2.detach().cpu().numpy().reshape(1,-1)), axis=0)   


   
    
    temp_model.eval()
    # offset = 100
    # for idx in range(0,memory_X.shape[0],offset):
    #     X_test1 = torch.from_numpy(memory_X[idx:idx+offset,:].astype(float)).to(device)
    #     if image_resolution is not None:
    #         X_test1 = X_test1.reshape(image_resolution)
    #     temp = temp_model(X_test1.float()).detach().cpu().round().numpy() 
    #     if idx==0:
    #         yhat = temp
    #     else:
    #         yhat = np.append(yhat, temp, axis=0)
    # X_temp = torch.from_numpy(memory_X.astype(float)).to(device)        
    # yhat = temp_model(X_temp.view(-1,3,224,224).float()).detach().cpu().round().numpy()   
    X_temp = torch.from_numpy(memory_X.astype(float)).to(device)
    #if image_resolution is not None:
     #   X_temp = X_temp.reshape(image_resolution) 
    yhat = temp_model(X_temp.float()).detach().cpu().round().numpy()     
    MIR_samples_indices = abs(yhat-memory_y.reshape(yhat.shape))  
    MIR_samples_indices = MIR_samples_indices.ravel().astype('int').tolist()
    MIR_samples_indices = list(map(bool,MIR_samples_indices))    
    interfere_samples = sum(MIR_samples_indices)    
    total_interfered_samples+=interfere_samples    

    
    temp_model.train()
    MIR_X = memory_X[MIR_samples_indices,:]
    MIR_y = memory_y[MIR_samples_indices]
    MIR_y_name = memory_y_name[MIR_samples_indices]
     
    
    
    return MIR_X,MIR_y,MIR_y_name,diff    




def retrieve_MIR_samples_moving_averge():
    global batch_size,total_interfered_samples,param_weights_queue,param_weights_container_Reg_SGD,param_weights_container_vpu_SGD,bool_store_weights,cos_sim_list,std_noise,gmm
     
    
    
    mean_param_weights = torch.mean(torch.stack(list(param_weights_queue)),0)
    gmm_noise = gmm.sample().to(device)
    mean_param_weights = gmm_noise#mean_param_weights + gmm_noise    
    torch.nn.utils.vector_to_parameters(mean_param_weights.float(),temp_model.parameters())
           
    
    
    temp_model.eval()
    X_temp = torch.from_numpy(memory_X.astype(float)).to(device)
    #if image_resolution is not None:
     #   X_temp = X_temp.reshape(image_resolution)

    yhat = temp_model(X_temp.float()).detach().cpu().round().numpy()  
    MIR_samples_indices = abs(yhat-memory_y.reshape(yhat.shape))  
    MIR_samples_indices = MIR_samples_indices.ravel().astype('int').tolist()
    MIR_samples_indices = list(map(bool,MIR_samples_indices))  
    # rand_indices = np.random.choice(range(0,len(MIR_samples_indices)),size=math.ceil(0.40*len(MIR_samples_indices)),replace=True).tolist()  
    # MIR_X = memory_X[rand_indices,:]
    # MIR_y = memory_y[rand_indices]
    # MIR_y_name = memory_y_name[rand_indices]
    rand_indices = MIR_samples_indices


    

    MIR_X = memory_X[MIR_samples_indices,:]
    MIR_y = memory_y[MIR_samples_indices]
    MIR_y_name = memory_y_name[MIR_samples_indices]

    if MIR_X.shape[0] > 1:
        temp = Counter(MIR_y_name.astype(np.int64))   
        weights = list(1/temp[MIR_y_name[i].astype(np.int64)] for i in range(0,MIR_y_name.shape[0],1))
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]      
            rand_indices = np.random.choice(range(0,MIR_X.shape[0]),size=math.ceil(0.15*MIR_X.shape[0]),replace=False,p=weights).tolist()
        else:
            rand_indices = np.random.choice(range(0,len(MIR_X.shape[0])),size=math.ceil(0.15*MIR_X.shape[0]),replace=True).tolist()  

        MIR_X = MIR_X[rand_indices,:]
        MIR_y = MIR_y[rand_indices]
        MIR_y_name = MIR_y_name[rand_indices]
        

    if bool_store_weights:
        param_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy().reshape(1,-1)
        param_vector2 = torch.nn.utils.parameters_to_vector(temp_model.parameters()).detach().cpu().numpy().reshape(1,-1)
        # param_vector = np.concatenate((param_vector,param_vector2),axis=1)
        param_weights_container_Reg_SGD= np.concatenate((param_weights_container_Reg_SGD,param_vector), axis=0)  
        param_weights_container_vpu_SGD= np.concatenate((param_weights_container_vpu_SGD,param_vector2), axis=0)  

             
       
     
     
    total_interfered_samples+=len(rand_indices)
    
    return MIR_X,MIR_y,MIR_y_name

    
    
   


def ecbrs_train_epoch(Xt, yt):
    global MIR_time,param_weights_queue,Regular_sgd_operations
    stream_dataset = Tempdataset(Xt, yt)
    stream_train_dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True,drop_last=False)       
    model.eval()
    for step, (stream) in enumerate((stream_train_dataloader)):  
        x_stream_train, y_stream_train = stream
        #if image_resolution is not None:
         #   x_stream_train = x_stream_train.reshape(image_resolution)
        y_hat = model(x_stream_train.float()).reshape(y_stream_train.shape)
        loss = loss_fn(y_hat,y_stream_train.float())

        
        opt.zero_grad()
        loss.backward()
        opt.step()  
        Regular_sgd_operations+=1
        param_vector = torch.nn.utils.parameters_to_vector(model.parameters())
        param_weights_queue.append(param_vector)
    train_acc = train_acc_metric(y_hat,y_stream_train.to(torch.int)).to(device) 
      
   

    return train_acc,loss  


def train_a_task(X,y,y_classname,task_num):
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,MIR_time,gmm_train_time
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time
  
    global gauss_noise_iter,gauss_noise_threshold
    
    gauss_noise_iter1,gauss_noise_threshold1 =0,3
     

    task_size = X.shape[0]
    gauss_noise_threshold = min(math.floor((task_size/replay_size))-1,4)
    print("gaussian noise threshold is:::",gauss_noise_threshold)
    for curr_batch in range(0, task_size, replay_size):
            print("till here", curr_batch+replay_size)
            Xt, yt, ynamet = X[curr_batch:curr_batch+replay_size,:], y[curr_batch:curr_batch+replay_size], y_classname[curr_batch:curr_batch+replay_size]
            Xt=convertto_3channels(Xt=Xt)
            print("Buffer memory",local_count)
            update_exemplars_global_counter(ynamet)        
            total_count=sum(global_count.values())
            gauss_noise_iter1 = gauss_noise_iter1+1
            a=1/nc    
            current_interfere_samples = total_interfered_samples
            # batch_size=100
            # for batch in range(0,Xt.shape[0],batch_size):
            #     Xt_temp = Xt[batch:batch+batch_size,:]
            #     yt_temp = yt[batch:batch+batch_size]
            #     beg_time = time.time()
            #     x_replay_train, y_replay_train,y_name_replay_train = retrieve_MIR_samples(Xt=memory_X, yt=memory_y,class_imbalance_aware_sample_selection=False)
            #     x_replay_train, y_replay_train,y_name_replay_train = x_replay_train[0:batch_size,:],y_replay_train[0:batch_size],y_name_replay_train[0:batch_size]
            #     update_replay_counter(binarymemorysamples=y_replay_train,classwisememorysamples=y_name_replay_train)
            #     MIR_time += time.time()-beg_time
            #     mem_begin=time.time()
            #     memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list = memory_update(Xt,yt,ynamet,task_num,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list)
            #     memory_population_time+=time.time()-mem_begin
            #     x_replay_train, y_replay_train,y_name_replay_train = numpyarray_to_tensorarray(x_replay_train),numpyarray_to_tensorarray(y_replay_train),numpyarray_to_tensorarray(y_name_replay_train)
                
            #     Xt_temp=torch.cat([numpyarray_to_tensorarray(Xt_temp),(x_replay_train)],0).to(device)        
            #     yt_temp=torch.cat([numpyarray_to_tensorarray(yt_temp),y_replay_train],0).to(device)
            #     epochs = 2
            #     for epoch in range(epochs):
            #         train_acc,loss = ecbrs_train_epoch(Xt=Xt_temp,yt=yt_temp)    
            
            
            if gauss_noise_iter1 < gauss_noise_threshold1:
                beg_time = time.time()
                x_replay_train, y_replay_train,y_name_replay_train,diff = retrieve_MIR_samples(Xt=Xt, yt=yt)
                MIR_time += time.time()-beg_time
                

            elif gauss_noise_iter1 == gauss_noise_threshold1:
                compute_variance()
                # exit()
                st = time.time()
                fit_gmm()
                gmm_train_time = time.time()-st
                x_replay_train, y_replay_train,y_name_replay_train = retrieve_MIR_samples_moving_averge()
            else:
                
                x_replay_train, y_replay_train,y_name_replay_train = retrieve_MIR_samples_moving_averge()
               
                    
             

            mem_begin=time.time()
            memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list = random_memory_update(Xt,yt,ynamet,task_num,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list)
            memory_population_time+=time.time()-mem_begin


            # update_replay_counter(binarymemorysamples=y_replay_train,classwisememorysamples=y_name_replay_train)
            x_replay_train, y_replay_train,y_name_replay_train = numpyarray_to_tensorarray(x_replay_train),numpyarray_to_tensorarray(y_replay_train),numpyarray_to_tensorarray(y_name_replay_train)
            Xt=torch.cat([numpyarray_to_tensorarray(Xt),(x_replay_train)],0).to(device)        
            yt=torch.cat([numpyarray_to_tensorarray(yt),y_replay_train],0).to(device) 
            shuffler = torch.randperm(Xt.size()[0])
            Xt = Xt[shuffler]
            yt = yt[shuffler]
            for epoch in range(epochs):
                train_acc,loss = ecbrs_train_epoch(Xt=Xt,yt=yt) 
            Xt.detach(),yt.detach()
            del Xt,yt    

                 
        
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print("loss over epoch: %.4f" % (float(loss),))
            intf_samples = total_interfered_samples-current_interfere_samples
           
            print("Number of interefere samples:",intf_samples)
           

           
    if ds.enable_checkpoint:
        checkpoint_location = str(cfg.outputdir) + '/models' +'/task_'+str(task_num)+ '.th'
        # print("location:",checkpoint_location)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location)        



def train(tasks):
    task_id_temp = 0
    global task_num,gauss_noise_iter,gauss_noise_arary
    for X,y,y_classname in tasks:

        if not is_lazy_training:
            task_num = task_id_temp
        
        task_id_temp+=1 
        # log("training task %s",task_num)
        print("task number:",task_num)
        
        train_a_task(X=X,y=y,y_classname=y_classname,task_num=task_num)
        
       

            


def evaluate_on_testset():
    global X_test,y_test
    
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    #     X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    # else:
    #     X_test,y_test = get_balanced_testset(X=X_test,y=y_test)

    yhat = None
    model.to('cpu')
    model.eval()
    print("computing the results")
    for idx in range(0,X_test.shape[0],25000):
        idx1=idx
        idx2 = idx1+25000
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float))#.to(device)
        
        # temp = model(X_test1.float()).detach().cpu().numpy()
        #if image_resolution is not None:
         #   X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))



def taskwise_lazytrain():
    global test_x,test_y,task_num,flops_count
    random.shuffle(task_order)
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=True,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_dataset)
        X_test = convertto_3channels(Xt=X_test)
        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_id == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
            update_buffermemory_counter(memorysamples=memory_y_name)
            update_mem_samples_indexdict(memorysamples=memory_y_name)
            single_sample = tasks[0][0][0,:]
            #if image_resolution is not None:
             #   single_sample = single_sample.reshape(image_resolution)
            # flops_count = get_flopscount(torch.from_numpy(single_sample).to(device))
        train(tasks)





def start_execution():
    global input_shape,tasks,X_test,y_test,test_x,test_y,gmm_train_time,VPU_sgd_operations
    start_time=time.time()
    load_metadata()
    # load_model_metadata()
    # print(model)
    if is_lazy_training:
        test_x,test_y = [],[]
        taskwise_lazytrain()
        X_test,y_test = np.concatenate( test_x, axis=0 ),np.concatenate( test_y, axis=0 )
        

    else:
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False)
        initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        print('Total no.of tasks', len(tasks))
        update_buffermemory_counter(memorysamples=memory_y_name)
        update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks=tasks)
    print("Total execution time is--- %s seconds ---" % (time.time() - start_time))
    print("Total memory population time is--- %s seconds ---" % (memory_population_time))
    print("MIR time",MIR_time)
    print("Total interfered samples",total_interfered_samples)
    # print("Individual replay samples count",replay_individual_count)
    print("Regular SGD updates",Regular_sgd_operations)
    print("virtual parameter updates",VPU_sgd_operations)
    print("Total flops operation",flops_count*Regular_sgd_operations)
    # print("Average cosine similarity is:",sum(cos_sim_list) / len(cos_sim_list))
    print("GMM train time",gmm_train_time)
    print("Virtual parameter updates",VPU_sgd_operations)
    if bool_store_weights:
        checkpoint_location = str(cfg.param_weights_dir_MIR)+label
        os.makedirs(checkpoint_location,exist_ok = True)
        # print("location:",checkpoint_location)
        print("strategy",cfg.clstrategy)
        np.save(checkpoint_location+"/"+cfg.clstrategy+"_Reg_SGD",param_weights_container_Reg_SGD)   
        np.save(checkpoint_location+"/"+cfg.clstrategy+"_vpu_SGD",param_weights_container_vpu_SGD) 







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    #parser.add_argument('--ds', type=str, default="mnist_cifar10", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    s_time = time.time()
    args = parser.parse_args()
    #set_seed(args.seed)
    set_seed(2)
    get_gpu(args.gpu)
    auc_result= {}
    start_execution()
    e_time = time.time()-s_time
    #evaluate_on_testset()
    with open(args.filename, 'w') as fp:
         test_set_results = evaluate_on_testset()
         test_set_results.extend([e_time,memory_population_time])
         auc_result[str(args.seed)] = test_set_results
         json.dump(auc_result, fp) 

