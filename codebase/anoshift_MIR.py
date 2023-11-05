from turtle import st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset

import numpy as np
import pandas as pd

from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape,numpyarray_to_tensorarray,load_teset,get_balanced_testset
from utils.classifiers import CICIDS2018_FC
from utils.buffermemory import memory_update,retrieve_replaysamples,retrieve_MIR_replaysamples,random_memory_update
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model
from utils.config.configurations import cfg
from utils.config.configurations import anoshiftsubset as ds
from utils.metadata import initialize_metadata


import time
import random
from collections import Counter
from copy import deepcopy
from fvcore.nn import FlopCountAnalysis



from torchmetrics import Accuracy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot



import tempfile
import json



memory_population_time=0
MIR_time = 0
VPU_sgd_operations = 0
Regular_sgd_operations = 0
flops_count = 0
global_priority_list,local_priority_list=dict(),dict()
classes_so_far,full= set(),set()
local_store = {}
global_count, local_count, replay_count,replay_individual_count,local_count  = Counter(), Counter(),Counter(),Counter(),Counter()
task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,task_num,is_lazy_training,pth_testset,testset_class_ids = None,None,None,None,None,None,None,None,None,None,None
replay_size,memory_size,minority_allocation,epochs,batch_size,device = None,None,None,None,None,None
memory_X, memory_y, memory_y_name = None,None,None
model,opt,loss_fn,train_acc_metric,learning_rate = None,None,None,None,None
nc,total_interfered_samples = 0,0
X_test,y_test = None,None
image_resolution = None
no_tasks = int(cfg.cicids2017.no_tasks)
param_weights_container_Reg_SGD,param_weights_container_vpu_SGD=None,None
bool_store_weights = None
load_whole_dataset = False



def load_metadata():
    # set_seed(125)
    #get_gpu()
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,input_shape,pth_testset,testset_class_ids,load_whole_dataset
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training,param_weights_container_Reg_SGD,bool_store_weights,image_resolution
    # set_seed(125)
    #get_gpu()
    label = ds.label
    cfg.avalanche_dir = False
    set_cl_strategy_name("MIR")
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
    input_shape = get_inputshape(pth,class_ids)
    image_resolution = ds.image_resolution
    
    bool_store_weights = ds.store_weights
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata()
    create_directories(label)
    trigger_logging(label=label)

def load_model_metadata():
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric,input_shape,param_weights_container_Reg_SGD,param_weights_container_vpu_SGD
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    model = model.to(device)
    # model.train()
    opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    train_acc_metric = Accuracy().to(device)
    param_weights_container_Reg_SGD= np.zeros((1,torch.numel(torch.nn.utils.parameters_to_vector(model.parameters()))))
    param_weights_container_vpu_SGD= np.zeros((1,torch.numel(torch.nn.utils.parameters_to_vector(model.parameters()))))

def set_cl_strategy_name(strategy):
    cfg.clstrategy = strategy


def initialize_buffermemory(tasks,mem_size):
    log("initialising the buffer memory")
    global memory_X, memory_y, memory_y_name
    initial_X, initial_y, initial_yname = tasks[0]
    mem = min(mem_size, initial_yname.shape[0])    
    memory_X, memory_y, memory_y_name = initial_X[:mem,:], initial_y[:mem], initial_yname[:mem]
    # memory_X, memory_y, memory_y_name = memory_X.to(device),memory_y.to(device),memory_y_name(device)

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


def get_flopscount(input):
    return FlopCountAnalysis(model, input).total()




def retrieve_MIR_samples(Xt, yt,class_imbalance_aware_sample_selection):
    global batch_size,total_interfered_samples,VPU_sgd_operations,param_weights_container_Reg_SGD,param_weights_container_vpu_SGD,bool_store_weights
    stream_dataset = Tempdataset(Xt, yt)
    # print(Xt.shape)
    # stream_dataset = TensorDataset(Xt, yt)
    stream_train_dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True) 
    epochs = 5

    
    temp_model = deepcopy(model)
    opt = torch.optim.RMSprop(temp_model.parameters(), lr=learning_rate)
    for epoch in range(0,epochs):
        for step,stream in enumerate(stream_train_dataloader):
            x_stream_train, y_stream_train = stream
            x_stream_train,y_stream_train = x_stream_train.float() ,y_stream_train.float() 
            x_stream_train = x_stream_train.to(device)
            y_stream_train = y_stream_train.to(device)
            temp_model = temp_model.to(device)
            # x_stream_train = x_stream_train.reshape(-1,3,32,32)
            
            # print(x_stream_train.shape)
            if image_resolution is not None:
                x_stream_train =  x_stream_train.reshape(image_resolution)

            y_hat = temp_model(x_stream_train).reshape(y_stream_train.shape)
            loss = loss_fn(y_hat,y_stream_train) 
            opt.zero_grad()
            loss.backward()
            opt.step()
            VPU_sgd_operations+=1

    if bool_store_weights:
        param_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy().reshape(1,-1)
        param_vector2 = torch.nn.utils.parameters_to_vector(temp_model.parameters()).detach().cpu().numpy().reshape(1,-1)
        # param_vector = np.concatenate((param_vector,param_vector2),axis=1)
        param_weights_container_Reg_SGD= np.concatenate((param_weights_container_Reg_SGD,param_vector), axis=0)  
        param_weights_container_vpu_SGD= np.concatenate((param_weights_container_vpu_SGD,param_vector2), axis=0)     

    
    temp_model.eval()
    if image_resolution is not None:
        X_temp = torch.from_numpy(memory_X).reshape(image_resolution).to(device)
    else:
        X_temp = torch.from_numpy(memory_X).to(device)    
    yhat = temp_model(X_temp).detach().cpu().round().numpy() 

    # offset = 100
    # for idx in range(0,memory_X.shape[0],offset):
    #     X_test1 = torch.from_numpy(memory_X[idx:idx+offset,:].astype(float)).to(device)
    #     temp = temp_model(X_test1.reshape((-1,3,32,32)).float()).detach().cpu().round().numpy() 
    #     if idx==0:
    #         yhat = temp
    #     else:
    #         yhat = np.append(yhat, temp, axis=0)
    # X_temp = torch.from_numpy(memory_X.astype(float)).to(device)        
    # yhat = temp_model(X_temp.view(-1,3,32,32).float()).detach().cpu().round().numpy()    
    MIR_samples_indices = abs(yhat-memory_y.reshape(yhat.shape))  
    MIR_samples_indices = MIR_samples_indices.ravel().astype('int').tolist()
    MIR_samples_indices = list(map(bool,MIR_samples_indices))    
    interfere_samples = sum(MIR_samples_indices)    
    total_interfered_samples+=interfere_samples    

    

    MIR_X = memory_X[MIR_samples_indices,:]
    MIR_y = memory_y[MIR_samples_indices]
    MIR_y_name = memory_y_name[MIR_samples_indices]
     
    # b_size = int(batch_size/2) 
    # if class_imbalance_aware_sample_selection:
    #     # MIR_X,MIR_y,MIR_y_name = retrieve_MIR_replaysamples(memory_X=MIR_X,memory_y=MIR_y,memory_y_name=MIR_y_name,replay_size=b_size,input_shape=input_shape,local_count=local_count)
    #     MIR_X,MIR_y,MIR_y_name = retrieve_replaysamples(memory_X, memory_y ,memory_y_name,global_priority_list,local_count,replay_size,input_shape,minority_allocation,memory_size,local_store=True)            
    # else:
    #     if MIR_X.shape[0] <= b_size:
    #         b_size = MIR_X.shape[0]
    #     rand_indices = np.random.choice(range(0,MIR_X.shape[0]),size=b_size).tolist()
    #     MIR_X,MIR_y,MIR_y_name = MIR_X[rand_indices,:],MIR_y[rand_indices],MIR_y_name[rand_indices]

    
    
    return MIR_X,MIR_y,MIR_y_name     


def ecbrs_train_epoch(Xt, yt):
    global MIR_time,Regular_sgd_operations
    y_hat,y_stream_train,train_acc,loss = None,None,None,None
    stream_dataset = Tempdataset(Xt, yt)
    stream_train_dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True)       
    model.eval()
    for step, (stream) in enumerate((stream_train_dataloader)):  
        x_stream_train, y_stream_train = stream
        if image_resolution is not None:
            x_stream_train = x_stream_train.reshape(image_resolution)
        y_hat = model(x_stream_train.float()).reshape(y_stream_train.shape)
        loss = loss_fn(y_hat,y_stream_train.float())
        # l2_lambda = 0.0001
        # weights_vec = torch.nn.utils.parameters_to_vector(model.parameters())
        # l2_norm = torch.linalg.vector_norm(weights_vec)
        # # # l2_norm = sum(torch.square(weights_vec/l2_norm))
        # loss = loss + l2_lambda * ((l2_norm**2)-1)
        opt.zero_grad()
        loss.backward()
        opt.step()  
        Regular_sgd_operations+=1
    if y_hat is not None and y_stream_train is not None:
        train_acc = train_acc_metric(y_hat,y_stream_train.to(torch.int)).to(device)  
    else:
        train_acc = 0.0 
        loss = 0    

    return train_acc,loss  


def train_a_task(X,y,y_classname,task_num):
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,MIR_time
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time 

    task_size = X.shape[0]
    for curr_batch in range(0, task_size, replay_size):
            print("till here", curr_batch+replay_size)
            Xt, yt, ynamet = X[curr_batch:curr_batch+replay_size,:], y[curr_batch:curr_batch+replay_size], y_classname[curr_batch:curr_batch+replay_size]
            print("Buffer memory",local_count)
            update_exemplars_global_counter(ynamet)        
            total_count=sum(global_count.values())
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


            beg_time = time.time()
            x_replay_train, y_replay_train,y_name_replay_train = retrieve_MIR_samples(Xt=Xt, yt=yt,class_imbalance_aware_sample_selection=False)
            MIR_time += time.time()-beg_time 

            mem_begin=time.time()
            memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list = memory_update(Xt,yt,ynamet,task_num,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list)
            memory_population_time+=time.time()-mem_begin

            
            update_replay_counter(binarymemorysamples=y_replay_train,classwisememorysamples=y_name_replay_train)
            x_replay_train, y_replay_train,y_name_replay_train = numpyarray_to_tensorarray(x_replay_train),numpyarray_to_tensorarray(y_replay_train),numpyarray_to_tensorarray(y_name_replay_train)
            Xt=torch.cat([numpyarray_to_tensorarray(Xt),(x_replay_train)],0).to(device)        
            yt=torch.cat([numpyarray_to_tensorarray(yt),y_replay_train],0).to(device)
            shuffler = torch.randperm(Xt.size()[0])
            Xt = Xt[shuffler]
            yt = yt[shuffler]
            
            for epoch in range(epochs):
                train_acc,loss = ecbrs_train_epoch(Xt=Xt,yt=yt)        
        
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print("loss over epoch: %.4f" % (float(loss),))
            print("Number of interefere samples:",(total_interfered_samples-current_interfere_samples))
        

            
    if ds.enable_checkpoint:
        checkpoint_location = str(cfg.outputdir) + '/models' +'/task_'+str(task_num)+ '.th'
        # print("location:",checkpoint_location)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location)        



def train(tasks):
    task_id_temp = 0
    global task_num 
    for X,y,y_classname in tasks:

        if not is_lazy_training:
            task_num = task_id_temp
        
        task_id_temp+=1 
        log("training task %s",task_num)
        print("task number:",task_num)
        # X = X.reshape((-1,3,32,32))
        train_a_task(X=X,y=y,y_classname=y_classname,task_num=task_num)
        
       

            
def evaluate_on_testset():
    global X_test,y_test
    # X_test,y_test = avalanche_tensor_to_tensor(test_data_x = X_test,test_data_y = y_test)
    # X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    # X_test = torch.from_numpy(X_test.astype(float)).to(device)
    # model.eval()
    # yhat = model(X_test.float()).detach().cpu().numpy()
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
    #     X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    # else:
    #     X_test,y_test = get_balanced_testset(X=X_test,y=y_test)

    yhat = None
    # model.to('cpu')
    model.eval()
    print("computing the results")
    offset = 2500
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        
        # temp = model(X_test1.float()).detach().cpu().numpy()
        if image_resolution is not None:
            X_test1 = X_test1.reshape(image_resolution)
        temp = model(X_test1.float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))


def taskwise_lazytrain():
    global test_x,test_y,task_num,flops_count
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False,load_whole_train_data=load_whole_dataset)
        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_id == int(0):
            # tasks[0][0] = tasks[0][0].reshape((-1,3,32,32))
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
            update_buffermemory_counter(memorysamples=memory_y_name)
            update_mem_samples_indexdict(memorysamples=memory_y_name)
            # flops_count = get_flopscount(torch.from_numpy(tasks[0][0][0,:].reshape(1,3,32,32)).to(device))
        train(tasks)





def start_execution():
    global input_shape,tasks,X_test,y_test,test_x,test_y
    start_time=time.time()
    load_metadata()
    # load_model_metadata()
    print(model)
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
    print("Total MIR time is",MIR_time)
    print("Total interfered samples",total_interfered_samples)
    print("Individual replay samples count",replay_individual_count)
    print("Regular SGD updates",Regular_sgd_operations)
    print("VPU SGD ops",VPU_sgd_operations)
    total_sgd_ops = VPU_sgd_operations+Regular_sgd_operations
    VPU_total_ops = VPU_sgd_operations/total_sgd_ops
    print("Percentage of VPU ops",VPU_total_ops)
    print("Total flops operation",flops_count*total_sgd_ops)
    if bool_store_weights:
        checkpoint_location = str(cfg.param_weights_dir_MIR)
        # print("location:",checkpoint_location)
        print("strategy",cfg.clstrategy)
        np.save(checkpoint_location+label+"_"+cfg.clstrategy+"_Reg_SGD",param_weights_container_Reg_SGD)   
        np.save(checkpoint_location+label+"_"+cfg.clstrategy+"_vpu_SGD",param_weights_container_vpu_SGD) 







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    #parser.add_argument('--ds', type=str, default="cifar100", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    s_time = time.time()
    args = parser.parse_args()
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

