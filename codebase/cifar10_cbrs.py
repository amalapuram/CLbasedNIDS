from turtle import st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from utils.customdataloader import load_dataset,Tempdataset,compute_total_minority_testsamples,get_inputshape
from utils.classifiers import CICIDS2018_FC
from utils.buffermemory import cbrsmemory_update,retrieve_replaysamples
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model
from utils.config.configurations import cfg
from utils.config.configurations import cifar100_large_benign as ds
from utils.metadata import initialize_metadata
import json


import time
import random
from collections import Counter



from torchmetrics import Accuracy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

scaler = MinMaxScaler()

memory_population_time=0
global_priority_list=dict()
local_priority_list=dict()
local_count = Counter()
classes_so_far = set()
full = set()
local_store = {}
global_count, local_count, replay_count,replay_individual_count = Counter(), Counter(),Counter(),Counter()
input_shape,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate = None,None,None,None,None,None,None,None,None
replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training,task_num = None,None,None,None,None,None,None,None,None
memory_X, memory_y, memory_y_name = None,None,None
model,opt,loss_fn,train_acc_metric = None,None,None,None
test_x,test_y = None,None
nc = 0
no_tasks = 0




def load_metadata():
    #set_seed(125)
    #get_gpu()
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,learning_rate,input_shape
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,pattern_per_exp,is_lazy_training
    #set_seed(125)
    #get_gpu()
    label = ds.label
    cfg.avalanche_dir = False
    set_cl_strategy_name(0)
    no_tasks = ds.no_tasks
    metadata_dict = initialize_metadata(label)
    temp_dict = metadata_dict[no_tasks]
    task_order = temp_dict['task_order']
    class_ids = temp_dict['class_ids']
    minorityclass_ids = temp_dict['minorityclass_ids']
    pth = temp_dict['path']
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
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata()
    create_directories(label)
    trigger_logging(label=label)

def load_model_metadata():
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric,input_shape
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    model = model.to(device)
    # model.train()
    # opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    loss_fn = torch.nn.BCELoss()
    train_acc_metric = Accuracy().to(device)

def set_cl_strategy_name(strategy_id):
    if strategy_id == 0:
        cfg.clstrategy = "CBRS"            
    elif strategy_id == 1:
        cfg.clstrategy = "ECBRS"
     
          

def initialize_buffermemory(tasks,mem_size):
    global memory_X, memory_y, memory_y_name
    initial_X, initial_y, initial_yname = tasks[0]
    mem = min(mem_size, initial_yname.shape[0])    
    memory_X, memory_y, memory_y_name = initial_X[:mem,:], initial_y[:mem], initial_yname[:mem]

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

def get_balanced_testset(X,y):
    X_test,y_test = X,y
    bool_idx_list = list()
    no_of_ones= np.count_nonzero(y_test == 1)
    c=0
    for idx in range(X_test.shape[0]):
        if y_test[idx] == 0 and c <=no_of_ones:
            bool_idx_list.append(True)
            c+=1
        elif y_test[idx] == 0 and c >no_of_ones:
            bool_idx_list.append(False)
        else:
            bool_idx_list.append(True)
    X_test = X_test[bool_idx_list]
    y_test = y_test[bool_idx_list]

    return X_test,y_test




def ecbrs_train_epoch(Xt, yt,replay_Xt, replay_yt,a):
    # global model,opt
    stream_dataset = Tempdataset(Xt, yt)
    stream_train_dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True)    

    replay_dataset = Tempdataset(replay_Xt, replay_yt)
    replay_train_dataloader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True) 

    for step, (stream,replay) in enumerate(zip(stream_train_dataloader,replay_train_dataloader)):

      x_stream_train, y_stream_train = stream
      x_replay_train, y_replay_train = replay
      x_stream_train, y_stream_train = x_stream_train.to(device),y_stream_train.to(device)
      x_replay_train, y_replay_train = x_replay_train.to(device) , y_replay_train.to(device)            

      y_hat_stream = model(x_stream_train.reshape((-1,3,32,32))).reshape(y_stream_train.shape)
      loss_stream = loss_fn(y_hat_stream,y_stream_train)

      y_hat_replay = model(x_replay_train.reshape((-1,3,32,32))).reshape(y_replay_train.shape)
      loss_replay = loss_fn(y_hat_replay,y_replay_train)

      loss = a * loss_stream + (1-a) * loss_replay
            

      opt.zero_grad()
      loss.backward()
      opt.step()  
    train_acc = train_acc_metric(y_hat_stream,y_stream_train.to(torch.int)).to(device)   

    return train_acc,loss  


def train(tasks):
    global memory_X, memory_y, memory_y_name,local_count,global_count,local_store,input_shape,memory_size,task_num
    global classes_so_far,full,global_priority_list,local_priority_list,memory_population_time,replay_size
    task_id_temp = 0
    

    for X,y,y_classname in tasks:
        
        if not is_lazy_training:
            task_num = task_id_temp
        
        task_id_temp+=1    

        # print("task number:",task_num)
        task_size = X.shape[0]

        for curr_batch in range(0, task_size, replay_size):
            print("till here", curr_batch+replay_size)
            Xt, yt, ynamet = X[curr_batch:curr_batch+replay_size,:], y[curr_batch:curr_batch+replay_size], y_classname[curr_batch:curr_batch+replay_size]
            print("Buffer memory",local_count)
            update_exemplars_global_counter(ynamet)        
            total_count=sum(global_count.values())
            a=1/nc    
            replay_Xt,replay_yt,replay_yname = retrieve_replaysamples(memory_X, memory_y ,memory_y_name,global_priority_list,local_count,replay_size,input_shape,minority_allocation,memory_size)            
            update_replay_counter(binarymemorysamples=replay_yt,classwisememorysamples=replay_yname)   
            replay_Xt, Xt = replay_Xt.astype('float32'), Xt.astype('float32')   
            replay_yt,yt = replay_yt.astype('float32'), yt.astype('float32')

            for epoch in range(epochs):
                train_acc,loss = ecbrs_train_epoch(Xt=Xt,yt=yt,replay_Xt=replay_Xt,replay_yt=replay_yt,a=a)        
        
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print("loss over epoch: %.4f" % (float(loss),))
        

            mem_begin=time.time()
            memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list = cbrsmemory_update(Xt,yt,ynamet,task_num,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list)
            memory_population_time+=time.time()-mem_begin


def taskwise_lazytrain():
    global test_x,test_y,task_num
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=True,bool_encode_anomaly=False,label=label,bool_create_tasks_avalanche=False)
        test_x.extend([X_test])
        test_y.extend([y_test])
        print("Training task:",task_id)
        task_num = task_id
        if task_id == int(0):
            initialize_buffermemory(tasks=tasks,mem_size=memory_size)
            update_buffermemory_counter(memorysamples=memory_y_name)
            update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks)

def evaluate_on_testset():
    global X_test,y_test
    # X_test,y_test = avalanche_tensor_to_tensor(test_data_x = X_test,test_data_y = y_test)
    # X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    # X_test = torch.from_numpy(X_test.astype(float)).to(device)
    # model.eval()
    # yhat = model(X_test.float()).detach().cpu().numpy()
    yhat = None
    model.to('cpu')
    model.eval()
    print("computing the results")
    for idx in range(0,X_test.shape[0],25000):
        idx1=idx
        idx2 = idx1+25000
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float))#.to(device)
        
        # temp = model(X_test1.float()).detach().cpu().numpy()
        temp = model(X_test1.reshape((-1,3,32,32)).float()).detach().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)



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
        input_shape,tasks,X_test,y_test,_,_ = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=True,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=False)
        initialize_buffermemory(tasks=tasks,mem_size=memory_size)
        print('Total no.of tasks', len(tasks))
        update_buffermemory_counter(memorysamples=memory_y_name)
        update_mem_samples_indexdict(memorysamples=memory_y_name)
        train(tasks=tasks)
    print("Total execution time is--- %s seconds ---" % (time.time() - start_time))
    print("Total memory population time is--- %s seconds ---" % (memory_population_time))





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    #parser.add_argument('--ds', type=str, default="svhn", metavar='S',help='dataset name')
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
