from statistics import mode
from turtle import st

# from neurips_2022.CBRS.anoshift_AGEM import X_train
import torch
import numpy as np
import warnings
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy
from torch.optim import Adam
import avalanche
from avalanche.benchmarks.generators import tensors_benchmark

from avalanche.benchmarks.generators import tensor_scenario
from avalanche.benchmarks.scenarios.classification_scenario import GenericCLScenario
from avalanche.evaluation.metrics import (
    ExperienceForgetting,
    StreamConfusionMatrix,
    accuracy_metrics,
    loss_metrics,
    
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import EWC, AGEM, Naive, LwF, SynapticIntelligence, GSS_greedy,GEM

import json

from utils.customdataloader import load_dataset,avalanche_tensor_to_tensor,get_inputshape,compute_total_minority_testsamples,load_teset,get_balanced_testset
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model
from utils.config.configurations import cfg
from utils.config.configurations import clear100 as ds
from utils.metadata import initialize_metadata




import time
import random
from collections import Counter
from sys import getsizeof as size
from typing import (
    Sequence,
    Union,
    Any,
    Tuple,
    Dict,
    Optional,
    Iterable,
    NamedTuple,
)
from memory_profiler import memory_usage

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"








memory_population_time=0

pattern_per_exp,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,pth_testset,testset_class_ids = None,None,None,None,None,None,None,None,None,None
batch_size,device = None,None
test_x,test_y = [],[]

cl_strategy,model,opt,loss_fn,train_acc_metric,learning_rate,is_lazy_training = None,None,None,None,None,None,None
nc = 0
no_tasks = 0
bool_create_tasks_avalanche = True
warnings.filterwarnings("ignore")



def load_metadata(cls): 
    log('loading meta data')   
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,no_tasks,pattern_per_exp,cl_strategy,pth_testset,testset_class_ids
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,learning_rate,is_lazy_training,loss_fn
    print(avalanche.__version__)
    # set_seed(125)
    #get_gpu()
    label = ds.label
    cfg.avalanche_dir = True    
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
    tasks_list = temp_dict['tasks_list']
    task2_list = temp_dict['task2_list']
    replay_size = ds.replay_size
    memory_size = ds.mem_size
    minority_allocation = ds.minority_allocation
    epochs = ds.n_epochs
    batch_size = ds.batch_size
    device = cfg.device
    # device = torch.device("cpu")
    print(device)
    learning_rate = ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata()
    strategy = set_cl_strategy_name(cls)
    create_directories(label)
    trigger_logging(label=label)
    cl_strategy = get_cl_strategy(strategy)
    # if strategy == 2:
    #     task_order = temp_dict['task_order2']
        # loss_fn = torch.nn.BCEWithLogitsLoss()
    
    

def load_model_metadata():
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    # print(model)
    model = model.to(device)
    # model.train()
    # opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    # opt = torch.optim.SGD(model.parameters(),momentum=0.9, lr=0.01, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()
    train_acc_metric = Accuracy().to(device)




def create_avalanche_scenario(tasks_dataset,task_labels):
    
    generic_scenario = tensors_benchmark(train_tensors = tasks_dataset[0],
                                           test_tensors = tasks_dataset[1],#tasks_dataset[1],
                                           task_labels = task_labels,
                                        )

    return generic_scenario  

 

def train_a_lazytask(train_scenario,task_label,task_id):

    for idx in range(0,len(train_scenario[0])):
        temp = train_scenario[0][idx][0].reshape((-1,3,224,224))
        train_scenario[0][idx][0] = temp

        # train_scenario[0] = train_scenario[0].reshape((-1,3,32,32))
    generic_scenario = create_avalanche_scenario([train_scenario[0],train_scenario[1]],task_labels=task_label)
    for task_number, experience in enumerate(generic_scenario.train_stream):
            res = cl_strategy.train(experience)
            
    if ds.enable_checkpoint:
        checkpoint_location = str(cfg.outputdir) + '/models' +'/task_'+str(task_id)+ '.th'
        # print("location:",checkpoint_location)
        torch.save({'epoch': 5, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location)                  


def evaluation_plugin():
    # log to text file
    text_logger = TextLogger(open(f"{cfg.outputdir}/logs/log.txt", "w+"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin( 
                                    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                                    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                                    ExperienceForgetting(),
                                    StreamConfusionMatrix(num_classes=2, save_image=False),
                                    loggers=[interactive_logger, text_logger],
                                   )

    return eval_plugin    


def set_cl_strategy_name(strategy_id):
    if strategy_id == 1:
        cfg.clstrategy = "GEM"            
    elif strategy_id == 0:
        cfg.clstrategy = "EWC"
    elif strategy_id == 2:
        cfg.clstrategy = "AGEM" 
    elif strategy_id == 3:
        cfg.clstrategy = "GSS-greedy"   
    elif strategy_id == 4:
        cfg.clstrategy = "SI"    
    return strategy_id    


def get_cl_strategy(strategy_id):
    strategy = None
    global input_shape

    if cfg.clstrategy == "GEM":
        strategy = GEM(model,
                  optimizer = opt,
                  patterns_per_exp = pattern_per_exp,
                  criterion = loss_fn,
                  train_mb_size = batch_size,
                  train_epochs = epochs,
                  eval_mb_size = 128,
                  evaluator = evaluation_plugin(),
                  device = device,
                  )  
                   
    elif cfg.clstrategy == "EWC":
        strategy = EWC(
           model,
           optimizer=opt,
           ewc_lambda=0.001,
           criterion=loss_fn,
           train_mb_size=batch_size,
           train_epochs=epochs,
           eval_mb_size=128,
           evaluator=evaluation_plugin(),
           device=device,
       )
        
    elif cfg.clstrategy =="AGEM" :
        strategy = AGEM(model,
                  optimizer = opt,
                  patterns_per_exp = pattern_per_exp,
                  criterion = loss_fn,
                  train_mb_size = batch_size,
                  train_epochs = epochs,
                  eval_mb_size = 128,
                  evaluator = evaluation_plugin(),
                  device = device,
                  )  

    elif cfg.clstrategy == "GSS-greedy":
        print("shape is:",get_inputshape(pth,class_ids))
        strategy = GSS_greedy( model,
                   optimizer=opt,
                   mem_size=replay_size,
                  criterion=loss_fn,
                  train_mb_size=batch_size,
                  train_epochs=epochs,
                  eval_mb_size=128,
                  input_size=[3,224,224],
                  evaluator=evaluation_plugin(),
                 device=device,
                )  

    elif cfg.clstrategy == "SI":
        strategy = SynapticIntelligence(
           model,
           optimizer=opt,
           si_lambda=0.1,
        #    alpha=1, temperature=2,
           criterion=loss_fn,
           train_mb_size=batch_size,
           train_epochs=epochs,
           eval_mb_size=128,
           evaluator=evaluation_plugin(),
           device=device,
       )                          

    print(strategy)              
                                                   

    return strategy          

def train(train_scenario,cl_strategy):
    for task_number, experience in enumerate(train_scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        # print(type(experience))
        res = cl_strategy.train(experience)
        if ds.enable_checkpoint:
            checkpoint_location = str(cfg.outputdir) + '/models' +'/task_'+str(task_number)+ '.th'
            # print("location:",checkpoint_location)
            torch.save({'epoch': 5, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, checkpoint_location) 




def taskwise_lazytrain():
    global test_x,test_y
    for task_id,task in enumerate(task_order):
        task_class_ids = []
        task_minorityclass_ids = []
        for class_ in task:
            task_class_ids.extend([class_])
            if class_ in minorityclass_ids:
                task_minorityclass_ids.extend([class_])
        print("loading task:",task_id)     
        input_shape,train_scenario = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=True,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=True,load_whole_train_data=False)
        test_x.extend([train_scenario[2][0]])
        test_y.extend([train_scenario[3][0]])
        print("Training task:",task_id)
        train_a_lazytask(train_scenario,task_label = [0,],task_id=task_id)
            # train(train_scenario=train_scenario,cl_strategy=get_cl_strategy()) 

        



       

            
def evaluate_on_testset(X_test,y_test):
    pth_testset = None
    
    if pth_testset is not None:
        X_test,y_test = load_teset(pth_testset,testset_class_ids,label)
        #X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    else:
        X_test,y_test = avalanche_tensor_to_tensor(test_data_x = X_test,test_data_y = y_test)
    # X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    # X_test = torch.from_numpy(X_test.astype(float)).to(device)
    # model.eval()
    # yhat = model(X_test.float()).detach().cpu().numpy()
    yhat = None
    # model.to('cpu')
    model.eval()
    print("computing the results")
    offset = 25
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float))#.to(device)
        
        # temp = model(X_test1.float()).detach().cpu().numpy()
        X_test1 = X_test1.to(device)
        temp = model(X_test1.reshape((-1,3,224,224)).float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    return compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))
    




def start_execution(cls):
    global input_shape,label,is_lazy_training,cl_strategy,test_x,test_y
    start_time=time.time()
    # load_model_metadata()
    load_metadata(cls)
    if is_lazy_training:
        taskwise_lazytrain()
        
    else:
        input_shape,train_scenario = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=True)    
        test_x,test_y = train_scenario[2],train_scenario[3]
        train_scenario = create_avalanche_scenario(tasks_dataset=train_scenario,task_labels= [0 for key in range(0,no_tasks)])# using same variable to avoid multiple copies in the memory
        train(train_scenario=train_scenario,cl_strategy=cl_strategy)
       
    
    print("total training time is--- %s seconds ---" % (time.time() - start_time)) 
    return evaluate_on_testset(test_x,test_y)
        
        
        
        
          
           
        
    
    
    




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    #parser.add_argument('--ds', type=str, default="cifar100", metavar='S',help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',help='gpu id (default: 0)')
    parser.add_argument('--cls', type=int, default=0, metavar='S',help='cl strategy (default: 0)')
    parser.add_argument('--filename', type=str,default="temp", metavar='S',help='json file name')
    s_time = time.time()
    args = parser.parse_args()
    get_gpu(args.gpu)
    auc_result= {}
    #start_execution(args.cls)
    #e_time = time.time()-s_time
    #evaluate_on_testset()
    with open(args.filename, 'w') as fp:
         test_set_results = start_execution(args.cls)
         e_time = time.time()-s_time
         test_set_results.extend([e_time,memory_population_time])
         auc_result[str(args.seed)] = test_set_results
         json.dump(auc_result, fp) 

    
    

