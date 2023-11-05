from turtle import st
# from neurips_2022.CBRS.anoshift_AGEM import X_train
import torch
import numpy as np
import torch.nn as nn
import warnings
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy
from torch.optim import Adam
import avalanche
from avalanche.benchmarks.generators import tensors_benchmark
from torchvision.models import resnet101,resnet18,resnet50,googlenet,squeezenet1_0,ResNet18_Weights

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
from avalanche.training import EWC, AGEM, Naive, LwF, SynapticIntelligence, GSS_greedy,GEM,ICaRL



from utils.customdataloader import load_dataset,avalanche_tensor_to_tensor,get_inputshape,compute_total_minority_testsamples
from utils.metrics import compute_results
from utils.utils import log,create_directories,trigger_logging,set_seed,get_gpu,load_model
from utils.config.configurations import cfg
from utils.config.configurations import svhn as ds
from utils.metadata import initialize_metadata
from utils.classifiers import ICARL_FC




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
os.environ['NCCL_P2P_DISABLE']="1"




memory_population_time=0

pattern_per_exp,task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label = None,None,None,None,None,None,None,None
batch_size,device = None,None
test_x,test_y = [],[]

cl_strategy,model,opt,loss_fn,train_acc_metric,learning_rate,is_lazy_training = None,None,None,None,None,None,None
nc = 0
no_tasks = 0
bool_create_tasks_avalanche = True
warnings.filterwarnings("ignore")



def load_metadata(): 
    log('loading meta data')   
    global task_order,class_ids,minorityclass_ids,pth,tasks_list,task2_list,label,no_tasks,pattern_per_exp,cl_strategy
    global replay_size,memory_size,minority_allocation,epochs,batch_size,device,learning_rate,is_lazy_training
    print(avalanche.__version__)
    # set_seed(125)
    get_gpu()
    label = ds.label
    cfg.avalanche_dir = True    
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
    # device = torch.device("cpu")
    print(device)
    learning_rate = ds.learning_rate
    no_tasks = ds.no_tasks
    pattern_per_exp = ds.pattern_per_exp
    is_lazy_training = ds.is_lazy_training
    compute_total_minority_testsamples(pth=pth,dataset_label=label,minorityclass_ids=minorityclass_ids,no_tasks=no_tasks)
    load_model_metadata()
    strategy = set_cl_strategy_name(3)
    create_directories(label)
    trigger_logging(label=label)
    cl_strategy = get_cl_strategy(strategy)
    
    

def load_model_metadata():
    log("loading model parameter")
    global model,opt,loss_fn,train_acc_metric
    model = load_model(label=label,inputsize=get_inputshape(pth,class_ids))
    # print(model)
    model = model.to(device)
    # model.train()
    # opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9, nesterov=True, weight_decay=0.0001)
    loss_fn = torch.nn.BCELoss()
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # loss_fn = torch.nn.functional.cross_entropy()
    train_acc_metric = Accuracy().to(device)




def create_avalanche_scenario(tasks_dataset,task_labels):
    
    generic_scenario = tensors_benchmark(train_tensors = tasks_dataset[0],
                                           test_tensors = tasks_dataset[1],#tasks_dataset[1],
                                           task_labels = task_labels,
                                        )

    return generic_scenario  

 

def train_a_lazytask(train_scenario,task_label,task_id):
    print(task_label)

    for idx in range(0,len(train_scenario[0])):
        temp = train_scenario[0][idx][0].reshape((-1,3,32,32))
        # temp2 = train_scenario[0][idx][1]
        # train_scenario[0][idx][1] = temp2.type(torch.LongTensor).reshape(-1,1)
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
    #elif strategy_id == 5:
     #   cfg.clstrategy = "icarl"     
    return strategy_id    


def get_cl_strategy(strategy_id):
    strategy = None
    global input_shape,model

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
                  input_size=[3,32,32],
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
        input_shape,train_scenario = load_dataset(pth,task_class_ids,task_minorityclass_ids,tasks_list,task2_list,[task,],bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=True)
        test_x.extend([train_scenario[2][0]])
        test_y.extend([train_scenario[3][0]])
        print("Training task:",task_id)
        train_a_lazytask(train_scenario,task_label = [0,],task_id=task_id)
            # train(train_scenario=train_scenario,cl_strategy=get_cl_strategy()) 

        



       

            
def evaluate_on_testset(X_test,y_test):

    X_test,y_test = avalanche_tensor_to_tensor(test_data_x = X_test,test_data_y = y_test)
    # X_test,y_test = get_balanced_testset(X=X_test,y=y_test)
    # X_test = torch.from_numpy(X_test.astype(float)).to(device)
    # model.eval()
    # yhat = model(X_test.float()).detach().cpu().numpy()
    yhat = None
    # model.to('cpu')
    model.eval()
    print("computing the results")
    offset = 250
    for idx in range(0,X_test.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X_test[idx1:idx2,:].astype(float)).to(device)
        
        # temp = model(X_test1.float()).detach().cpu().numpy()
        temp = model(X_test1.reshape((-1,3,32,32)).float()).detach().cpu().numpy()
        if idx1==0:
            yhat = temp
        else:
            yhat = np.append(yhat, np.array(temp), axis=0)  
    compute_results(y_test,yhat)
    # print("test sample counters are",Counter(y_test))
    




def start_execution():
    global input_shape,label,is_lazy_training,cl_strategy,test_x,test_y
    start_time=time.time()
    # load_model_metadata()
    load_metadata()
    if is_lazy_training:
        taskwise_lazytrain()
        
    else:
        input_shape,train_scenario = load_dataset(pth,class_ids,minorityclass_ids,tasks_list,task2_list,task_order,bool_encode_benign=False,bool_encode_anomaly=True,label=label,bool_create_tasks_avalanche=True)    
        test_x,test_y = train_scenario[2],train_scenario[3]
        train_scenario = create_avalanche_scenario(tasks_dataset=train_scenario,task_labels= [0 for key in range(0,no_tasks)])# using same variable to avoid multiple copies in the memory
        train(train_scenario=train_scenario,cl_strategy=cl_strategy)
       
    
    print("total training time is--- %s seconds ---" % (time.time() - start_time)) 
    evaluate_on_testset(test_x,test_y)
        
        
        
        
          
           
        
    
    
    





if __name__ == "__main__":
    start_execution()
    
    
