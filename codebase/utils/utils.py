
from pickle import TRUE
import torch
import numpy as np
from torch.nn.utils import prune
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from utils.otdd.ot_distance import compute_ot_distance
import torchvision

import random
import logging
# from imp import reload
# reload(logging)
import os
import time
import pprint

from utils.config.configurations import cfg
from utils.classifiers import *
from utils.resnet import ResNet34,ResNet50,ResNet18,ResNetCLEAR,ResNetCLEAR50,ResNetCLEAR101
from torchvision.models import resnet101,resnet18,resnet50,googlenet,squeezenet1_0,ResNet18_Weights

def create_directories(label):
    output_root_dir = cfg.root_outputdir
    cl_strategy = cfg.clstrategy
    param_weights_dir_MIR = cfg.param_weights_dir_MIR
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    if not os.path.exists(param_weights_dir_MIR):
        os.makedirs(param_weights_dir_MIR)    
    timestamp = time.strftime("%d_%b_%H_%M_%S")  
    
    output_dir = output_root_dir +'/'+label+'/'+str(cl_strategy)+'/'+timestamp 
    if cfg.avalanche_dir:
        output_dir = output_root_dir +'/'+label+'/'+'avalanche'+'/'+str(cl_strategy)+'/'+timestamp
    cfg.outputdir = output_dir
    cfg.timestamp = timestamp 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/models')
        os.makedirs(output_dir + '/encoded_models')
        os.makedirs(output_dir + '/pickles')
        os.makedirs(output_dir + '/logs')
        os.makedirs(output_dir + '/plots')
        os.makedirs(output_dir + '/weights')

        # os.makedirs(output_dir + '')


def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)             




def trigger_logging(label):
    output_root_dir = cfg.root_outputdir
    log_dir = output_root_dir+'/'+label+'/'+cfg.timestamp+'/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("time stamp is:",cfg.timestamp)    

    logging.basicConfig(filename=log_dir + '/'+cfg.timestamp + '.log', level=logging.DEBUG,force=True,
                        format='%(levelname)s:\t%(message)s')

    # log(pprint.pformat(cfg))    








def set_seed(seed):
    cfg.seed = seed
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

def get_gpu(id):
    #gpu_list = cfg.gpu_ids.split(',')
    #print("gpu list",gpu_list)
    #gpus = [int(iter) for iter in gpu_list]
    cfg.device = torch.device('cuda:' + str(id))
    #cfg.device = torch.device('cuda:' + str(gpus[0])) 


model_path={'byol_imagenet':'./clear10/pretrain_weights/features/byol_imagenet/state_dict.pth.tar',
			'imagenet':	'./clear10/pretrain_weights/features/imagenet/state_dict.pth.tar',
			'moco_b0':'./clear10/pretrain_weights/features/moco_b0/state_dict.pth.tar',
			'moco_imagenet':'./clear10/pretrain_weights/features/moco_imagenet/state_dict.pth.tar'
            }   

def compute_cosine_sim(X,device):
    avg_cos_sim_vec = [0] * X.shape[0]
    threshold = 1000
    for idx in range(0,X.shape[0]):
        if X.shape[0] < threshold:
            sim = cosine_similarity(torch.from_numpy(X),torch.from_numpy(X[idx,:])).detach().cpu().numpy() 
        else:
            indicies = np.random.choice(X.shape[0],size = threshold,replace=False) 
            sim = cosine_similarity(torch.from_numpy(X[indicies,:]),torch.from_numpy(X[idx,:])).detach().cpu().numpy()    
        avg_cos_sim_vec[idx] = 1-np.average(sim)
        # print("Cosine si is",avg_cos_sim_vec[idx])

    return avg_cos_sim_vec    




def obtain_grad_vector(model,numpy_array):

    temp_list = []
    for param in model.parameters():
        temp_list.append(param.grad.view(-1))
    grads = torch.cat(temp_list).cpu().numpy().reshape(1,-1)
    
    if numpy_array is None:
        numpy_array = grads
    else:
        numpy_array = np.concatenate((numpy_array,grads), axis=0)    
 
    return numpy_array


def plot_cosine_sim(array1,array2,dir):
    cos_array = []
    # cos_array = cosine_similarity(torch.from_numpy(array1),torch.from_numpy(array2)).detach().cpu().numpy()
    for idx in range(0,array1.shape[0]):
        cos_sim = dot(array1[idx,:], array2[idx,:])/((norm(array1[idx,:])*norm(array2[idx,:])))
        cos_array.append(cos_sim)
        # print(cos_sim)
    os.makedirs(dir,exist_ok=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('on')
    # matplotlib.rcParams.update({'font.size': 18})
    plt.legend(prop={'size': 18})
    plt.title("Cosine Similarity bw gradients") 
    plt.xlabel("Batch num")
    # plt.yticks(np.arange(-0.002, 1, 0.02))
    plt.ylabel("Cosine similarity")
    plt.figure(figsize=(10,6))
    
    plt.plot(range(0,len(cos_array)), cos_array, color ="red")
    plt.savefig(dir+'/'+'cosine_sim_grads.pdf')
    # plt.show()


def check_grad_exist(model):

    for param in model.parameters():
        if param.grad is None:
            return False 


    return True



def compute_otdd_bwtasks(task1,task2,classlabel):
    
    transform=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Grayscale(3),torchvision.transforms.ToTensor()])
    transform2=torchvision.transforms.Compose([torchvision.transforms.Resize([28,28])])

    for a,b,_ in task1:
        x1,y1 = a,b
        
    for a,b,_ in task2:
        x2,y2 = a,b  
    
    # for t1,t2,_,t3,t4,_ in zip((task1,task2)):
    #         x1,y1,x2,y2 = t1,t2,t3,t4
    
    if classlabel == "cifar10":
        x1,y1,x2,y2 = x1.reshape(-1,3,32,32),y1,x2.reshape(-1,3,32,32),y2
        return compute_ot_distance(x1,y1,x2,y2,device="cpu",bool_feature_cost="mnist")
    
    elif classlabel == "mnist_cifar10":
        x_temp = np.zeros((x1.shape[0],3,28,28))
        x_temp2 = np.zeros((x2.shape[0],3,28,28))
        for i in range(0,x1.shape[0]):  
            if  x1[i,:].shape[0] == 784:
                x_temp[i,:,:,:] = transform((x1[i,:].reshape(28,28,1))).numpy()
            else:
                x_temp[i,:,:,:] = transform2(torch.from_numpy(x1[i,:].reshape(3,32,32))).numpy() 

        for i in range(0,x2.shape[0]):  
            if x2[i,:].shape[0] != 784:    
                x_temp2[i,:,:,:] = transform2(torch.from_numpy(x2[i,:].reshape(3,32,32))).numpy()  
            else:
                x_temp2[i,:,:,:] = transform((x2[i,:].reshape(28,28,1))).numpy()

        return compute_ot_distance(x_temp,y1,x_temp2,y2,device="cpu",bool_feature_cost="mnist_cifar10")  
    
    elif classlabel == "mnist":
        x_temp = np.zeros((x1.shape[0],3,28,28))
        x_temp2 = np.zeros((x2.shape[0],3,28,28))
        for i in range(0,x1.shape[0]):  
            x_temp[i,:,:,:] = transform((x1[i,:].reshape(28,28,1))).numpy()
            
        for i in range(0,x2.shape[0]):  
            x_temp2[i,:,:,:] = transform((x2[i,:].reshape(28,28,1))).numpy()

        return compute_ot_distance(x_temp,y1,x_temp2,y2,device="cpu",bool_feature_cost="mnist")    







def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model             

def extract_features_with_resnet18(X,device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    offset = 100
    for idx in range(0,X.shape[0],offset):
        idx1=idx
        idx2 = idx1+offset
        X_test1 = torch.from_numpy(X[idx1:idx2,:]).to(device)#.astype(float))#.to(device)
        temp = feature_extractor(X_test1.float()).detach().cpu().numpy() 
        X_test1.detach()
        del X_test1
        if idx1==0:
            X_features = temp
        else:
             X_features = np.append( X_features, np.array(temp), axis=0)
    with torch.no_grad():
        # temp_model.detach()
        torch.cuda.empty_cache()
        del feature_extractor
    return  X_features     






def load_model_clear10(state_dict_path):
    # model=resnet50(pretrained=False)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    n_inputs = model.fc.in_features
    model.fc=torch.nn.Identity()
    # state_dict=torch.load(state_dict_path)
    # model.load_state_dict(state_dict)
    # for p in model.parameters():
    #     p.requires_grad= False
    
    layers = []
    layers.append(nn.Linear(n_inputs,1))
    layers.append(nn.Sigmoid())   
    model.fc = nn.Sequential(*layers)
	
    return model  

def load_googlenetmodel_clear10(state_dict_path):
    # model=resnet50(pretrained=False)
    model = googlenet(pretrained=True)
    n_inputs = model.fc.in_features
    model.fc=torch.nn.Identity()
    # state_dict=torch.load(state_dict_path)
    # model.load_state_dict(state_dict)
    # for p in model.parameters():
    #     p.requires_grad= False
    
    layers = []
    layers.append(nn.Linear(n_inputs,1))
    layers.append(nn.Sigmoid())   
    model.fc = nn.Sequential(*layers)
	#model.fc=torch.nn.Identity()
    
    
	# model.eval()
    return model  
    
def load_squeezenetmodel_clear10(state_dict_path):
    # model=resnet50(pretrained=False)
    model = squeezenet1_0(pretrained=True)
    model.num_classes = 1
    final_conv = nn.Conv2d(512, model.num_classes, kernel_size=1)
    model.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.Sigmoid(), nn.AdaptiveAvgPool2d((1, 1))
        )
# change the internal num_classes variable rather than redefining the forward pass
    
    # state_dict=torch.load(state_dict_path)
    # model.load_state_dict(state_dict)
    # for p in model.parameters():
    #     p.requires_grad= False
    
    
    
    
	# model.eval()
    return model             




def load_model(label,inputsize):
    model = None
    if label in ['cicids2017','ctu13']:
        model = CICIDS2017_FC(inputsize=inputsize)
    elif label ==  'cicids2018':
        model = CICIDS2018_FC(inputsize=inputsize)
    elif label == 'unswnb15':
        model = UNSWNB15_FC(inputsize=inputsize)  
    elif label == 'anoshift_subset':
        model = ANOSHIFT_FC(inputsize=inputsize)
    elif label == 'mnist':
        model = MNIST_FC(inputsize=inputsize)  
    elif label == 'svhn':
        # model =  SVHN_FC(inputsize=inputsize)   
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        # layers.append(nn.Linear(n_inputs,100))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(100,50))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(50,2))
        # layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
    elif label == "mnist_cifar10":
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())
        #layers.append(torch.nn.Softmax(dim=1))
        # layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
    elif label in ['cifar100','cifar100_large_benign']:
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)    
    elif label == 'clear100':
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
    elif label == 'cifar10':
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        # layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)
     
    elif label in ['kddcup99','nslkdd']:
        model = KDDCUP99_FC(inputsize=inputsize) 
    elif label == 'clear10':        
        model = resnet18(pretrained=True)
        n_inputs = model.fc.in_features
        layers = []
        layers.append(nn.Linear(n_inputs,100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100,50))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(50,1))
        layers.append(nn.Sigmoid())
        # layers.append(torch.nn.Softmax(dim=1))
        # layers.append(nn.Linear(n_inputs,1))
        # # layers.append(nn.ReLU())
        # layers.append(nn.Sigmoid())   
        model.fc = nn.Sequential(*layers)

        # model = ResNetCLEAR()
    elif label in ['MSL','SMD','SMAP']:
        model=SMAP_FC(inputsize=inputsize)   
    else:
        model = CIDDS_FC(inputsize=inputsize)   

    return model    


def load_LSTM(input_size, hidden_size, num_layers,output_size):
    return custom_LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size)   

def load_weightsFC(inputsize):
    return Weights_FC(inputsize)        


