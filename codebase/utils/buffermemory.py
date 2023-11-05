import numpy as np
import random
from collections import Counter
import math
from functorch import make_functional_with_buffers, vmap, grad
import torch
from torch import linalg as LA



def retrieve_replaysamples(memory_X, memory_y ,memory_y_name,global_priority_list,local_count,replay_size,input_shape,minority_allocation,memory_size,local_store=False,ecbrs_taskaware = False,pred_diff=None):
    numbers = None
    
    if local_store:
        temp_replay_size = replay_size
        indicies = []
        tmp_local_store = {}
        for idx,class_ in enumerate(memory_y_name):
            if class_ in tmp_local_store :
                tmp_local_store[class_].append(idx)
            else:
                tmp_local_store[class_] = [idx]
        
        weight_dict = {}
        unique_classes = np.unique(memory_y_name)
        # print(unique_classes)
        weight_dict = {}
        # print(global_priority_list)
        for class_idx,_ in enumerate(unique_classes):
            _class = unique_classes[class_idx]
            if _class in list(global_priority_list.keys()) and global_priority_list[_class] == max(global_priority_list.values()) and local_count[_class] == (minority_allocation*memory_size):
                weight_dict[_class] = 1
            else:
                weight_dict[_class] = 1/local_count[_class]
        # print(weight_dict)
        # exit() 
              
        

        total = sum(weight_dict.values())
        weight_dict = {key: value / total for key, value in weight_dict.items()}     
        # print(weight_dict) 

        while temp_replay_size >0 and weight_dict:
            itemMaxValue=max(weight_dict, key=weight_dict.get)
            # print("max values is:",itemMaxValue)
            listOfKeys = list()
            # Iterate over all the items in dictionary to find keys with max value
            for key, value in weight_dict.items():
                if value == weight_dict[itemMaxValue]:
                    listOfKeys.append(key)
                    # print("key:",key,len(tmp_local_store[key]))
            # print(listOfKeys)    
            # print(temp_replay_size)    
            if len(tmp_local_store[listOfKeys[0]])>0:
                for key in listOfKeys:
                    
                    if temp_replay_size > 0 and len(tmp_local_store[key])>0:
                        indicies.extend([tmp_local_store[key].pop(0)])
                        temp_replay_size = temp_replay_size-1
            else:
                for key in listOfKeys:
                    del weight_dict[key]
                total = sum(weight_dict.values())
                weight_dict = {key: value / total for key, value in weight_dict.items()} 
        #     print(weight_dict)  
        #     print("temp replay size:",temp_replay_size)    

        # print("temp replay size:",temp_replay_size)
        numbers = indicies  
        # 
        # print("replaysamples counter:",Counter(numbers))       
    else:
        weights,probas = [],[]
        for x in range(memory_y_name.shape[0]):
            # weights.append(1/local_count[memory_y_name[x]])
            if memory_y_name[x] in list(global_priority_list.keys()) and global_priority_list[memory_y_name[x]] == max(global_priority_list.values()) and local_count[memory_y_name[x]] == (minority_allocation*memory_size):
                weights.append(1)
            else:
                weights.append(1/local_count[memory_y_name[x]])

        #appending predict differences to weight values
        
        if pred_diff is not None:
            weights = [x+y for x,y in zip(weights, pred_diff)]       

        total_weight = sum(weights)
        probas = [weight/total_weight for weight in weights] #translates class_weigths to their probabilities
        numbers = np.random.choice(range(0,memory_y_name.shape[0]), min(replay_size,memory_y_name.shape[0]), replace=False, p=probas)
    
    
    if ecbrs_taskaware:
        replay_Xt, replay_yt, replay_yname = np.zeros((min(replay_size,memory_size),input_shape)), np.zeros(min(replay_size,memory_size)), []
    else:
        # numbers = np.random.choice(range(0,memory_y_name.shape[0]), min(replay_size,memory_y_name.shape[0]), replace=False, p=probas) 
        replay_Xt, replay_yt, replay_yname = np.zeros((replay_size,input_shape)), np.zeros(replay_size), [] # Xt.shape[1] = 7, Xt.shape[2] = 10, Xt.shape[3] = 1

    for ind,rand in enumerate(numbers): #loading the samples from CBRS Memory which will replay later
      replay_Xt[ind], replay_yt[ind] = memory_X[rand], memory_y[rand] 
      replay_yname.append(memory_y_name[rand])  

    
    
    return replay_Xt,replay_yt,replay_yname  


def retrieve_MIR_replaysamples(memory_X, memory_y ,memory_y_name,local_count,replay_size,input_shape):
    weights,probas = [],[]

    for x in range(memory_y_name.shape[0]):
      weights.append(1/local_count[memory_y_name[x]])

    total_weight = sum(weights)
    probas = [weight/total_weight for weight in weights] #translates class_weigths to their probabilities

    numbers = np.random.choice(range(0,memory_y_name.shape[0]), min(replay_size,memory_y_name.shape[0]), replace=False, p=probas) 

    replay_Xt, replay_yt, replay_yname = np.zeros((replay_size,input_shape)), np.zeros(replay_size), [] # Xt.shape[1] = 7, Xt.shape[2] = 10, Xt.shape[3] = 1

    for ind,rand in enumerate(numbers): #loading the samples from CBRS Memory which will replay later
      replay_Xt[ind], replay_yt[ind] = memory_X[rand], memory_y[rand] 
      replay_yname.append(memory_y_name[rand])  

    return replay_Xt,replay_yt,replay_yname  



def cbrsmemory_update(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list):
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory noot filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,70)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                largest = set()     
                for class_ in local_count.keys():
                    if local_count[class_] == max(local_count.values()):          
                        largest.add(class_)
                        full.add(class_)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                rand = random.randint(0,len(largest_members)-1)
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
            #rand = random.randint(0,len(same_class)-1)
            #rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   





def memory_update(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list):
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory not filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                # print("ynametj",ynamet[j])
                for class_ in classes_so_far:
                    if class_ in local_count.keys():
                        #print(class_)
                        global_priority_list[class_]=global_count[class_]/sum(global_count.values())
                        local_priority_list[class_]=local_count[class_]/sum(local_count.values())

                largest = set()
                class_to_add=ynamet[j]
                list_keys = list(local_count.keys())
                list_keys = [int(x) for x in list_keys]
                global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                global_max_class = [int(x) for x in global_max_class]
                list_keys =  [x for x in list_keys if x not in  global_max_class] 
                class_to_add =  list_keys[0] if len(list_keys)>0 else ynamet[j] # fix when no data shuffling and loading initial samples for custom LSTM train
                for class_ in list_keys:
                    if local_count[class_] >= local_count[class_to_add]:# and global_priority_list[class_] != max(global_priority_list.values()):
                        class_to_add = class_        
                
          
         
                temp = min(global_priority_list.values())
                for idx,value in local_count.items():
                    if (global_priority_list[idx] >= temp and global_priority_list[idx] == max(global_priority_list.values()) and local_count[idx] > int(minority_allocation*memory_size)):# or round(global_priority_list[idx],offset) >= round(temp,offset):
                        class_to_add = idx
                        temp =global_priority_list[idx]  

                if global_priority_list[class_to_add] != max(global_priority_list.values()):
                    keys_list = list(local_count.keys())
                    keys_list = [int(x) for x in keys_list]
                    global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                    #print("global max class is",global_max_class)
                    global_max_class = [int(x) for x in global_max_class]
                    keys_list = [x for x in keys_list if x not in  global_max_class]
                    curr_max = local_count[keys_list[0]]
                    for class_ in keys_list:
                        if local_count[class_] >curr_max:# max(local_count.values()) and global_priority_list[class_] != max(global_priority_list.values()):
                            class_to_add = class_
                            curr_max = local_count[class_]              

                largest.add(class_to_add)
                full.add(class_to_add)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                rand = random.randint(0,len(largest_members)-1)
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
                #print("threshold value is",threshold)
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
                    # rand_same_class = random.randint(0,len(local_store[ynamet[j]])-1)
                    # rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   






def memory_update2(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list,model,loss_fn,device,data_reshape=None):
    model.eval()
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory not filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                # print("ynametj",ynamet[j])
                for class_ in classes_so_far:
                    if class_ in local_count.keys():
                        #print(class_)
                        global_priority_list[class_]=global_count[class_]/sum(global_count.values())
                        local_priority_list[class_]=local_count[class_]/sum(local_count.values())

                largest = set()
                class_to_add=ynamet[j]
                list_keys = list(local_count.keys())
                list_keys = [int(x) for x in list_keys]
                global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                global_max_class = [int(x) for x in global_max_class]
                list_keys =  [x for x in list_keys if x not in  global_max_class] 
                class_to_add =  list_keys[0] if len(list_keys)>0 else ynamet[j] # fix when no data shuffling and loading initial samples for custom LSTM train
                for class_ in list_keys:
                    if local_count[class_] >= local_count[class_to_add]:# and global_priority_list[class_] != max(global_priority_list.values()):
                        class_to_add = class_        
                
          
         
                temp = min(global_priority_list.values())
                for idx,value in local_count.items():
                    if (global_priority_list[idx] >= temp and global_priority_list[idx] == max(global_priority_list.values()) and local_count[idx] > int(minority_allocation*memory_size)):# or round(global_priority_list[idx],offset) >= round(temp,offset):
                        class_to_add = idx
                        temp =global_priority_list[idx]  

                if global_priority_list[class_to_add] != max(global_priority_list.values()):
                    keys_list = list(local_count.keys())
                    keys_list = [int(x) for x in keys_list]
                    global_max_class = [max(global_priority_list, key=global_priority_list.get)]
                    #print("global max class is",global_max_class)
                    global_max_class = [int(x) for x in global_max_class]
                    keys_list = [x for x in keys_list if x not in  global_max_class]
                    curr_max = local_count[keys_list[0]]
                    for class_ in keys_list:
                        if local_count[class_] >curr_max:# max(local_count.values()) and global_priority_list[class_] != max(global_priority_list.values()):
                            class_to_add = class_
                            curr_max = local_count[class_]              

                largest.add(class_to_add)
                full.add(class_to_add)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                
                def compute_loss_stateless_model (params, buffers, sample, target):
                    # if data_reshape is not None:
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = fmodel(params, buffers, batch) 
                    loss = loss_fn(predictions, targets)
                    return loss
                    
                fmodel, params, buffers = make_functional_with_buffers(model)
                ft_compute_grad = grad(compute_loss_stateless_model)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                grad_list = []
                
                for lm_idx in largest_members:
                    # if data_reshape is not None:
                    data = torch.from_numpy(memory_X[lm_idx]).reshape(data_reshape).to(device)
                    # else:
                    #     data = torch.from_numpy(memory_X[lm_idx]).to(device)

                    target = torch.as_tensor(memory_y[lm_idx]).reshape(1,1).float().to(device)
                    ft_per_sample_grad = ft_compute_sample_grad(params, buffers, data, target)
                    grad2_list =[]
                    sum_grad=0
                    for tch_len in range(len(ft_per_sample_grad)):
                        temp_grad = torch.flatten(ft_per_sample_grad[tch_len])                        
                        sum_grad = sum_grad+(LA.norm(temp_grad).item())                       

                       
                    grad_list.extend([sum_grad])

                # print(len(grad_list))
                # print(len(largest_members))
                # print(grad_list.index(min(grad_list)))
                # print(np.amin(grad_list))
                # rand = random.randint(0,len(largest_members)-1)
                rand = grad_list.index(min(grad_list))
                # rand = random.randint(0,len(largest_members)-1)
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
            #rand = random.randint(0,len(same_class)-1)
            #rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 

    model.train()
    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   

def memory_update3(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list,model,loss_fn,device):
    model.eval()
    if(task_id == 0 and memory_y_name.shape[0]==memory_size and curr_batch+replay_size>memory_y_name.shape[0]) or (task_id != 0):
        for j in range(len(ynamet)):
            #if memory not filled
            if (memory_y_name.shape[0]<memory_size):
                Xtreshape = Xt[j].reshape(1,input_shape)#.reshape(1,7,10,1)
                ytnew = np.empty(1, dtype=object)
                ytnew[0] = yt[j]
                ynametnew = np.empty(1, dtype=object)
                ynametnew[0] = ynamet[j]
                memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
                memory_y = np.concatenate((memory_y,ytnew), axis=0)
                memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
                local_count[ynamet[j]]+=1

                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(memory_y_name.shape[0])
                else:
                    local_store[ynamet[j]] = [memory_y_name.shape[0]]

            elif ynamet[j] not in full:
                # print("ynametj",ynamet[j])
                for class_ in classes_so_far:
                    if class_ in local_count.keys():
                        #print(class_)
                        global_priority_list[class_]=global_count[class_]/sum(global_count.values())
                        local_priority_list[class_]=local_count[class_]/sum(local_count.values())

                largest = set()
                class_to_add=ynamet[j]
                list_keys = list(local_count.keys())
                list_keys = [int(x) for x in list_keys]
                global_max_class = [min(global_priority_list, key=global_priority_list.get)]
                global_max_class = [int(x) for x in global_max_class]
                list_keys =  [x for x in list_keys if x not in  global_max_class] 
                class_to_add =  list_keys[0] if len(list_keys)>0 else ynamet[j] # fix when no data shuffling and loading initial samples for custom LSTM train
                for class_ in list_keys:
                    if local_count[class_] <= local_count[class_to_add]:# and global_priority_list[class_] != max(global_priority_list.values()):
                        class_to_add = class_        
                
          
         
                temp = max(global_priority_list.values())
                for idx,value in local_count.items():
                    if (global_priority_list[idx] <= temp and global_priority_list[idx] == min(global_priority_list.values()) and local_count[idx] > int(minority_allocation*memory_size)):# or round(global_priority_list[idx],offset) >= round(temp,offset):
                        class_to_add = idx
                        temp =global_priority_list[idx]  

                if global_priority_list[class_to_add] != min(global_priority_list.values()):
                    keys_list = list(local_count.keys())
                    keys_list = [int(x) for x in keys_list]
                    global_max_class = [min(global_priority_list, key=global_priority_list.get)]
                    #print("global max class is",global_max_class)
                    global_max_class = [int(x) for x in global_max_class]
                    keys_list = [x for x in keys_list if x not in  global_max_class]
                    curr_max = local_count[keys_list[0]]
                    for class_ in keys_list:
                        if local_count[class_] <curr_max:# max(local_count.values()) and global_priority_list[class_] != max(global_priority_list.values()):
                            class_to_add = class_
                            curr_max = local_count[class_]              

                largest.add(class_to_add)
                full.add(class_to_add)

                largest_members = []

                for k in local_store.keys():
                    if k in largest:
                        largest_members.extend(local_store[k])
              
          # ------------------------------------------------------------------------------------------------# 

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
                def compute_loss_stateless_model (params, buffers, sample, target):
                    batch = sample.unsqueeze(0)
                    targets = target.unsqueeze(0)

                    predictions = fmodel(params, buffers, batch) 
                    loss = loss_fn(predictions, targets)
                    return loss
                    
                fmodel, params, buffers = make_functional_with_buffers(model)
                ft_compute_grad = grad(compute_loss_stateless_model)
                ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                grad_list = []
                
                for lm_idx in largest_members:
                    data = torch.from_numpy(memory_X[lm_idx]).reshape(1,3,32,32).to(device)
                    target = torch.as_tensor(memory_y[lm_idx]).reshape(1,1).float().to(device)
                    ft_per_sample_grad = ft_compute_sample_grad(params, buffers, data, target)
                    grad2_list =[]
                    sum_grad=0
                    for tch_len in range(len(ft_per_sample_grad)):
                        temp_grad = torch.flatten(ft_per_sample_grad[tch_len])                        
                        sum_grad = sum_grad+(LA.norm(temp_grad).item())                       

                       
                    grad_list.extend([sum_grad])

                # print(len(grad_list))
                # print(len(largest_members))
                # print(grad_list.index(min(grad_list)))
                # print(np.amin(grad_list))
                #rand = random.randint(0,len(largest_members)-1)
                #if len(grad_list) > 0:
                rand = grad_list.index(min(grad_list))
                
                        
                rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
                local_count[memory_y_name[rand_largest]]-=1
                local_count[ynamet[j]]+=1

        # Update the local store  ,, at rand_largest position, we are removing  'memory_y_name[rand_largest]' class instance and adding 'ynamet[j]' class instance
                local_store[memory_y_name[rand_largest]].remove(rand_largest)
                if ynamet[j] in local_store :
                    local_store[ynamet[j]].append(rand_largest)
                else:
                    local_store[ynamet[j]] = [rand_largest]
          #local_store[ynamet[j]].append(rand_largest)

        # Replace the "largest" class member we picked earlier with the current instance
                memory_X[rand_largest] = Xt[j]
                memory_y[rand_largest] = yt[j]
                memory_y_name[rand_largest] = ynamet[j]


            else:
                threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
                u = random.uniform(0, 1)
          
                if u <= threshold:
                    rand_same_class = local_store[ynamet[j]][0]

          # We randomly pick one of the same class members (which we stored in same_class)
            #rand = random.randint(0,len(same_class)-1)
            #rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
                    memory_X[rand_same_class] = Xt[j]
                    memory_y[rand_same_class] = yt[j]
                    memory_y_name[rand_same_class] = ynamet[j] 
    model.train()                

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                   



def random_memory_update(Xt,yt,ynamet,task_id,minority_allocation,input_shape,curr_batch,replay_size,memory_size,memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list):

    indices_len = math.ceil(0.30*memory_X.shape[0]) if Xt.shape[0] > math.ceil(0.30*memory_X.shape[0]) else math.ceil(0.30*Xt.shape[0])
    memory_indices = np.random.choice(range(0,memory_X.shape[0]),size=indices_len,replace=False).tolist()
    xt_indices = np.random.choice(range(0,(Xt.shape[0])),size=indices_len,replace=False).tolist()
    for i,j in zip(memory_indices,xt_indices):
        memory_X[i] = Xt[j]
        memory_y[i] = yt[j]
        memory_y_name[i] = ynamet[j]


                

    return memory_X, memory_y, memory_y_name,local_count,global_count,local_store,classes_so_far,full,global_priority_list,local_priority_list                  


