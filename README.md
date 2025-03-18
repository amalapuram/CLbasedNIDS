# **Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection**.
## Our paper Accepted to 37th Conference on Advances in Neural Information Processing Systems (NeurIPS 2023)
## Paper on Open review https://openreview.net/forum?id=yGLokEhdh9

Intrusion detection is a form of anomalous activity detection in communication network traffic. Continual learning (CL) approaches to the intrusion detection task accumulate old knowledge while adapting to the latest threat knowledge. Previous works have shown the effectiveness of memory replay-based CL approaches for this task. In this work, we present two novel contributions to improve the performance of CL-based network intrusion detection in the context of class imbalance and scalability. First, we extend class balancing reservoir sampling (CBRS), a memory-based CL method, to address the problems of severe class imbalance for large datasets. Second, we propose a novel approach titled perturbation assistance for parameter approximation (PAPA) based on the Gaussian mixture model to reduce the number of _virtual stochastic gradient descent (SGD) parameter_ computations needed to discover maximally interfering samples for CL. We demonstrate that the proposed approaches perform remarkably better than the baselines on standard intrusion detection benchmarks created over shorter periods (KDDCUP'99, NSL-KDD, CICIDS-2017/2018, UNSW-NB15, and CTU-13) and a longer period with distribution shift (AnoShift). We also validated proposed approaches on standard continual learning benchmarks (SVHN, CIFAR-10/100, and CLEAR-10/100) and anomaly detection benchmarks (SMAP, SMD, and MSL). Further, the proposed PAPA approach significantly lowers the number of virtual SGD update operations, thus resulting in training time savings in the range of 12 to 40% compared to the maximally interfered samples retrieval algorithm.


## Setting the environment 

We recommend you create a  Python virtual environment and follow the instructions below
1. We provided the list of the necessary pre-requisites library to install in the requirements.txt
2. use  the "pip install -r requirements.txt" command to create the libraries

## Datasets
The preprocessed datasets can be downloaded from the [link](https://drive.google.com/drive/folders/1tbpLrPMOCXaKWzU97RXhRzaaqdpmhUod?usp=sharing)
<!--(https://iith-my.sharepoint.com/:f:/g/personal/tbr_iith_ac_in/EjEONoT1ZupLlZS_dEHhticBZnuR5tQa8Cl96568UqTDgg?e=lYshEM)-->
The shared link contains preprocessed datasets (Network Intrusion Detection, Computer Vision, and Anomaly Detection) used in our work. Specifically, computer vision datasets are processed in such a way that a class imbalance ratio (Attack: Benign) of **1:100** is maintained per task. For more information on how datasets are preprocessed and how tasks are created, we encourage reading sections **A.4, A.5, and A.6** of the supplementary material of our paper available at [link](https://openreview.net/attachment?id=yGLokEhdh9&name=supplementary_material)
## Extracting datasets files

1. unzip the preprocessed datasets.zip files
2. copy each dataset folder into the "datasets" directory in the **codebase** directory

##  Mapping the dataset files to code.
In case you face any issues while reading the dataset files, please set the appropriate dataset file path in the "metadata.py" file located in the codebase/utils directory.

For example, to set the correct dataset file path for the cifar10 dataset, configure the "path" key in  the metadata_dict_cifar10 dictionary in the metadata.py file.


## Running the code

To run the code, run the file with the main.py suffix. For example, to run ECBRS on the AnoShift dataset, use the following command.

1. python anoshift_ecbrs_main.py --gpu=GPU id --lr=learning rate(float) --w_d=weight decay(float)


To run baselines implemented in the Avalanche, you must run the file containing the suffix avalanche. For example, to run Anoshift, use the following command.

1. python anoshift_avalanche_main.py --gpu=GPU id --lr=learning rate(float) --w_d=weight decay(float) --cls=continual learning strategy (integer)
2. The continual learning strategy IDs for different algorithms are

| 0             | 1           | 2  |   3  | 4  |
| ------------- |:-------------:| -----:|-----:|-----:|
| EWC| GEM| A-GEM | GSS-greedy  | SI |


    
 ## Hyperparameter tuning

 To change the Hyperparameters, you need to make the changes in the configurations.py file located in codebase/utils/config/
 for example, to change the batch size of the cifar10 experiments, you need to change the line in configurations.py 

" root.cifar10.batch_size = 128"  




# Citation
Please cite this work in case you are referring to our work
```
@inproceedings{
amalapuram2023augmented,
title={Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection},
author={Suresh kumar Amalapuram and Sumohana S. Channappayya and Bheemarjuna Tamma},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=yGLokEhdh9}
}
```


