import numpy as np
from easydict import EasyDict as edict

root = edict()
cfg = root


root.gpu_ids = '0,1,5,1,0,3,1,3,7,6,1,0,1,0'
root.device = None
root.timestamp = None
root.root_outputdir = 'output'
root.param_weights_dir_MIR = 'output/weights/MIR/'
root.outputdir =None
root.avalanche_dir = False



root.seed = 25

#cicids2017 hyperparameters
root.cicids2017 = edict()
cicids2017 = root.cicids2017

root.cicids2017.no_tasks = 10
root.cicids2017.mem_size = 13334
root.cicids2017.replay_size = 10000
root.cicids2017.n_epochs = 5
root.cicids2017.minority_allocation = 0.1
root.cicids2017.batch_size = 1024
root.cicids2017.learning_rate = 0.001
root.cicids2017.taskaware_minority_allocation = 0.1
root.cicids2017.label = 'cicids2017'
root.cicids2017.enable_checkpoint = False
root.cicids2017.taskaware_ecbrs = False
root.cicids2017.train_data_lstm = 0.5
root.cicids2017.num_mini_tasks = 200
root.cicids2017.lstm_epochs = 3
root.cicids2017.interim_model_epochs = 1
root.cicids2017.lstm_hidden_size = 10
root.cicids2017.param_weights_queue_length = 10
root.cicids2017.pattern_per_exp = 50
root.cicids2017.is_lazy_training = True
root.cicids2017.clstrategy = None
root.cicids2017.store_weights = False
root.cicids2017.train_lstm = True
root.cicids2017.image_resolution = None
root.cicids2017.store_grads = False

root.ctu13 = edict()
ctu13 = root.ctu13

root.ctu13.no_tasks = 5
root.ctu13.mem_size = 1500
root.ctu13.replay_size = 1000
root.ctu13.n_epochs = 5
root.ctu13.minority_allocation = 0.1
root.ctu13.batch_size = 1024
root.ctu13.learning_rate = 0.001
root.ctu13.taskaware_minority_allocation = 0.1
root.ctu13.label = 'ctu13'
root.ctu13.enable_checkpoint = False
root.ctu13.taskaware_ecbrs = False
root.ctu13.train_data_lstm = 0.5
root.ctu13.num_mini_tasks = 200
root.ctu13.lstm_epochs = 3
root.ctu13.interim_model_epochs = 1
root.ctu13.lstm_hidden_size = 10
root.ctu13.param_weights_queue_length = 10
root.ctu13.pattern_per_exp = 50
root.ctu13.is_lazy_training = True
root.ctu13.clstrategy = None
root.ctu13.store_weights = False
root.ctu13.train_lstm = True
root.ctu13.image_resolution = None
root.ctu13.store_grads = False


#cicids2018 hyperparameters
root.cicids2018 = edict()
cicids2018 = root.cicids2018

root.cicids2018.no_tasks = 10
root.cicids2018.mem_size = 13334
root.cicids2018.replay_size =10000
root.cicids2018.n_epochs = 5
root.cicids2018.minority_allocation = 0.1#0.0
root.cicids2018.batch_size = 1024
root.cicids2018.learning_rate = 0.001
root.cicids2018.taskaware_minority_allocation = 0.1
root.cicids2018.label = 'cicids2018'
root.cicids2018.enable_checkpoint = True
root.cicids2018.taskaware_ecbrs = False
root.cicids2018.train_data_lstm = 0.5
root.cicids2018.num_mini_tasks = 200
root.cicids2018.lstm_epochs = 3
root.cicids2018.interim_model_epochs = 1
root.cicids2018.lstm_hidden_size = 10
root.cicids2018.param_weights_queue_length = 10
root.cicids2018.pattern_per_exp = 50
root.cicids2018.is_lazy_training = True
root.cicids2018.clstrategy = None
root.cicids2018.store_weights = False
root.cicids2018.train_lstm = True
root.cicids2018.store_grads = False
root.cicids2018.image_resolution = None

root.anoshift = edict()
anoshift= root.anoshift

root.anoshift.no_tasks = 10
root.anoshift.mem_size = 5000#13333
root.anoshift.replay_size = 3750#10000
root.anoshift.n_epochs = 5
root.anoshift.minority_allocation = 0.1
root.anoshift.batch_size = 1024
root.anoshift.learning_rate = 0.001
root.anoshift.label = 'anoshift'
root.anoshift.enable_checkpoint = False
root.anoshift.taskaware_ecbrs = False
root.anoshift.train_data_lstm = 0.1
root.anoshift.num_mini_tasks = 20
root.anoshift.lstm_epochs = 5
root.anoshift.interim_model_epochs = 5
root.anoshift.lstm_hidden_size = 10
root.anoshift.param_weights_queue_length = 5
root.anoshift.checkpoint_file = '/home/suresh/suresh/AnomalyDetection/IDS18/DILCBRS_Mem_replay_list/neurips_2022/CBRS/pytorch/output/anoshift/22_Sep_01_11_41/models/task_10.th'
root.anoshift.pattern_per_exp = 500
root.anoshift.is_lazy_training = True
root.anoshift.clstrategy = None
root.anoshift.image_resolution = None

root.mnist_cifar10 = edict()
mnist_cifar10= root.mnist_cifar10

root.mnist_cifar10.no_tasks = 9
root.mnist_cifar10.mem_size = 500#2000
root.mnist_cifar10.replay_size = 375#1500
root.mnist_cifar10.n_epochs = 5
root.mnist_cifar10.minority_allocation = 0.2#0.9#0.3#0.1
root.mnist_cifar10.batch_size = 130
root.mnist_cifar10.learning_rate = 0.001
root.mnist_cifar10.label = 'mnist_cifar10'
root.mnist_cifar10.taskaware_ecbrs = False
root.mnist_cifar10.taskaware_minority_allocation = 0.5
root.mnist_cifar10.enable_checkpoint = False
root.mnist_cifar10.train_data_lstm = 0.1
root.mnist_cifar10.num_mini_tasks = 20
root.mnist_cifar10.lstm_epochs = 1
root.mnist_cifar10.interim_model_epochs = 1
root.mnist_cifar10.lstm_hidden_size = 2
root.mnist_cifar10.param_weights_queue_length = 5
root.mnist_cifar10.pattern_per_exp = 50
root.mnist_cifar10.is_lazy_training = True
root.mnist_cifar10.clstrategy = None
root.mnist_cifar10.store_weights = False
root.mnist_cifar10.train_lstm = True
root.mnist_cifar10.store_grads = False
root.mnist_cifar10.image_resolution = (-1,3,28,28)


root.anoshiftsubset = edict()
anoshiftsubset= root.anoshiftsubset

root.anoshiftsubset.no_tasks = 10
root.anoshiftsubset.mem_size = 13333
root.anoshiftsubset.replay_size = 10000
root.anoshiftsubset.n_epochs = 5
root.anoshiftsubset.minority_allocation = 0.8
root.anoshiftsubset.batch_size = 1024
root.anoshiftsubset.learning_rate = 0.001
root.anoshiftsubset.label = 'anoshift_subset'
root.anoshiftsubset.enable_checkpoint = False
root.anoshiftsubset.taskaware_ecbrs = False
root.anoshiftsubset.train_data_lstm = 0.5
root.anoshiftsubset.num_mini_tasks = 200
root.anoshiftsubset.lstm_epochs = 3
root.anoshiftsubset.interim_model_epochs = 1
root.anoshiftsubset.lstm_hidden_size = 10
root.anoshiftsubset.param_weights_queue_length = 50
root.anoshiftsubset.pattern_per_exp = 50
root.anoshiftsubset.is_lazy_training = True
root.anoshiftsubset.clstrategy = None
root.anoshiftsubset.store_weights = False
root.anoshiftsubset.train_lstm = True
root.anoshiftsubset.store_grads = False
root.anoshiftsubset.image_resolution = None




root.cifar10 = edict()
cifar10= root.cifar10

root.cifar10.no_tasks = 9
root.cifar10.mem_size = 500
root.cifar10.replay_size = 375
root.cifar10.n_epochs = 5
root.cifar10.minority_allocation = 0.5
root.cifar10.batch_size = 128
root.cifar10.learning_rate = 0.001
root.cifar10.label = 'cifar10'
root.cifar10.taskaware_ecbrs = False
root.cifar10.taskaware_minority_allocation = 0.5
root.cifar10.enable_checkpoint = False
root.cifar10.train_data_lstm = 0.1
root.cifar10.num_mini_tasks = 20
root.cifar10.lstm_epochs = 1
root.cifar10.interim_model_epochs = 1
root.cifar10.lstm_hidden_size = 2
root.cifar10.param_weights_queue_length = 5
root.cifar10.pattern_per_exp = 50
root.cifar10.is_lazy_training = True
root.cifar10.clstrategy = None
root.cifar10.store_weights = False
root.cifar10.train_lstm = True
root.cifar10.store_grads = False
root.cifar10.image_resolution = (-1,3,32,32)

root.cifar100 = edict()
cifar100= root.cifar100

root.cifar100.no_tasks = 19
root.cifar100.mem_size = 500
root.cifar100.replay_size = 375
root.cifar100.n_epochs = 5
root.cifar100.minority_allocation = 0.2
root.cifar100.batch_size = 128
root.cifar100.learning_rate = 0.001
root.cifar100.label = 'cifar100'
root.cifar100.taskaware_ecbrs = False
root.cifar100.taskaware_minority_allocation = 0.5
root.cifar100.enable_checkpoint = False
root.cifar100.train_data_lstm = 0.1
root.cifar100.num_mini_tasks = 20
root.cifar100.lstm_epochs = 1
root.cifar100.interim_model_epochs = 1
root.cifar100.lstm_hidden_size = 2
root.cifar100.param_weights_queue_length = 5
root.cifar100.pattern_per_exp = 26
root.cifar100.is_lazy_training = True
root.cifar100.clstrategy = None
root.cifar100.store_weights = False
root.cifar100.train_lstm = True
root.cifar100.store_grads = False
root.cifar100.image_resolution = (-1,3,32,32)




root.clear10 = edict()
clear10= root.clear10

root.clear10.no_tasks = 10
root.clear10.mem_size = 666
root.clear10.replay_size = 500
root.clear10.n_epochs = 5
root.clear10.minority_allocation = 0.1
root.clear10.batch_size = 128
root.clear10.learning_rate = 0.001#0.0000001
root.clear10.label = 'clear10'
root.clear10.taskaware_ecbrs = False
root.clear10.enable_checkpoint = False
root.clear10.train_data_lstm = 0.1#0.1
root.clear10.num_mini_tasks = 20#20
root.clear10.lstm_epochs = 1#3
root.clear10.interim_model_epochs = 1#2
root.clear10.lstm_hidden_size = 2#4
root.clear10.param_weights_queue_length = 5
root.clear10.pattern_per_exp = 50
root.clear10.is_lazy_training = True
root.clear10.clstrategy = None
root.clear10.store_weights = False
root.clear10.store_grads = False
root.clear10.train_lstm = True
root.clear10.image_resolution = (-1,3,224,224)


root.clear100 = edict()
clear100= root.clear100

root.clear100.no_tasks = 10
root.clear100.mem_size = 2666
root.clear100.replay_size =2000
root.clear100.n_epochs = 5
root.clear100.minority_allocation = 0.1
root.clear100.batch_size = 128
root.clear100.learning_rate = 0.001#0.0000001
root.clear100.label = 'clear100'
root.clear100.taskaware_ecbrs = False
root.clear100.enable_checkpoint = False
root.clear100.train_data_lstm = 0.1#0.1
root.clear100.num_mini_tasks = 20#20
root.clear100.lstm_epochs = 1#3
root.clear100.interim_model_epochs = 1#2
root.clear100.lstm_hidden_size = 2#4
root.clear100.param_weights_queue_length = 5
root.clear100.pattern_per_exp = 200
root.clear100.is_lazy_training = True
root.clear100.clstrategy = None
root.clear100.store_weights = False
root.clear100.store_grads = False
root.clear100.train_lstm = True
root.clear100.image_resolution = (-1,3,224,224)



root.cifar100_large_benign = edict()
cifar100_large_benign= root.cifar100_large_benign

root.cifar100_large_benign.no_tasks = 10
root.cifar100_large_benign.mem_size = 500
root.cifar100_large_benign.mem_size_semi_supervised = 500
root.cifar100_large_benign.replay_size = 375
root.cifar100_large_benign.n_epochs = 5
root.cifar100_large_benign.minority_allocation = 0.1
root.cifar100_large_benign.batch_size = 128
root.cifar100_large_benign.learning_rate = 0.001
root.cifar100_large_benign.label = 'cifar100_large_benign'
root.cifar100_large_benign.taskaware_ecbrs = False
root.cifar100_large_benign.taskaware_minority_allocation = 0.5
root.cifar100_large_benign.enable_checkpoint = False
root.cifar100_large_benign.train_data_lstm = 0.1
root.cifar100_large_benign.num_mini_tasks = 20
root.cifar100_large_benign.lstm_epochs = 1
root.cifar100_large_benign.interim_model_epochs = 1
root.cifar100_large_benign.lstm_hidden_size = 2
root.cifar100_large_benign.param_weights_queue_length = 5
root.cifar100_large_benign.pattern_per_exp = 26
root.cifar100_large_benign.is_lazy_training = True
root.cifar100_large_benign.clstrategy = None
root.cifar100_large_benign.store_weights = False
root.cifar100_large_benign.train_lstm = True
root.cifar100_large_benign.store_grads = False
root.cifar100_large_benign.image_resolution = (-1,3,32,32)
root.cifar100_large_benign.bool_encode_anomaly=False
root.cifar100_large_benign.bool_encode_benign=True
root.cifar100_large_benign.load_whole_train_data=False


root.clear10_cifar10 = edict()
clear10_cifar10=root.clear10_cifar10
root.clear10_cifar10.label = 'clear10_cifar10'
root.clear10_cifar10.no_tasks=10
root.clear10_cifar10.mem_size = 1000#2000
root.clear10_cifar10.replay_size = 750#1500
root.clear10_cifar10.n_epochs = 5
root.clear10_cifar10.minority_allocation = 0.5#0.9#0.3#0.1
root.clear10_cifar10.batch_size = 10
root.clear10_cifar10.learning_rate = 0.001
root.clear10_cifar10.taskaware_ecbrs = False
root.clear10_cifar10.taskaware_minority_allocation = 0.5
root.clear10_cifar10.enable_checkpoint = False
root.clear10_cifar10.train_data_lstm = 0.1
root.clear10_cifar10.num_mini_tasks = 20
root.clear10_cifar10.lstm_epochs = 1
root.clear10_cifar10.interim_model_epochs = 1
root.clear10_cifar10.lstm_hidden_size = 2
root.clear10_cifar10.param_weights_queue_length = 10
root.clear10_cifar10.pattern_per_exp = 50
root.clear10_cifar10.is_lazy_training = True
root.clear10_cifar10.clstrategy = None
root.clear10_cifar10.store_weights = False
root.clear10_cifar10.train_lstm = True
root.clear10_cifar10.image_resolution=(-1,3,32,32)
root.clear10_cifar10.image_resolution_cifar = (-1,3,32,32)
root.clear10_cifar10.image_resolution_clear = (-1,3,224,224)


root.mnist = edict()
mnist= root.mnist

root.mnist.no_tasks = 9
root.mnist.mem_size = 500
root.mnist.replay_size = 375
root.mnist.n_epochs = 5
root.mnist.minority_allocation = 0.0
root.mnist.batch_size = 128#10
root.mnist.learning_rate = 0.001
root.mnist.label = 'mnist'
root.mnist.enable_checkpoint = False
root.mnist.taskaware_ecbrs = False
root.mnist.train_data_lstm = 0.5
root.mnist.num_mini_tasks = 10
root.mnist.lstm_epochs = 5
root.mnist.interim_model_epochs = 5
root.mnist.lstm_hidden_size = 10
root.mnist.param_weights_queue_length = 5
root.mnist.pattern_per_exp = 38
root.mnist.is_lazy_training = True
root.mnist.clstrategy = None
root.mnist.image_resolution = None
root.mnist.store_weights = False
root.mnist.store_grads = False
root.mnist.train_lstm = False


root.svhn = edict()
svhn= root.svhn

root.svhn.no_tasks = 9
root.svhn.mem_size = 500
root.svhn.replay_size = 375
root.svhn.n_epochs = 5
root.svhn.minority_allocation = 0.2#0.1
root.svhn.batch_size = 129#128#10
root.svhn.learning_rate = 0.001
root.svhn.label = 'svhn'
root.svhn.enable_checkpoint = False
root.svhn.taskaware_ecbrs = False
root.svhn.train_data_lstm = 0.5
root.svhn.num_mini_tasks = 10
root.svhn.lstm_epochs = 5
root.svhn.interim_model_epochs = 5
root.svhn.lstm_hidden_size = 10
root.svhn.param_weights_queue_length = 5
root.svhn.pattern_per_exp = 38
root.svhn.is_lazy_training = True
root.svhn.clstrategy = None
root.svhn.image_resolution = None
root.svhn.store_weights = False
root.svhn.store_grads = False
root.svhn.train_lstm = False
root.svhn.image_resolution = (-1,3,32,32)


root.unswnb15 = edict()
unswnb15 = root.unswnb15

root.unswnb15.no_tasks = 9
root.unswnb15.mem_size = 6666
root.unswnb15.replay_size = 5000
root.unswnb15.n_epochs = 5
root.unswnb15.minority_allocation = 0.1
root.unswnb15.batch_size = 1024
root.unswnb15.learning_rate = 0.001
root.unswnb15.label = 'unswnb15'
root.unswnb15.enable_checkpoint = False
root.unswnb15.taskaware_minority_allocation = 0.1
root.unswnb15.taskaware_ecbrs = False
root.unswnb15.train_data_lstm = 0.1
root.unswnb15.num_mini_tasks = 20
root.unswnb15.lstm_epochs = 5
root.unswnb15.interim_model_epochs = 5
root.unswnb15.lstm_hidden_size = 10
root.unswnb15.param_weights_queue_length = 5
root.unswnb15.pattern_per_exp = 500
root.unswnb15.is_lazy_training = True
root.unswnb15.clstrategy = None
root.unswnb15.image_resolution = None
root.unswnb15.store_weights = False
root.unswnb15.image_resolution = None
root.unswnb15.store_grads = False


#cidds hyperparameters
root.cidds01 = edict()
cidds01 = root.cidds01

root.cidds01.no_tasks = 4
root.cidds01.mem_size = 5333
root.cidds01.replay_size = 4000
root.cidds01.n_epochs = 5
root.cidds01.minority_allocation = 0.1
root.cidds01.taskaware_minority_allocation = 0.1
root.cidds01.batch_size = 1024
root.cidds01.learning_rate = 0.001
root.cidds01.label = 'cidds01'
root.cidds01.enable_checkpoint = False
root.cidds01.taskaware_ecbrs = False
root.cidds01.train_data_lstm = 0.1#0.4
root.cidds01.num_mini_tasks = 20
root.cidds01.lstm_epochs = 5
root.cidds01.interim_model_epochs = 5
root.cidds01.lstm_hidden_size = 10
root.cidds01.param_weights_queue_length = 5
root.cidds01.pattern_per_exp = 1000
root.cidds01.is_lazy_training = True
root.cidds01.clstrategy = None
root.cidds01.image_resolution = None
root.cidds01.store_weights = False
root.cidds01.train_lstm = True
root.cidds01.store_grads = False
root.cidds01.image_resolution = None

#kddcup99 dataset
root.kddcup99 = edict()
kddcup99 = root.kddcup99

root.kddcup99.no_tasks = 5
root.kddcup99.mem_size = 5333
root.kddcup99.replay_size = 4000
root.kddcup99.n_epochs = 5
root.kddcup99.minority_allocation = 0.1
root.kddcup99.batch_size = 1024
root.kddcup99.learning_rate = 0.001#1e-12
root.kddcup99.label = 'kddcup99'
root.kddcup99.taskaware_ecbrs = False
root.kddcup99.enable_checkpoint = False
root.kddcup99.train_data_lstm = 0.1#0.4
root.kddcup99.num_mini_tasks = 20
root.kddcup99.lstm_epochs = 5
root.kddcup99.interim_model_epochs = 5
root.kddcup99.lstm_hidden_size = 10
root.kddcup99.param_weights_queue_length = 5
root.kddcup99.pattern_per_exp = 800
root.kddcup99.store_weights = False
root.kddcup99.is_lazy_training = True
root.kddcup99.clstrategy = None
root.kddcup99.store_grads = False
root.kddcup99.image_resolution = None

#nslkdd dataset
root.nslkdd = edict()
nslkdd = root.nslkdd

root.nslkdd.no_tasks = 5
root.nslkdd.mem_size = 1333
root.nslkdd.replay_size = 1000
root.nslkdd.n_epochs = 5
root.nslkdd.minority_allocation = 0.8
root.nslkdd.batch_size = 500
root.nslkdd.learning_rate = 0.001
root.nslkdd.label = 'nslkdd'
root.nslkdd.enable_checkpoint = False
root.nslkdd.taskaware_ecbrs = False
root.nslkdd.train_data_lstm = 0.1#0.4
root.nslkdd.num_mini_tasks = 20
root.nslkdd.lstm_epochs = 5
root.nslkdd.interim_model_epochs = 5
root.nslkdd.lstm_hidden_size = 10
root.nslkdd.param_weights_queue_length = 5
root.nslkdd.pattern_per_exp = 200
root.nslkdd.is_lazy_training = True
root.nslkdd.clstrategy = None
root.nslkdd.store_weights = False
root.nslkdd.image_resolution = None
root.nslkdd.store_grads = False



root.SMAP = edict()
SMAP = root.SMAP

root.SMAP.no_tasks = 6
root.SMAP.mem_size = 4500
root.SMAP.replay_size = 3375
root.SMAP.n_epochs = 5
root.SMAP.minority_allocation = 0.1
root.SMAP.batch_size = 128
root.SMAP.learning_rate = 0.001
root.SMAP.taskaware_minority_allocation = 0.1
root.SMAP.label = 'SMAP'
root.SMAP.enable_checkpoint = False
root.SMAP.taskaware_ecbrs = False
root.SMAP.train_data_lstm = 0.5
root.SMAP.num_mini_tasks = 200
root.SMAP.lstm_epochs = 3
root.SMAP.interim_model_epochs = 1
root.SMAP.lstm_hidden_size = 10
root.SMAP.param_weights_queue_length = 10
root.SMAP.pattern_per_exp = 50
root.SMAP.is_lazy_training = True
root.SMAP.clstrategy = None
root.SMAP.store_weights = False
root.SMAP.train_lstm = True
root.SMAP.image_resolution = None
root.SMAP.store_grads = False



root.SMD = edict()
SMD = root.SMD

root.SMD.no_tasks = 3
root.SMD.mem_size = 7000
root.SMD.replay_size = 5250
root.SMD.n_epochs = 5
root.SMD.minority_allocation = 0.1
root.SMD.batch_size = 128
root.SMD.learning_rate = 0.001
root.SMD.taskaware_minority_allocation = 0.1
root.SMD.label = 'SMD'
root.SMD.enable_checkpoint = False
root.SMD.taskaware_ecbrs = False
root.SMD.train_data_lstm = 0.5
root.SMD.num_mini_tasks = 200
root.SMD.lstm_epochs = 3
root.SMD.interim_model_epochs = 1
root.SMD.lstm_hidden_size = 10
root.SMD.param_weights_queue_length = 10
root.SMD.pattern_per_exp = 50
root.SMD.is_lazy_training = True
root.SMD.clstrategy = None
root.SMD.store_weights = False
root.SMD.train_lstm = True
root.SMD.image_resolution = None
root.SMD.store_grads = False


root.MSL = edict()
MSL = root.MSL

root.MSL.no_tasks = 6
root.MSL.mem_size = 750
root.MSL.replay_size = 560
root.MSL.n_epochs = 5
root.MSL.minority_allocation = 0.1
root.MSL.batch_size = 128
root.MSL.learning_rate = 0.001
root.MSL.taskaware_minority_allocation = 0.1
root.MSL.label = 'MSL'
root.MSL.enable_checkpoint = False
root.MSL.taskaware_ecbrs = False
root.MSL.train_data_lstm = 0.5
root.MSL.num_mini_tasks = 200
root.MSL.lstm_epochs = 3
root.MSL.interim_model_epochs = 1
root.MSL.lstm_hidden_size = 10
root.MSL.param_weights_queue_length = 10
root.MSL.pattern_per_exp = 50
root.MSL.is_lazy_training = True
root.MSL.clstrategy = None
root.MSL.store_weights = False
root.MSL.train_lstm = True
root.MSL.image_resolution = None
root.MSL.store_grads = False




