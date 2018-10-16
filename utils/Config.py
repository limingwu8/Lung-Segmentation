"""
Common utility functions and classes
"""
import torch

class Option():
    """Training Configuration """
    name = "Lung-Segmentation"
    # root dir of training and validation set
    root_dir = '/media/storage/wu1114/RSNA/rsna-unet/'

    test_root = '/media/storage/wu1114/RSNA/stage_1_test'

    result_root = '/media/storage/wu1114/RSNA/rsna-unet/test_result/'

    img_size = 512
    num_workers = 1     	# number of threads for data loading
    shuffle = False      	# shuffle the data set
    batch_size = 8     		# GTX1060 3G Memory
    epochs = 150			# number of epochs to train
    plot_every = 5          # vis every N batches
    is_train = True     	# True for training, False for making prediction
    save_model = True   	# True for saving the model, False for not saving the model
    caffe_pretrain = False
    env = 'RSNA-UNet'

    n_gpu = 2				# number of GPUs

    learning_rate = 1e-3    # learning rage
    weight_decay = 1e-4		# weight decay

    pin_memory = True   	# use pinned (page-locked) memory. when using CUDA, set to True

    is_cuda = torch.cuda.is_available()  	# True --> GPU
    num_gpus = torch.cuda.device_count()  	# number of GPUs
    checkpoint_dir = "./checkpoints"  		# dir to save checkpoints
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

opt = Option()