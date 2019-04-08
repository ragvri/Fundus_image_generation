    
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os


# setting environment for GPU
def setup_gpu(gpu_id: str):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # don't show any messages
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))