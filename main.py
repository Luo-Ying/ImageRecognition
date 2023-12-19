print("ok")

import tensorflow as tf
import pathlib
import os
from tensorflow.python.client import device_lib

from dataPreprocessing import handle_dataset
from testImageRandom import handle_image_gave

'''Choix de model import'''
# from model_CNN import save_and_train_model   
# from model_MLP import save_and_train_model
from model_ResNet50 import save_and_train_model
# from model_VGG16 import save_and_train_model
 
print("lol")

dataset_path = './data_set/skin-disease-dataset/'
BATCH_SIZE = 64
seed = 42
# steps_per_epoch = 13 # 832 / BATCH_SIZE

log_dir_base = "logs_train_model/"

'''Choix de model'''
# model_path = "saved_models/model_CNN"
# model_path = "saved_models/model_MLP"
model_path = "saved_models/model_Resnet50"
# model_path = "saved_models/model_VGG16"

image_to_predict = "./test_images_random/FU-ringworm-1.jpeg"

def train():
    
    train_dataset = handle_dataset(dataset_path + 'train_set/', BATCH_SIZE, seed)
    validation_dataset = handle_dataset(dataset_path + 'validation_set/', BATCH_SIZE, seed)
    test_dataset = handle_dataset(dataset_path + 'test_set/', BATCH_SIZE, seed)
    
    '''
    Donner une variable booléen d'entrée = True de plus pour le model ResNet50 et VGG16 si on prend un model pré-entrainné
    '''
    save_and_train_model(model_path, log_dir_base, train_dataset, validation_dataset, test_dataset)
    # save_and_train_model(model_path, log_dir_base, train_dataset, validation_dataset, test_dataset, True)
    
def predict():
    
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    
    input_image = handle_image_gave(image_to_predict)
    prediction = model.predict(input_image)
    
    print(prediction)
    

def main():
    
    train()
    predict()
    

if __name__ == '__main__':
    
    # os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
    # print(device_lib.list_local_devices())
    
    print("GPUs Available: ", tf.test.is_gpu_available())
    print("GPUs name: ", tf.test.gpu_device_name())
    
    with tf.device('gpu:1'):
        main()