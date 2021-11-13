import os
import sys
import cv2
import random
import logging
import yaml
import pickle
import logging.config
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import constants

def configure_logs(log_config_file_path): 
    with open(log_config_file_path) as f:
        log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)
    return logging.getLogger(__name__)
logger = configure_logs(constants.LOG_CONFIG_FILE_PATH)    

def get_file_list(root_file_dir, samples=None, training=False):
    dir_list = os.listdir(root_file_dir)
    file_list = []
    if samples:
        logger.info('Getting a sample of files')
    if training:
        for file_dir in dir_list:
            temp_file_list = os.listdir(os.path.join(root_file_dir, file_dir))
            temp_file_list = [os.path.join(root_file_dir, file_dir, file_name) for file_name in temp_file_list]
            if not temp_file_list:
                logger.info(f'No files in {temp_file_list}')
                continue
            if samples:
                temp_file_list = random.sample(temp_file_list, samples)
            logger.info(f'Getting {len(temp_file_list)} files list from {os.path.join(root_file_dir, file_dir)}')            
            file_list += temp_file_list
    else:
        file_list = dir_list.copy()
        file_list = [os.path.join(root_file_dir, file_name) for file_name in file_list]
    logger.info(f'Returning {len(file_list)} files')
    random.shuffle(file_list)
    return file_list

def read_images(file_list, le_file_path, training=False):
    images, labels = [], []   
    for file_path in file_list:
        image = tf.keras.utils.load_img(
           file_path, target_size=(constants.IMAGE_WIDTH, constants.IMAGE_HEIGHT)
)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        images.append(image)
        if training:
            label = file_path.split('\\')[-2]
            labels.append(label)
    if training:
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        logger.info('Saving label encoder to file')
        with open(le_file_path, 'wb') as f:
            pickle.dump(le, f)
        labels = to_categorical(labels, constants.UNQ_LABELS)
    return np.array(images), np.array(labels)    