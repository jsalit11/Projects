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
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import constants
import utils

def configure_logs(log_config_file_path): 
    with open(log_config_file_path) as f:
        log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)
    return logging.getLogger(__name__)
logger = configure_logs(constants.LOG_CONFIG_FILE_PATH)  

def create_model(unique_labels):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(unique_labels, activation='softmax')
    ])

    model.compile(optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
    return model

def train_model(unique_labels, file_dir, num_samples, le_file_path, num_epochs, model_file_path):
    model = create_model(unique_labels)
    file_list = utils.get_file_list(file_dir, samples=num_samples, training=True)
    logger.info('Beginning training model')
    images, labels = utils.read_images(file_list, le_file_path, training=True)
    model.fit(images, labels, epochs=num_epochs)
    logger.info('Finished fitting model')
    logger.info(f'Saving model to {model_file_path}')
    model.save(model_file_path)

if __name__ == "__main__":
    train_model(
        constants.UNQ_LABELS, 
        constants.FILE_DIR, 
        constants.SAMPLES, 
        constants.LE_FILE_PATH, 
        constants.NUM_EPOCHS, 
        constants.MODEL_FILE_PATH)