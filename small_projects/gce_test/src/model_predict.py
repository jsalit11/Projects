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
logger = configure_logs(constants.PREDICT_LOG_CONFIG_FILE_PATH)   

def predict_images(model_file_path, predict_file_dir, le_file_path):
    model = tf.keras.models.load_model(model_file_path)
    file_list = utils.get_file_list(predict_file_dir)
    images, _ = utils.read_images(file_list, le_file_path)
    with open(le_file_path, 'rb') as f:
        le = pickle.load(f)
    predictions = model.predict(images)
    predicted_classes = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
    for idx in range(len(predictions)):
        file_name = file_list[idx].split('\\')[-1]
        predicted_class = predicted_classes[idx]
        logger.info(f'File: {file_name} | Predicted Class: {predicted_class}')

if __name__ == "__main__":
    predict_images(
        constants.MODEL_FILE_PATH, 
        constants.PREDICT_FILE_DIR, 
        constants.LE_FILE_PATH)