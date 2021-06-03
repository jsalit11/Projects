import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import speech_recognition as sr
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
import threading
from queue import Queue

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

import utils
import shapes
import constants
import audio_parser

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hand_detection_model = tf.saved_model.load(constants.saved_model_path)
message = []

def audio():
    while True:
        r = sr.Recognizer()
        sleep_count = 0
        with sr.Microphone(device_index=2) as source:
            r.adjust_for_ambient_noise(source)
            audio=r.listen(source)
        try:    
            # global message
            message.append(r.recognize_google(audio))
        except:
            pass

def video_capture(hand_detection_model):
    frame_count = 0
    text_objs = {}
    shape_objs = {}
    draw_objs = {}
    constants_dict = {}
    constants_dict['resize'] = False

    cap = cv2.VideoCapture(0) 
    while(True): 


        ret, frame = cap.read()
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            logger.info("Error converting to RGB")        
        frame_count += 1
        _ = [shape.reset() for shape in shape_objs.values()]
        if message:
            audio_obj = audio_parser.AudioParser(message)
            try:
                logger.info('Message: {}'.format(message))
                frame, shape_objs, text_objs, draw_objs, constants_dict = \
                    audio_obj.parse_message(frame, shape_objs, text_objs, draw_objs, constants_dict)   
            except Exception as e:
                logger.info('Message {} cannot be parsed'.format(message))
                logger.warning(e)
            message.clear()   

        if frame_count % constants.HAND_DETECT_FRAME_COUNT == 0:
            boxes = utils.detect_hands(hand_detection_model, frame, 
                    constants.MIN_SCORE_THRESHOLD, only_box=constants.ONLY_BOX)
        if frame_count >= constants.HAND_DETECT_FRAME_COUNT:
            if constants.ONLY_BOX:
                for box in boxes:
                    height, width, channels = frame.shape
                    box_coords = utils.get_box_coords(box, width, height)
                    box_area = (box_coords.get('xmax') - box_coords.get('xmin')) * (box_coords.get('ymax') - box_coords.get('ymin'))
                    if box_area > constants.MAX_AREA:
                        continue
                    box_center = utils.get_box_center(box_coords)
                    if constants_dict.get('draw_box'):
                        frame = cv2.rectangle(frame, (box_coords.get('xmin'), box_coords.get('ymin')), 
                            (box_coords.get('xmax'), box_coords.get('ymax')), constants.COLOR_BLUE, 2)

                    if constants_dict.get('draw_mode'):
                        drawing_number = utils.get_drawing_number(draw_objs, frame_count)
                        if drawing_number not in draw_objs.keys():
                            draw_objs[drawing_number] = shapes.Drawing(frame_count, drawing_number)
                        draw_objs[drawing_number].drawing_mode(box_center, frame_count, drawing_number, 
                            draw_mode=True)
                    for draw_obj in draw_objs.values():            
                        frame = draw_obj.draw(frame)
                    for shape in shape_objs.values():
                        if utils.coord_intersect(box_center, shape):
                            # Uncomment to show center location of bounding boxes
                            # frame = cv2.circle(frame, (box_center.get('x'), box_center.get('y')), 
                            #         radius=constants.CIRCLE_RADIUS, color=constants.COLOR_BLUE, thickness=-1)
                            shape.update(box_center, frame, constants_dict)
                for text_obj in text_objs.values():
                    frame = text_obj.draw_text(frame)
                for shape_obj in shape_objs.values():
                    # Uncomment to snap shapes to their location when near similar ones
                    # shape_obj.snap_to_location(shape_objs, frame_count)                    
                    frame = shape_obj.draw_shape(frame)
                    if shape_obj.shape_connections:
                        for shape_2 in shape_obj.shape_connections:
                            frame = shape_obj.connect_shapes(shape_objs, shape_objs.get(shape_2), frame)
            else:
                frame = boxes.copy()

        cv2.imshow('Frame',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_detection_model = tf.saved_model.load(constants.saved_model_path)

    t1 = threading.Thread(target=video_capture, args=(hand_detection_model,))
    t2 = threading.Thread(target=audio, args=())
    t1.start()
    t2.start()
    t1.join()
    t2.join()