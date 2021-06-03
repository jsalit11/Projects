import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import speech_recognition as sr
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

import utils
import shapes
import audio_parser
import constants

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Rectangle:
    
    def __init__(self, name, shape_type='rectangle', color=constants.DEFAULT_SHAPE_COLOR,
        thickness=constants.THICKNESS, movement_multiplier=constants.MOVEMENT_MULTIPLIER):
        self.coords = self.initialize_shape()
        self.name = name
        self.shape_type = shape_type
        self.color = color
        self.thickness = thickness
        self.movement_multiplier = movement_multiplier
        self.hand_intersections = 0
        self.shape_connections = []

    def initialize_shape(self):
        return {
                        'xmin': constants.INITIAL_BOX_LOCATION_MIN[0],
                        'ymin': constants.INITIAL_BOX_LOCATION_MIN[1],
                        'xmax': constants.INITIAL_BOX_LOCATION_MAX[0],
                        'ymax': constants.INITIAL_BOX_LOCATION_MAX[1]
                    }        

    def draw_shape(self, frame):
        shape_center = self.get_center()
        center_x = shape_center.get('x')
        center_y = shape_center.get('y')
        for word in self.name.split(' '):
            frame = cv2.putText(frame, word, (center_x - constants.X_TEXT_OFFSET,
                center_y),cv2.FONT_HERSHEY_SIMPLEX, constants.FONT_SCALE, self.color, constants.TEXT_THICKNESS, cv2.LINE_AA)
            center_y += constants.Y_MULTI_WORD_ADJUSTER
        return cv2.rectangle(frame, (self.coords.get('xmin'), self.coords.get('ymin')),
            (self.coords.get('xmax'), self.coords.get('ymax')), self.color, self.thickness)

    def update(self, hand_box_center, frame, constants_dict):
        self.hand_intersections += 1
        if self.hand_intersections == 1 or not constants_dict.get('resize'):
            self.update_coords(hand_box_center)
        else:
            self.resize()
    
    def reset(self):
        self.hand_intersections = 0

    def update_coords(self, hand_box_center):
        logger.info('Updating coordinates of {}'.format(self.name))
        x_change = (self.coords.get('xmax') - self.coords.get('xmin')) - hand_box_center.get('x')
        x_change = 1 if x_change > 0 else -1
        y_change = (self.coords.get('ymax') - self.coords.get('ymin')) - hand_box_center.get('y')
        y_change = 1 if y_change > 0 else -1  
        self.coords['xmin'] += self.movement_multiplier * x_change
        self.coords['xmax'] += self.movement_multiplier * x_change
        self.coords['ymin'] += self.movement_multiplier * y_change
        self.coords['ymax'] += self.movement_multiplier * y_change   

    def get_center(self):
        shape_center = {}
        shape_center['x'] = int((self.coords.get('xmax') + self.coords.get('xmin')) / 2)
        shape_center['y'] = int((self.coords.get('ymax') + self.coords.get('ymin')) / 2)
        return shape_center

    def resize(self):
        logger.info('Resizing {}'.format(self.name))
        self.coords['xmax'] += self.movement_multiplier // constants.RESIZE_DIVIDER 
        self.coords['ymax'] += self.movement_multiplier // constants.RESIZE_DIVIDER 

    def get_edge(self):
        edges = {}
        edges['bottom'] = (self.coords.get('xmin'), self.coords.get('ymax'), 
                            self.coords.get('xmax'), self.coords.get('ymax'))
        edges['left'] = (self.coords.get('xmin'), self.coords.get('ymin'), 
                            self.coords.get('xmin'), self.coords.get('ymax'))
        edges['right'] = (self.coords.get('xmax'), self.coords.get('ymin'), 
                            self.coords.get('xmax'), self.coords.get('ymax'))
        edges['top'] = (self.coords.get('xmin'), self.coords.get('ymin'), 
                            self.coords.get('xmax'), self.coords.get('ymin'))
        return edges

    def get_edge_center(self, edge):
        x = int((edge[0] + edge[2]) / 2)
        y = int((edge[1] + edge[3]) / 2)
        return (x, y)

    def connect_shapes(self, shape_objs, shape2, frame):
        shape1_centers = {k:self.get_edge_center(v) for k, v in self.get_edge().items()}
        shape2_centers = {k:shape2.get_edge_center(v) for k, v in shape2.get_edge().items()}
        shape1_edge_orient = self.get_edge_orient(shape1_centers, shape2_centers)
        shape2_edge_orient = self.get_edge_orient(shape2_centers, shape1_centers)
        shape1_edge = self.get_edge().get(shape1_edge_orient)
        shape1_edge_center = self.get_edge_center(shape1_edge)
        shape2_edge = shape2.get_edge().get(shape2_edge_orient)
        shape2_edge_center = shape2.get_edge_center(shape2_edge)
        return cv2.line(frame, shape1_edge_center, shape2_edge_center,  
            constants.COLOR_BLUE, 2)

    def get_edge_orient(self, shape1_centers, shape2_centers):
        edge_orient = None
        #Arbitary large number
        distance = constants.ARB_LARGE_NUMBER
        for orient_1, center_1 in shape1_centers.items():
            for orient_2, center_2 in shape2_centers.items():
                temp_distance = np.abs(center_2[0] - center_1[0]) + \
                    np.abs(center_2[1] - center_1[1])
                if temp_distance < distance:
                    edge_orient = orient_1
                    distance = temp_distance
        return edge_orient

    def snap_to_location(self, shape_objs, frame_count):
        if frame_count != constants.FRAME_SNAP_COUNT:
            return
        current_center = self.get_center()
        for shape_obj in shape_objs.values():
            other_shape_obj_center = shape_obj.get_center()
            x_center_diff = current_center.get('x') - other_shape_obj_center.get('x')
            y_center_diff = current_center.get('y') - other_shape_obj_center.get('y')            
            if np.abs(x_center_diff) < constants.LOCATION_SNAP:
                self.coords['xmin'] += x_center_diff
                self.coords['xmax'] += x_center_diff
            if np.abs(y_center_diff) < constants.LOCATION_SNAP:            
                self.coords['ymin'] += y_center_diff
                self.coords['ymax'] += y_center_diff                                                 

    def get_area(self):
        x = self.coords.get('xmax') - self.coords.get('xmin')
        y = self.coords.get('ymax') - self.coords.get('ymin')
        return x*y

    def snap_to_size(self, shape_objs, frame_count):
        if frame_count != constants.FRAME_SNAP_COUNT:
            return
        for shape_obj in shape_objs.values():
            area_diff, other_obj_area = self.get_area_diff(shape_obj)
            area_diff_perc = np.abs(area_diff) / other_obj_area
            if area_diff_perc < constants.AREA_SNAP:
                while area_diff != 0:
                    self.coords['xmax'] += 1
                    self.coords['ymax'] += 1

    def get_area_diff(self, shape_obj):
        area = self.get_area()
        other_obj_area = shape_obj.get_area()
        area_diff = area - other_obj_area
        return area_diff, other_obj_area                    

class Circle:
    
    def __init__(self, name, shape_type='circle', color=constants.DEFAULT_SHAPE_COLOR,
            thickness=constants.THICKNESS, movement_multiplier=constants.MOVEMENT_MULTIPLIER):
        self.coords = self.initialize_shape()
        self.name = name
        self.shape_type = shape_type
        self.color = color
        self.thickness = thickness
        self.movement_multiplier = movement_multiplier
        self.hand_intersections = 0
        self.radius = constants.DEFAULT_CIRCLE_RADIUS
        self.shape_connections = []

    def initialize_shape(self):
        return {
                        'x': constants.INITIAL_BOX_LOCATION_MAX[0],
                        'y': constants.INITIAL_BOX_LOCATION_MAX[1]
                    }        

    def draw_shape(self, frame):
        center_y = self.coords.get('y')
        for word in self.name.split(' '):
            frame = cv2.putText(frame, word, (self.coords.get('x') - constants.X_TEXT_OFFSET,
                center_y),cv2.FONT_HERSHEY_SIMPLEX, constants.FONT_SCALE, self.color, constants.TEXT_THICKNESS, cv2.LINE_AA)
            center_y += constants.Y_MULTI_WORD_ADJUSTER
        return cv2.circle(frame, (self.coords.get('x'), self.coords.get('y')), self.radius, 
                self.color, self.thickness)

    def update(self, hand_box_center, frame, constants_dict):
        self.hand_intersections += 1
        if self.hand_intersections == 1 or not constants_dict.get('resize'):
            self.update_coords(hand_box_center)
        else:
            self.resize()
    
    def reset(self):
        self.hand_intersections = 0

    def update_coords(self, hand_box_center):
        logger.info('Updating coordinates of {}'.format(self.name))
        x_change = self.coords.get('x') - hand_box_center.get('x')
        x_change = 1 if x_change > 0 else -1
        y_change = self.coords.get('y') - hand_box_center.get('y')
        y_change = 1 if y_change > 0 else -1  
        self.coords['x'] += self.movement_multiplier * x_change
        self.coords['y'] += self.movement_multiplier * y_change

    def get_center(self):
        return {
            'x': self.coords.get('x'),
            'y': self.coords.get('y')
        }

    def resize(self):
        logger.info('Resizing {}'.format(self.name))
        self.radius += self.movement_multiplier // constants.RESIZE_DIVIDER

    def get_edge(self):
        edges = {}
        edges['bottom'] = (self.coords.get('x'), self.coords.get('y') + self.radius)
        edges['left'] = (self.coords.get('x') - self.radius, self.coords.get('y'))
        edges['right'] = (self.coords.get('x') + self.radius, self.coords.get('y'))
        edges['top'] = (self.coords.get('x'), self.coords.get('y') - self.radius)
        return edges

    def get_edge_center(self, edge):
        return (edge[0], edge[1])

    def connect_shapes(self, shape_objs, shape2, frame):
        shape1_centers = {k:self.get_edge_center(v) for k, v in self.get_edge().items()}
        shape2_centers = {k:shape2.get_edge_center(v) for k, v in shape2.get_edge().items()}
        shape1_edge_orient = self.get_edge_orient(shape1_centers, shape2_centers)
        shape2_edge_orient = self.get_edge_orient(shape2_centers, shape1_centers)
        shape1_edge = self.get_edge().get(shape1_edge_orient)
        shape1_edge_center = self.get_edge_center(shape1_edge)
        shape2_edge = shape2.get_edge().get(shape2_edge_orient)
        shape2_edge_center = shape2.get_edge_center(shape2_edge)
        return cv2.line(frame, shape1_edge_center, shape2_edge_center,  
            constants.COLOR_BLUE, 2)

    def get_edge_orient(self, shape1_centers, shape2_centers):
        edge_orient = None
        #Arbitary large number
        distance = constants.ARB_LARGE_NUMBER
        for orient_1, center_1 in shape1_centers.items():
            for orient_2, center_2 in shape2_centers.items():
                temp_distance = np.abs(center_2[0] - center_1[0]) + \
                    np.abs(center_2[1] - center_1[1])
                if temp_distance < distance:
                    edge_orient = orient_1
                    distance = temp_distance
        return edge_orient

    def snap_to_location(self, shape_objs, frame_count):
        if frame_count != constants.FRAME_SNAP_COUNT:
            return
        current_center = self.get_center()
        for shape_obj in shape_objs.values():
            other_shape_obj_center = shape_obj.get_center()
            x_center_diff = current_center.get('x') - other_shape_obj_center.get('x')
            y_center_diff = current_center.get('y') - other_shape_obj_center.get('y')            
            if np.abs(x_center_diff) < constants.LOCATION_SNAP:
                self.coords['x'] += x_center_diff
            if np.abs(y_center_diff) < constants.LOCATION_SNAP:            
                self.coords['y'] += y_center_diff

    def get_area(self):
        return np.pi * (self.radius**2)

    def snap_to_size(self, shape_objs, frame_count):
        if frame_count != constants.FRAME_SNAP_COUNT:
            return
        for shape_obj in shape_objs.values():
            area_diff, other_obj_area = self.get_area_diff(shape_obj)
            area_diff_perc = np.abs(area_diff) / other_obj_area
            if area_diff_perc < constants.AREA_SNAP:
                while area_diff != 0:
                    self.coords['x'] += 1
                    self.coords['y'] += 1
                    area_diff = self.get_area_diff(shape_obj)

    def get_area_diff(self, shape_obj):
        area = self.get_area()
        other_obj_area = shape_obj.get_area()
        area_diff = area - other_obj_area
        return area_diff, other_obj_area

class Text:

    def __init__(self, text, shape=None, color=constants.COLOR_BLUE):
        self.text = text
        self.shape = shape
        self.coords = {}
        self.x_offset = 20     
        self.color = color

    def update_coords(self):
        if self.shape is not None:
            self.coords = self.shape.get_center()
            self.coords['x'] -= self.x_offset
        elif self.coords:
            pass
        else:         
            self.coords = {
                            'x': constants.INITIAL_BOX_LOCATION_MIN[0] - self.x_offset,
                            'y': constants.INITIAL_BOX_LOCATION_MIN[1]
                        }        

    def draw_text(self, frame):
        self.update_coords()
        center_x = self.coords.get('x')
        center_y = self.coords.get('y')
        for word in self.text.split(' '):
            frame = cv2.putText(frame, word, (center_x,center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                constants.FONT_SCALE, self.color, 2, cv2.LINE_AA)
            center_y += constants.Y_MULTI_WORD_ADJUSTER        
        return frame

class Drawing:

    def __init__(self, frame_number, drawing_number,
            color=constants.DEFAULT_SHAPE_COLOR, thickness=-1):        
        self.color = color
        self.thickness = thickness
        self.circle_coords = []
        self.frame_start = frame_number
        self.drawing_number = drawing_number
        self.radius = 5

    def draw_flag(self, frame_number, draw_mode=False):
        if draw_mode:
            frame_difference = self.frame_start - frame_number
            if frame_difference > constants.DRAWING_TIME_LIMIT:
                draw_mode = False
        return draw_mode

    def drawing_mode(self, box_center, frame_number, drawing_number, draw_mode=False):
        if self.draw_flag(frame_number, draw_mode=draw_mode):
            self.circle_coords.append((box_center.get('x'), box_center.get('y')))
        return draw_mode

    def draw(self, frame):
        for circle_coord in self.circle_coords:
            frame = cv2.circle(frame, circle_coord, radius=self.radius, 
                color=self.color, thickness=self.thickness)
        return frame


