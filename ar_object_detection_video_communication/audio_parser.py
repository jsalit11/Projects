import os
import sys
import speech_recognition as sr
import shapes
import constants
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioParser:

    def __init__(self, message):
        self.message = message

    def _standardize_message(self):
        return self.message[0].lower().split(' ')

    def _action(self, message, standardized_message, frame, shape_objs, text_objs, draw_objs, constants_dict):
        if 'system' in standardized_message:
            if 'on' in standardized_message:
                constants_dict['system'] = True
                logger.info('Turning system on')
            elif 'off' in standardized_message:
                constants_dict['system'] = False
                logger.info('Turning system off')
        if not constants_dict.get('system'):
            return frame, shape_objs, text_objs, draw_objs, constants_dict
        if 'name' or 'named' in standardized_message:
            if 'named' in standardized_message:
                name_idx = standardized_message.index('named') + 1
                name = ' '.join(standardized_message[name_idx:])
            elif 'name' in standardized_message:
                name_idx = standardized_message.index('name') + 1
                name = ' '.join(standardized_message[name_idx:])
        if 'create' in standardized_message:
            if 'color' in standardized_message:
                color_idx = standardized_message.index('color')
                color = standardized_message[color_idx + 1]
                color = 'COLOR_' + color.upper()                    
            else:    
                color = constants.DEFAULT_SHAPE_COLOR
            if 'rectangle' in standardized_message:
                shape_objs[name] = shapes.Rectangle(name, color=color)
            elif 'circle' in standardized_message:
                shape_objs[name] = shapes.Circle(name, color=color)
        elif 'text' in standardized_message:
            text_idx = standardized_message.index('text')
            text_message = standardized_message[text_idx + 1]
            logger.info('Message text: {}'.format(text_message))
            if 'shape' in standardized_message:
                shape_idx = standardized_message.index('shape')
                shape_name = standardized_message[shape_idx + 1]
                logger.info('Message shape name: {}'.format(shape_name))
                text_objs[name] = shapes.Text(text_message, shape_objs[shape_name])
            else:
                text_objs[name] = shapes.Text(text_message)
        elif 'remove' in standardized_message:
            if 'shape' in standardized_message:
                del shape_objs[name]
            elif 'text' in standardized_message:
                del text_objs[name]
            elif 'draw' in standardized_message:
                draw_objs.clear()
        elif 'clear all objects' in message:
            shape_objs.clear()
            text_objs.clear()
            draw_objs.clear()
        elif 'connect' in standardized_message:
            shape_idx = standardized_message.index('shape')
            with_idx = standardized_message.index('with')
            shape_1 = standardized_message[(shape_idx + 1)]
            shape_2 = standardized_message[with_idx + 1]
            if shape_1 and shape_2 in shape_objs.keys():
                shape_objs[shape_1].shape_connections.append(shape_2)
        elif 'draw mode' in standardized_message:
            constants_dict['draw_mode'] = True
        elif 'boxes' in standardized_message:
            if 'on' in standardized_message:
                constants_dict['draw_box'] = True
            elif 'off' in standardized_message:
                constants_dict['draw_box'] = False
        elif 'resize' in standardized_message:
            if 'on' in standardized_message:
                constants_dict['resize'] = True
            if 'off' in standardized_message:
                constants_dict['resize'] = False
        elif 'load total process' in message:
            constants_dict['load_process'] = True
        elif 'clear total process' in message:
            constants_dict['load_process'] = False
        return frame, shape_objs, text_objs, draw_objs, constants_dict

    def parse_message(self, frame, shapes, text, draw_objs, constants_dict):
        standardized_message = self._standardize_message()
        return self._action(self.message[0].lower(), standardized_message, frame, shapes, text, draw_objs,
            constants_dict)