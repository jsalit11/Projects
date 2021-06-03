import os
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import tensorflow as tf
# sys.path.append('/content/gdrive/My Drive/capstone/src')
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection import exporter_lib_v2
import utils

import logging

logger = logging.getLogger(__name__)
f_handler = logging.FileHandler(r'models\hands\ssd_model_4\results.log')
# c_handler = logging.StreamHandler()
# logger.addHandler(c_handler)
logger.addHandler(f_handler)
# c_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
# c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.setLevel(logging.INFO)

# !wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# !tar -xf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# !mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint gdrive/MyDrive/capstone/models/hands
# !mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config gdrive/MyDrive/capstone/models/hands/pipeline.config
# !cp -r ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint gdrive/MyDrive/capstone/models/hands

# !wget http://vision.soic.indiana.edu/egohands_files/egohands_data.zip
# !cp egohands_data.zip gdrive/MyDrive/capstone/data/hands/train_new/train.zip
# !unzip -q gdrive/MyDrive/capstone/data/hands/train_new/train.zip -d gdrive/MyDrive/capstone/data/hands/train_new/

def load_image_into_numpy_array(path):
    # image = Image.open(path)    
    # (im_width, im_height) = image.size
    # image = np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)
    return cv2.imread(path, cv2.IMREAD_COLOR)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_idnex,
                    figsize=(12,16),
                    image_name=None):
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)

def get_box_values(labels, file_name):
    row = [i for i in labels if i[0] == file_name]
    if not row:
        return
    row = row[0]
    ymin = int(row[-3]) / int(row[2])
    xmin = int(row[-4]) / int(row[1])
    ymax = int(row[-1]) / int(row[2])
    xmax = int(row[-2]) / int(row[1])
    box = np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)
    return box

def load_images(file_dir, images, samples=None):
    train_images_np = []
    train_gt_box = []
    # images = os.listdir(file_dir)
    if samples is not None:
        images = random.sample(images, samples)

    with open(os.path.join(file_dir, 'train_labels.csv')) as f:
        labels = f.read().splitlines()     
    labels = [i.split(',') for i in labels]   

    # print('Start loading {} images'.format(len(images)))
    for idx, image in enumerate(images):
        if not image.endswith('.jpg'):
            continue
        image_path = os.path.join(file_dir, image)
        box = get_box_values(labels, image)
        if box is None:
            continue
        train_images_np.append(load_image_into_numpy_array(image_path))            
        train_gt_box.append(box)
        # if idx % 50 == 0:
        # print('Finished loading {} images'.format(len(images)))
    # print('Finished loading {} images'.format(idx))
    return train_images_np, train_gt_box

def prepare_data(train_images_np, gt_boxes):
    hand_class_id = 1
    num_classes = 1
    category_index = {hand_class_id: {'id': hand_class_id, 'name': 'hand'}}            
    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    # print('Start prepping data.')
    for (train_image_np, gt_box_np) in zip(train_images_np, gt_boxes):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            np.ones(shape=[gt_box_np.shape[0]], dtype = np.int32) - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))
    # print('Done prepping data.')
    return train_image_tensors, gt_classes_one_hot_tensors, gt_box_tensors

def create_model(pipeline_config_path, output_directory, checkpoint_path):
    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)
    num_classes = 1
    output_checkpoint_dir = os.path.join(output_directory, 'checkpoint')
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)
    pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_proto, output_directory)

    latest_checkpoint_number = int(checkpoint_path.split('-')[-1])
    print(latest_checkpoint_number)
    if latest_checkpoint_number == 0:
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
            )
        fake_model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()

    exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt_manager = tf.train.CheckpointManager(
        exported_ckpt, output_checkpoint_dir, max_to_keep=1)
    if latest_checkpoint_number > 0:
        status = exported_ckpt.restore(ckpt_manager.latest_checkpoint)

    image, shapes = detection_model.preprocess(tf.zeros([1, 320, 320, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')
    return detection_model, pipeline_proto, ckpt_manager

def train_model(detection_model, train_images_np, train_image_tensors, gt_classes_one_hot_tensors, 
                gt_box_tensors, ckpt_manager):    

    batch_size = 32
    learning_rate = 0.04
    num_batches = len(train_images_np) // batch_size

    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune, batch_size)

    # print('Start fine-tuning!', flush=True)
    # for idx in range(num_batches):
    all_keys = list(range(len(gt_box_tensors)))
    random.shuffle(all_keys)
    example_keys = all_keys[:batch_size]
    gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
    gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
    image_tensors = [train_image_tensors[key] for key in example_keys]
    total_loss, losses_dict = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        # if idx % 2 == 0:
        #     print('Batch ' + str(idx) + '/' + str(num_batches) + 'Total loss - ' +  str(total_loss.numpy()), flush=True)

    # print('Done fine-tuning!')
    # ckpt_manager.save()
    # print('Checkpoint saved!')
    return detection_model, losses_dict

def get_model_train_step_function(model, optimizer, vars_to_fine_tune, batch_size):

    # @tf.function
    def train_step_fn(image_tensors,
                        groundtruth_boxes_list,
                        groundtruth_classes_list):
        shapes = tf.constant(batch_size * [[320, 320, 3]], dtype=tf.int32)      
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat([model.preprocess(image_tensor)[0]
                for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss, losses_dict

    return train_step_fn

def get_image_dict(image_dict_path, labels_path):
    with open(image_dict_path) as f:
        image_dict = json.load(f)

    with open(labels_path) as f:
        labels = f.read().splitlines()
        labels = [i.split(',') for i in labels]   

    train_images_np = []
    train_gt_box = []

    print('Start loading images')
    for image_name, image in image_dict.items():
        box = get_box_values(labels, image_name)
        if box is None:
            continue
        train_gt_box.append(box)
        train_images_np.append(image)
    print('Finished loading {} images'.format(len(image_dict)))
    return train_images_np, train_gt_box


def main(train_images_dir, pipeline_config_path, output_directory, checkpoint_path,
         num_epochs=1, image_dict=None, labels_path=None, samples=None):
    detection_model, pipeline_proto, ckpt_manager = create_model(
                                                         pipeline_config_path, 
                                                         output_directory,
                                                         checkpoint_path)
    
    train_files = os.listdir(train_images_dir)
    # all_keys = list(range(len(train_files)))
    # train_files = random.sample(train_files, 2000)
    random.shuffle(train_files)    
    BATCH_SIZE = 32
    num_batches = (len(train_files) // BATCH_SIZE) - 1
    # @tf.function
    for epoch in range(num_epochs):
        # print('Epoch Number - {} / {}'.format(epoch, num_epochs))
        for idx in range(num_batches):
            # print('Training batch number - {} / {}'.format(idx, num_batches))
            batch_files = train_files[BATCH_SIZE*idx:BATCH_SIZE*(idx+1)]
            train_images_np, train_gt_box = load_images(train_images_dir, batch_files)            
            train_image_tensors, gt_classes_one_hot_tensors, gt_box_tensors = \
                prepare_data(train_images_np, train_gt_box)        
            detection_model, losses_dict = train_model(detection_model, train_images_np, 
                                            train_image_tensors, gt_classes_one_hot_tensors, 
                                            gt_box_tensors, ckpt_manager)  
            logger.info(utils.log_results(epoch, num_epochs, idx, num_batches, 
                losses_dict))
            if idx % 10 == 0:
                ckpt_manager.save()
                print('Checkpoint saved!')
    exporter_lib_v2.export_inference_graph(
        input_type='image_tensor',
        pipeline_config = pipeline_proto,
        trained_checkpoint_dir = os.path.join(output_directory, r'checkpoint'),
        output_directory = output_directory
    )    



if __name__ == "__main__":
    pipeline_config_path = r'models\hands\ssd_model_4\pipeline.config'
    output_directory = r'models\hands\ssd_model_4'
    checkpoint_path = os.path.join(output_directory, r'checkpoint\ckpt-86')
    train_images_dir = r'images\train'
    main(train_images_dir, pipeline_config_path, output_directory, checkpoint_path, num_epochs=80)