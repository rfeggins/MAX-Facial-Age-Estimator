import logging
import os
import io
import cv2
import numpy as np
from core.src.SSRNET_model import SSR_net
from mtcnn.mtcnn import MTCNN
from PIL import Image
from config import DEFAULT_MODEL_PATH
from maxfw.model import MAXModelWrapper
import tensorflow as tf
global graph
from flask import abort

logger = logging.getLogger()

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def read_still_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        return image
    except IOError as e:
        logger.error(e)
        abort(400, 'Invalid file type/extension. Please provide a valid image (supported formats: JPEG, PNG, TIFF).')

def img_resize(input_data):
    img_h, img_w, _ = np.shape(input_data)
    if img_w > 1024:
        ratio=1024/img_w
        input_data=cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    elif img_h > 1024:
        ratio = 1024/img_h
        input_data=cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    return input_data, ratio


class ModelWrapper(MAXModelWrapper):
    MODEL_META_DATA = {
        'id': 'ssrnet',
        'name': 'SSR-Net Facial Age Estimator Model',
        'description': 'SSR-Net Facial Recognition and Age Prediction model; trained using Keras on the IMDB-WIKI dataset',
        'type': 'Facial Recognition',
        'source': 'https://developer.ibm.com/exchanges/models/all/max-facial-age-estimator/',
        'license': 'MIT'
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        # for face detection
        self.detector = MTCNN()
        try:
            os.mkdir('./img')
        except OSError:
            pass

        # load model and weights
        self.img_size = 64
        self.stage_num = [3, 3, 3]
        self.lambda_local = 1
        self.lambda_d = 1

        # load pre-trained model
        self.model = SSR_net(self.img_size, self.stage_num, self.lambda_local, self.lambda_d)()
        self.model.load_weights(path)
        self.graph = tf.get_default_graph()

        logger.info('Loaded model')

    def _pre_process(self, input_img):
        ad = 0.4
        ratio=1
        img_h, img_w, _ = np.shape(input_img)

        # if image size > 1024 then resize
        if img_h > 1024 or img_w > 1024:
            input_img, ratio =img_resize(input_img)

        img_h, img_w, _ = np.shape(input_img)
        detected = self.detector.detect_faces(input_img)
        faces = np.empty((len(detected), self.img_size, self.img_size, 3))

        for i, d in enumerate(detected):
            if d['confidence'] > 0.85:
                x1, y1, w, h = d['box']
                x2 = x1 + w
                y2 = y1 + h
                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))
        return (faces, detected, ratio)

    def _predict(self, pre_x):
        faces=pre_x[0]
        with self.graph.as_default():
            predicted_ages = self.model.predict(faces)
        return (predicted_ages,pre_x[1], pre_x[2])

    def _post_process(self,post_rst):
        predicted_ages=post_rst[0]
        detected=post_rst[1]
        ratio=post_rst[2]
        pred_res = []
        for i, d in enumerate(detected):
            if d['confidence'] > 0.85:
                pre_age=predicted_ages[i].astype(int)
                if ratio!=1:
                    ratio_box = [ int(x/ratio) for x in d['box']]
                    d['box']=ratio_box
                pred_res.append([{'box': d['box'], 'age':pre_age}])
        return pred_res