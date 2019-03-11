import pytest
import requests
import os
from PIL import Image
import numpy as np
from core.util import img_resize


def test_swagger():
    model_endpoint = 'http://localhost:5000/swagger.json'
    r = requests.get(url=model_endpoint)

    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'
    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Facial Age Estimator'

def test_metadata():
    model_endpoint = 'http://localhost:5000/model/metadata'
    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'ssrnet'
    assert metadata['name'] == 'SSR-Net Facial Age Estimator Model'
    assert metadata['description'] == 'SSR-Net Facial Recognition and Age Prediction model; trained using Keras on ' \
                                      'the IMDB-WIKI dataset'
    assert metadata['license'] == 'MIT'


def _check_response(r):
    assert r.status_code == 200
    json = r.json()
    assert json['status'] == "ok"
    assert 55 > json['predictions'][0]['age_estimation'] > 35
    assert 575 > json['predictions'][0]['face_box'][0] > 560
    assert 335 > json['predictions'][0]['face_box'][1] > 320
    assert 720 > json['predictions'][0]['face_box'][2] > 700
    assert 975 > json['predictions'][0]['face_box'][3] > 960


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'
    formats = ['jpg', 'png', 'tiff']
    file_path = 'tests/tom_cruise.{}'

    for f in formats:
        p = file_path.format(f)
        with open(p, 'rb') as file:
            file_form = {'image': (p, file, 'image/{}'.format(f))}
            r = requests.post(url=model_endpoint, files=file_form)
        _check_response(r)

    file_path3 = 'tests/non_face.jpg'
    with open(file_path3, 'rb') as file:
        file_form3 = {'image': (file_path3, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form3)
    assert r.status_code == 200
    json = r.json()
    assert json['status'] == "ok"
    assert json['predictions'] ==[]

    file_path = 'README.md'
    with open(file_path,'rb') as file:
        file_form = {'text': (file_path, file, 'text/plain')}
        r = requests.post(url=model_endpoint, files=file_form)
    assert r.status_code == 400

def test_img_resize():
        """
        The image resize test.
        """

        resize_path=[]
        for file in os.listdir("tests/"):
            if file.startswith("resize_"):
                resize_path.append("tests/"+ file)

        for i in range (len(resize_path)):
            image = Image.open(resize_path[i])
            image = np.array(image)
            input_img_h, input_img_w, _ = image.shape
            input_img, ratio = img_resize(image)
            resize_img_h, resize_img_w, _ = input_img.shape
            assert 3 > input_img_h - int(resize_img_h / ratio) >= 0
            assert 3 > input_img_w - int(resize_img_w / ratio) >= 0

if __name__ == '__main__':
    pytest.main([__file__])
