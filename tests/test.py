import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)

    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'
    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Microservice'

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


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'assets/tom_cruise.jpg'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    json = r.json()
    assert json['status'] == "ok"
    assert 55 > json['predictions'][0]['age_estimation'] > 35
    assert 310 > json['predictions'][0]['face_box'][0] > 290
    assert 180 > json['predictions'][0]['face_box'][1] > 160
    assert 390 > json['predictions'][0]['face_box'][2] > 370
    assert 525 > json['predictions'][0]['face_box'][3] > 500


if __name__ == '__main__':
    pytest.main([__file__])
