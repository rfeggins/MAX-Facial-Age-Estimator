from maxfw.core import MAX_API, PredictAPI
from core.model import ModelWrapper, read_still_image
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest

input_parser = MAX_API.parser()
# Example parser for file input
input_parser.add_argument('image', type=FileStorage, location='files', required=True, help='An image encoded as JPEG, PNG, or TIFF')

label_prediction = MAX_API.model('LabelPrediction', {
    'age_estimation': fields.Integer(required=True, description='Estimated age for the face'),
    'detection_box': fields.List(fields.Float(required=True), description='Bounding box coordinates for the face, ' + \
         'in the form of an array of normalized coordinates [ymin, xmin, ymax, xmax]. Each coordinate is in the range [0, 1]')
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction), description='Predicted age and bounding box for each detected face')
})

class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        input_data = args['image'].read()
        stillimg = read_still_image(input_data)
        preds = self.model_wrapper.predict(stillimg)

        label_preds=[]
        for res in preds:
            label_preds.append({'age_estimation':res[0]['age'],'detection_box':res[0]['box']})
        result['predictions'] = label_preds
        result['status'] = 'ok'
        return result
