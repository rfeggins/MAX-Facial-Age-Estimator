# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Facial Age Estimator'
API_DESC = 'Recognize faces in an image and estimate the age of each face.'
API_VERSION = '2.0.0'

# default model
MODELNAME = 'ssrnet_3_3_3_64_1.0_1.0.h5'
DEFAULT_MODEL_PATH = 'assets/{}'.format(MODELNAME)
MODEL_LICENSE = 'MIT'
