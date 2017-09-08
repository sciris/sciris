# Copy this file to config.py
# We probably will end up nixing some of these lines as unneeded under Scris.
SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://optima:optima@localhost:5432/optima'
SECRET_KEY = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'
UPLOAD_FOLDER = '/tmp/uploads'
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml']
SQLALCHEMY_TRACK_MODIFICATIONS = False
REDIS_URL = CELERY_BROKER_URL
MATPLOTLIB_BACKEND = "agg"
