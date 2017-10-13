# Copy this file to config.py
SECRET_KEY = 'Pick something unique for your site here'
# NOTE: For the _DIR parameters, you can use full absolute paths also, though 
# for Windows, you need to convert all backslashes to forward slashes.  If 
# you use a relative path, it is interpreted as being with respect to the 
# root path of the sciris repository.
CLIENT_DIR = 'vueinterface'
MODEL_DIR = 'scirismodel'
WEBAPP_DIR = 'webapp'
UPLOADS_DIR = 'uploads'
FILESAVEROOT_DIR = 'savedfiles'
REDIS_URL = 'redis://localhost:6379/1/'
REGISTER_AUTOACTIVATE = True