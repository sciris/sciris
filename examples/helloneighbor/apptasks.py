"""
apptasks.py -- The Celery tasks module for this webapp
    
Last update: 6/17/18 (gchadder3)
"""

import time
from sciris.weblib.tasks import register_async_task
from sciris.weblib.tasks import make_celery
import config

# Create the Celery instance for this module, and simultaneously the 
# RPC dictionary which includes RPCs defined in tasks.py for managing 
# asynchronous tasks.
celery_instance, RPC_dict = make_celery(config=config)

# This is needed (or creation of a Task class instead) to allow run_task() to 
# be found on Windows using celery version 3.1.25.  I have NO idea why!  :-(
@celery_instance.task
def dummy_result():
    return 'here be dummy result'

@register_async_task
def async_add(x, y):
    time.sleep(10)
    return x + y

@register_async_task
def async_sub(x, y):
    time.sleep(10)
    return x - y