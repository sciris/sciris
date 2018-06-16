"""
apptasks.py -- The Celery tasks module for this webapp
    
Last update: 6/16/18 (gchadder3)
"""

import time
from sciris.weblib.tasks import register_async_task
from sciris.weblib.tasks import make_celery
import config

import celery

#celery_instance, RPC_dict = make_celery(config=config)
celery_instance, RPC_dict, run_task = make_celery(config=config)
#celery_instance, RPC_dict, RunTask = make_celery(config=config)

class RunTask(celery.Task):
    name = "tasks.run_task"
    
    def run(self, task_id, func_name, args, kwargs):
        return run_task(task_id, func_name, args, kwargs)
        
celery_instance.tasks.register(RunTask)

#@celery_instance.task
#def async_mult(x, y):
#    time.sleep(60)
#    return x * y

@register_async_task
def async_add(x, y):
    time.sleep(60)
    return x + y

@register_async_task
def async_sub(x, y):
    time.sleep(60)
    return x - y