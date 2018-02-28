###########################################################################################
#%% IMPORTS
###########################################################################################


# Twisted imports

import sys
import os

from twisted.internet import reactor
from twisted.internet.endpoints import serverFromString
from twisted.logger import globalLogBeginner, FileLogObserver, formatEvent
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.static import File
from twisted.web.wsgi import WSGIResource
from twisted.python.threadpool import ThreadPool

print('STARRRRRRRRRRRRRRRRRRRRTTTING')
import api
print('OKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAy')
#api.runapp()


# Autoreload imports

import time

try:
    import thread
except ImportError:
    import dummy_thread as thread

# This import does nothing, but it's necessary to avoid some race conditions
# in the threading module. See http://code.djangoproject.com/ticket/2330 .
try:
    import threading
    threading
except ImportError:
    pass


# Run the config file.
execfile(os.path.dirname(__file__)+os.sep+'config.py')
print('Warning fix config file path')


###########################################################################################
#%% TWISTED
###########################################################################################

def runtwisted():
    """
    Run the Twisted server.
    """
    globalLogBeginner.beginLoggingTo([
        FileLogObserver(sys.stdout, lambda _: formatEvent(_) + "\n")])

    threadpool = ThreadPool(maxthreads=30)
    app = api.makeapp()
    wsgi_app = WSGIResource(reactor, threadpool, app)

    class OptimaResource(Resource):
        isLeaf = True

        def __init__(self, wsgi):
            self._wsgi = wsgi

        def render(self, request):
            request.prepath = []
            request.postpath = ['api'] + request.postpath[:]

            r = self._wsgi.render(request)

            request.responseHeaders.setRawHeaders(
                b'Cache-Control', [b'no-cache', b'no-store', b'must-revalidate'])
            request.responseHeaders.setRawHeaders(b'expires', [b'0'])
            return r
        
    # If we have a full path for the client directory, use that directory.
    if os.path.isabs(CLIENT_DIR):
        clientDirTarget = CLIENT_DIR
        
    # Otherwise (we have a relative path), use it (correcting so it is with 
    # respect to the sciris repo directory).
    else:
        clientDirTarget = '%s%s%s' % (os.pardir, os.sep, CLIENT_DIR) 
        
    base_resource = File('%s%sdist%s' % (clientDirTarget, os.sep, os.sep))   
    base_resource.putChild('dev', File('%s%ssrc%s' % (clientDirTarget, 
        os.sep, os.sep))) 
    base_resource.putChild('api', OptimaResource(wsgi_app))
    
    site = Site(base_resource)

    try:
        port = str(sys.argv[1])
    except IndexError:
        port = "8091"

    # Start the threadpool now, shut it down when we're closing
    threadpool.start()
    reactor.addSystemEventTrigger('before', 'shutdown', threadpool.stop)

    endpoint = serverFromString(reactor, "tcp:port=" + port)
    endpoint.listen(site)

    reactor.run()




###########################################################################################
#%% AUTORELOAD
###########################################################################################

# Autoreloading launcher.
# Borrowed from Peter Hunt and the CherryPy project (http://www.cherrypy.org).

RUN_RELOADER = True

_mtimes = {}
_win = (sys.platform == "win32")

def code_changed():
    global _mtimes, _win
    for filename in filter(lambda v: v, map(lambda m: getattr(m, "__file__", None), sys.modules.values())):
        if filename.endswith(".pyc") or filename.endswith(".pyo"):
            filename = filename[:-1]
        if not os.path.exists(filename):
            continue # File might be in an egg, so it can't be reloaded.
        stat = os.stat(filename)
        mtime = stat.st_mtime
        if _win:
            mtime -= stat.st_ctime
        if filename not in _mtimes:
            _mtimes[filename] = mtime
            continue
        if mtime != _mtimes[filename]:
            _mtimes = {}
            return True
    return False

def reloader_thread(softexit=False):
    """If ``soft_exit`` is True, we use sys.exit(); otherwise ``os_exit``
    will be used to end the process.
    """
    while RUN_RELOADER:
        if code_changed():
            # force reload
            if softexit:
                sys.exit(3)
            else:
                os._exit(3)
        time.sleep(1)

def restart_with_reloader():
    while True:
        args = [sys.executable] + sys.argv
        if sys.platform == "win32":
            args = ['"%s"' % arg for arg in args]
        new_environ = os.environ.copy()
        new_environ["RUN_MAIN"] = 'true'
        exit_code = os.spawnve(os.P_WAIT, sys.executable, args, new_environ)
        if exit_code != 3:
            return exit_code

def python_reloader(main_func, args, kwargs, check_in_thread=True):
    """
    If ``check_in_thread`` is False, ``main_func`` will be run in a separate
    thread, and the code checker in the main thread. This was the original
    behavior of this module: I (Michael Elsdoerfer) changed the default
    to be the reverse: Code checker in thread, main func in main thread.
    This was necessary to make the thing work with Twisted
    (http://twistedmatrix.com/trac/ticket/4072).
    """
    if os.environ.get("RUN_MAIN") == "true":
        if check_in_thread:
            thread.start_new_thread(reloader_thread, (), {'softexit': False})
        else:
            thread.start_new_thread(main_func, args, kwargs)

        try:
            if not check_in_thread:
                reloader_thread(softexit=True)
            else:
                main_func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
    else:
        try:
            sys.exit(restart_with_reloader())
        except KeyboardInterrupt:
            pass

def jython_reloader(main_func, args, kwargs):
    from _systemrestart import SystemRestart
    thread.start_new_thread(main_func, args)
    while True:
        if code_changed():
            raise SystemRestart
        time.sleep(1)


def main(main_func, args=None, kwargs=None, **more_options):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sys.platform.startswith('java'):
        reloader = jython_reloader
    else:
        reloader = python_reloader
    reloader(main_func, args, kwargs, **more_options)


###########################################################################################
#%% START THE SERVER
###########################################################################################


def start(*args, **kwargs):
    ''' Start the server by combining the autoreload with the Twisted server '''
    main(runtwisted, args=args, kwargs=kwargs)
    return None