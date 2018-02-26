# Autoreloading launcher.
# Borrowed from Peter Hunt and the CherryPy project (http://www.cherrypy.org).
# Some taken from Ian Bicking's Paste (http://pythonpaste.org/).
# Adjustments made by Michael Elsdoerfer (michael@elsdoerfer.com).
#
# Portions copyright (c) 2004, CherryPy Team (team@cherrypy.org)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of the CherryPy Team nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys, time

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
