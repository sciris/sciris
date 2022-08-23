Welcome to Sciris
=================

What is Sciris?
---------------

Glad you asked! **Sciris** (http://sciris.org) is a library of tools that make it faster and more pleasant to write scientific Python code. Built on top of `NumPy <https://numpy.org/>`__ and `Matplotlib <https://matplotlib.org/>`__, Sciris provides functions covering a wide range of common array and plotting operations. This means you can get more done with less code, and spend less time looking things up on StackOverflow.

**ScirisWeb** is an extension of Sciris that allows you to build Python webapps without reinventing the wheel – kind of like `Shiny <https://shiny.rstudio.com/>`__ for Python. In contrast to `Plotly Dash <https://plotly.com/dash/>`__ and `Streamlit <https://www.streamlit.io/>`__, which have limited options for customization, ScirisWeb is completely modular, so you have control over which tools to use for which aspects of the project. Out of the box, ScirisWeb provides a "just works" solution using `Vuejs <https://vuejs.org/>`__ for the frontend, `Flask <https://flask.palletsprojects.com/>`__ as the web framework, `Redis <https://redis.io/>`__ for the (optional) database, and Matplotlib/`mpld3 <https://github.com/mpld3/mpld3>`__ for plotting. But if you want a React frontend linked to an SQL database with Plotly figures, ScirisWeb can serve as the glue holding all of that together.

Sciris is available on `PyPi <https://pypi.org/project/sciris/>`__ (``pip install sciris``) and `GitHub <https://github.com/sciris/sciris>`__. Full documentation is available at http://docs.sciris.org. If you have questions, feature suggestions, or would like some help getting started, please reach out to us at info@sciris.org.


Highlights
~~~~~~~~~~
Some highlights of Sciris (``import sciris as sc``):

- **Powerful containers** – The ``sc.odict`` class is what ``OrderedDict`` (almost) could have been, allowing reference by position or key, casting to a NumPy array, sorting and enumeration functions, etc.
- **Array operations** – Want to find the indices of an array that match a certain value or condition? ``sc.findinds()`` will do that. How about just the nearest value, regardless of exact match? ``sc.findnearest()``. What about the last matching value? ``sc.findlast()``. Yes, you could do ``np.nonzero()[0][-1]`` instead, but ``sc.findlast()`` is easier to read, type, and remember, and handles edge cases more elegantly.
- **File I/O** – One-liner functions for saving and loading text, JSON, spreadsheets, or even arbitrary Python objects.
- **Plotting recipes** – Simple functions for mapping sequential or qualitative data onto colors, manipulating color data, and updating axis limits and tick labels, plus several new colormaps.

Some highlights of ScirisWeb (``import scirisweb as sw``):

-  **ScirisApp** – An extension of a Flask App that can be created as simply as ``app = sw.ScirisApp(config)`` and run with ``app.run()``.
-  **RPCs** – Simple "remote procedure calls" that define how the frontend (web interface) interacts with the backend (Python server).
-  **Datastore** – For more fully-featured webapps, user and data management are available based on Redis (with additional options for SQL or file-based databases).


I'm not convinced.
~~~~~~~~~~~~~~~~~~
That's OK. Perhaps you'd be interested in seeing what a script that performs tasks like parallelization, saving and loading files, and 3D plotting looks like when written in "`vanilla Python <https://github.com/sciris/sciris/blob/develop/tests/showcase_vanilla.py>`__" (left) compared to `using Sciris <https://github.com/sciris/sciris/blob/develop/tests/showcase.py>`__ (right):

|Sciris showcase|

Both of these do the same thing, but the plain Python version requires 50% more lines of code to produce the same graph:

|Sciris output|


Where did Sciris come from?
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Development of Sciris began in 2014 to support development of the `Optima <http://optimamodel.com>`__ suite of models. We kept encounting the same issues and inconveniences over and over while building scientific webapps, and began collecting the tools we used to overcome these issues into a shared library. This library evolved into Sciris. (Note: while "Sciris" doesn't mean anything, "iris" means "rainbow" in Greek, and the name was loosely inspired by the wide spectrum of scientific computing features included in Sciris.)

To give a based-on-a-true-story example, let's say you have a dictionary of results for multiple runs of your model, called ``results``. The output of each model run is itself a dictionary, with keys such as ``name`` and ``data``. Now let's say you want to access the data from the first model run. Using plain Python dictionaries, this would be ``results[list(results.keys())[0]]['data']``. Using a Sciris ``objdict``, this is ``results[0].data`` – almost 3x shorter.


Is Sciris ready yet?
~~~~~~~~~~~~~~~~~~~~
**Yes.** Sciris is currently used by a number of scientific computing libraries, including `Atomica <http://atomica.tools>`__ and `Covasim <http://covasim.org>`__. ScirisWeb provides the backend for webapps such as the `Cascade Analysis Tool <http://cascade.tools>`__, `HIPtool <http://hiptool.org>`__, and `Covasim <http://app.covasim.org>`__. Note that Sciris is still undergoing rapid development, and ScirisWeb, while functional, is still in beta development.


Features
-------------------

Here are a few more of the most commonly used features.

Containers
~~~~~~~~~~
-  ``sc.odict()``: flexible container representing the best-of-all-worlds across lists, dicts, and arrays
-  ``sc.objdict()``: like an odict, but allows get/set via e.g. ``foo.bar`` instead of ``foo['bar']``

Array operations
~~~~~~~~~~~~~~~~
-  ``sc.findinds()``: find indices of an array matching a value or condition
-  ``sc.findnearest()``: find nearest matching value
-  ``sc.smooth()``: simple smoothing of 1D or 2D arrays
-  ``sc.isnumber()``: checks if something is any number type
-  ``sc.promotetolist()``: converts any object to a list, for easy iteration
-  ``sc.promotetoarray()``: tries to convert any object to an array, for easy use with NumPy

File I/O
~~~~~~~~
-  ``sc.save()/sc.load()``: efficiently save/load any Python object (via pickling)
-  ``sc.savejson()/sc.loadjson()``: likewise, for JSONs
-  ``sc.thisdir()``: get current folder
-  ``sc.getfilelist()``: easy way to access glob

Plotting
~~~~~~~~
-  ``sc.hex2rgb()/sc.rgb2hex()``: convert between different color conventions
-  ``sc.vectocolor()``: map a list of sequential values onto a list of colors
-  ``sc.gridcolors()``: map a list of qualitative categories onto a list of colors
-  ``sc.plot3d()/sc.surf3d()``: easy way to render 3D plots
-  ``sc.boxoff()``: turn off top and right parts of the axes box
-  ``sc.commaticks()``: convert labels from "10000" and "1e6" to "10,000" and "1,000,0000"
-  ``sc.SIticks()``: convert labels from "10000" and "1e6" to "10k" and "1m"
-  ``sc.maximize()``: make the figure fill the whole screen
-  ``sc.savemovie()``: save a sequence of figures as an MP4 or other movie

Parallelization
~~~~~~~~~~~~~~~
-  ``sc.parallelize()``: as-easy-as-possible parallelization
-  ``sc.loadbalancer()``: very basic load balancer

Other utilities
~~~~~~~~~~~~~~~
-  ``sc.readdate()``: convert strings to dates using common formats
-  ``sc.tic()/sc.toc()``: simple method for timing durations
-  ``sc.runcommand()``: simple way of executing shell commands (shortcut to ``subprocess.Popen()``)
-  ``sc.dcp()``: simple way of copying objects (shortcut to ``copy.deepcopy()``)
-  ``sc.pr()``: print full representation of an object, including methods and each attribute
-  ``sc.heading()``: print text as a 'large' heading
-  ``sc.colorize()``: print text in a certain color
-  ``sc.sigfigs()``: truncate a number to a certain number of significant figures


Installation and run instructions
---------------------------------


5-second quick start guide
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install Sciris: ``pip install sciris``

2. Use Sciris: ``import sciris as sc``


20-second quick start guide (for ScirisWeb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download ScirisWeb (e.g. ``git clone http://github.com/sciris/scirisweb``)

2. Install ScirisWeb (which will install Sciris as well): ``cd scirisweb; pip install -e .``

3. Change to the Hello World folder: ``cd examples/helloworld``

4. Run the app: ``python app.py``

5. Go to ``localhost:8080`` in your browser

6. Have fun!


Medium-quick start guide
~~~~~~~~~~~~~~~~~~~~~~~~

Note: if you're a developer, you'll likely already have some/all of these packages installed.

1. Install `NodeJS <https://nodejs.org/en/download/>`__ (JavaScript manager)

2. Install `Redis <https://redis.io/topics/quickstart>`__ (database)

3. Install `Anaconda Python <https://www.anaconda.com/download/>`__ (scientific Python environment)

4. Clone and install Sciris: ``git clone http://github.com/sciris/sciris``

5. Clone ScirisWeb: ``git clone http://github.com/sciris/scirisweb``

6. Once you've done all that, to install, simply run ``python setup.py develop`` in the root folders of ``sciris`` and ``scirisweb``. This should install Sciris and ScirisWeb as importable Python modules.

To test, open up a new Python window and type ``import sciris`` (and/or ``import scirisweb``)

If you have problems, please email info@sciris.org, or consult the rest of this guide for more information.


Installing on Linux
~~~~~~~~~~~~~~~~~~~

The easiest way to install Sciris is by using pip: ``pip install scirisweb`` (which will also automatically install ``sciris``). If you want to install from source, follow these steps:

1. Install Git: ``sudo apt install git``

2. Install NodeJS: ``sudo apt install nodejs``

3. Install Redis: https://redis.io/topics/quickstart

4. (Optional) Install `Anaconda Python <https://www.anaconda.com/download/>`__ (as of version 0.15, Sciris is only compatible with Python 3), and make sure it's the default Python, e.g.

::

   your_computer:~> python
   Python 3.7.4 (default, Aug 13 2019, 20:35:49)
   [GCC 7.3.0] :: Anaconda, Inc. on linux
   Type "help", "copyright", "credits" or "license" for more information.

5. Clone the Sciris repositories:
   ``git clone http://github.com/sciris/sciris.git`` and
   ``git clone http://github.com/sciris/scirisweb.git``.

6. Run ``python setup.py develop`` in each of the two Sciris folders.

7. To test, open up a new Python window and type ``import sciris`` and
   ``import scirisweb``. You should see something like:

::

   >>> import sciris
   >>> import scirisweb


Installing on Windows
~~~~~~~~~~~~~~~~~~~~~


Package and library dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, make sure that you have ``npm`` (included in Node.js installation) and ``git`` installed on your machine.

Install `Anaconda Python <https://www.anaconda.com/download/>`__. In your Python setup, you also need to have the following packages (instructions in parentheses show how to install with Anaconda Python environment already installed). **Note**, these should all be installed automatically when you type ``python setup.py develop`` in the Sciris and ScirisWeb folders.


Database dependencies
^^^^^^^^^^^^^^^^^^^^^

If you use Redis as your DataStore mode, you will need to have Redis installed on your computer (as a service). Redis does not directly support Windows, but there is a `MicrosoftArchive page on GitHub <https://github.com/MicrosoftArchive/redis>`__ where you may go for installation directions on your Windows machine. (For example, it can be installed at `this site <https://github.com/MicrosoftArchive/redis/releases>`__ , downloading a .msi file). It ends up being installed as a service which you can navigate to by going the Windows Task Manager and going to the Services tab. Make sure the ``Redis`` service is in the Running state.

Most likely, the directory for your Redis executables will be installed at ``C:\Program Files\Redis``. In that directory, you can double-click the icon for ``redis-cli.exe`` to start the redis database command line interface at the default Redis database (#0). You can do ``keys *`` to look at all of the store key / value pairs in the database, and ``exit`` exits the interface.

You will probably want to use a non-default (i.e. ``N`` is not 0) database. To investigate what keys are in, for example, database #2, while you are within ``redis-cli``, you can type ``select 2`` to switch to that database.


Installing on Mac
~~~~~~~~~~~~~~~~~

1. Install Git. This can be done by installing Xcode commandline tools.

   ::

           xcode-select --install

2. Install NodeJS. Visit https://nodejs.org/en/download/ and download the Mac version and install.

3. Install Redis: https://redis.io/topics/quickstart or run (Assumming brew is installed)

   ::

           brew install redis

4. Install `Anaconda Python 3 <https://www.anaconda.com/download/>`__, and make sure it's the default Python, e.g.

::

   your_computer:~> python
   Python 3.7.4 (default, Aug 13 2019, 20:35:49)
   [GCC 7.3.0] :: Anaconda, Inc. on linux
   Type "help", "copyright", "credits" or "license" for more information.

5.  Create a directory that will hold Sciris. For reference purposes we will create and refer to that directory as ``pyenv``.

6.  Clone the Sciris repository into ``pyenv``:
    ``git clone http://github.com/sciris/sciris.git``

7.  Create a Python virtual environment (venv) inside the directory of your choice. This will be the parent of the Sciris folder.

    ::

        `virtualenv venv`

    More information about `python virtual environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__ can be found `here <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__. The project structure should be as follows;

    ::

                -pyenv
                    -venv
                    -sciris

8.  Get into the virtual environment. While inside the ``pyenv`` folder, to activate the virtual environment, type:

    ::

            ./venv/bin/activate

9.  Change to the Sciris root folder and type:

    ::

       python setup.py develop

10. Repeat in the ScirisWeb root folder:

::

   python setup.py develop

11. To test if the if everything is working accordingly, open Python window within the virtual environment and type ``import sciris`` and ``import scirisweb``. If no errors occur, then the import worked.


Multhreaded deployment
----------------------

The problem with the simple deployment method described above is that requests are single-threaded. If this is an issue, recommended deployment is using ``nginx`` to serve the static files, and ``gunicorn`` to run the Flask app. Note that it is common for an application to call several RPCs with each page load. This means that the multithreaded deployment can result in improved site performance even for a single user.


Requirements
~~~~~~~~~~~~

You must have nginx (``sudo apt install nginx``) and gunicorn
(``pip install gunicorn``) installed.


Set up nginx
~~~~~~~~~~~~

1. Copy ``examples/gunicorn/example_nginx_config`` to e.g.
   ``/etc/nginx/sites-enabled/my_app`` (can change filename if desired)
2. Edit the copied file to specify

   -  The hostname/URL for the site e.g. ``my_app.com``
   -  The full path to the directory containing ``index.html`` on the
      system running ``nginx``
   -  Change the port in ``proxy_pass`` line if desired - it must match
      the port in ``launch_gunicorn``

3. Reload or restart ``nginx`` e.g. ``sudo service nginx reload``

For example, this will start it running at ``localhost:8188``:

.. code:: bash

   server {
       listen 8188;
       server_name localhost;
       location / {
           root /home/my_username/my_sciris_app;
       }
       location /api {
           proxy_pass http://127.0.0.1:8097/;
       }
   }


Run gunicorn
~~~~~~~~~~~~

1. Copy ``examples/gunicorn/example_launch_gunicorn`` to the folder with your app (e.g. ``launch_my_app_gunicorn``), and set the number of workers as desired - usual recommendation is twice the number of CPUs but for applications that are CPU bound (e.g., an RPC call runs a model) then it may be better to reduce it to just the number of CPUs.
2. The example script references the Flask app using ``name_of_your_app:flask_app``. The ``name_of_your_app`` should be importable in Python (either via running Python in the current directory, or installing as a package via ``pip``) and ``flask_app`` is the name of a variable containing the Flask application. So for  example, you might have a file ``foo.py`` containing

.. code:: python

   app = sw.ScirisApp(__name__, name="My App")
   the_app = app.flask_app

in which case the ``launch_my_app_gunicorn`` script should contain ``foo:the_app`` instead of ``name_of_your_app:flask_app``.

3. Run ``launch_my_app_gunicorn``. This will need to be kept running to support the site (so run via ``nohup`` or ``screen`` etc.).

For example:

.. code:: bash

   cd my_app
   screen -S my_app_session
   ./launch_my_app_gunicorn
   <you can now close the terminal>

   ...

   <coming back later, you can restart it with>
   screen -R my_app_session

Note that for local development, you can add the ``--reload`` flag to the ``gunicorn`` command to automatically reload the site. This can be helpful if using the ``nginx+gunicorn`` setup for local development.


Examples
--------

In the ``examples`` and ``vue_proto_webapps`` directories are contained a number of working examples of web applications combining Vue, Flask, and Twisted. These are being used as stepping stones for developing the main framework based in ``user_interface``, ``session_manager``, ``model_code``, and ``bin``.


Hello World
~~~~~~~~~~~

A very simple test case of Sciris. In the ``examples/helloworld`` folder, type ``python app.py``. If you go to ``localhost:8080`` in your browser, it should be running a simple Python webapp.

See the directions `here <https://github.com/sciris/scirisweb/tree/develop/examples/helloworld>`__ on how to install and run this example.

.. |Sciris showcase| image:: https://github.com/sciris/sciris/raw/develop/docs/sciris-showcase-code.png
.. |Sciris output| image:: https://github.com/sciris/sciris/raw/develop/docs/sciris-showcase-output.png