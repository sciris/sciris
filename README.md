# Welcome to Sciris


## What is Sciris?

Glad you asked! Sciris is a flexible open source framework for building scientific web applications using Python and JavaScript. It comes in two parts: `sciris` is a collection of tools that should make scientific Python coding a more pleasant experience, while `scirisweb` is a collection of tools that allow you to easily build Python webapps. Sciris is built on Numpy and Matplotlib, while ScirisWeb is built on Vue.js, Flask, Twisted, Redis, and `mpld3`.

Some highlights of `sciris`:
* `odict` and `objdict` -- like an OrderedDict, but allows reference by position like a list, as well as many powerful methods (such as casting to array, sorting and enumeration functions, etc.). For example, instead of `my_plain_dict[list(my_plain_dict.keys())[0]]['value']`, you can use `my_obj_dict[0].value`.
* `promotetoarray` -- standardizes any kind of numeric input to a Numpy array, so e.g. `1`, `[1]`, `(1,)` etc. are all converted to `array([1])`
* `checktype` -- quickly determine the type of the input, e.g. `checktype([1,2,3], 'arraylike', subtype='number') # returns True`
* `findnearest` -- find the element of an array closest to the input value
* `loadobj`, `saveobj` -- flexible methods to save/load arbitrary Python objects
* `vectocolor` -- map a given vector into a set of colors
* `gridcolors` -- pick a set of colors from maximally distant parts of color-space (e.g. for plots with large numbers of lines)
* `smoothinterp` -- linear interpolation with smoothing
* `asd` -- adaptive stochastic descent, an algorithm for optimizing functions as few function evaluations as possible

Some highlights of `scirisweb`:
* `ScirisApp` -- a fully featured server that can be created as simply as `app = ScirisApp(config)` and run with `app.run()`
* `RPC` -- a simple function for defining links between the frontend (web interface) and the backend (server)
* `Datastore` -- user and data management based on Redis


## Okay, tell me more.

Here are a few more of the most commonly used features.

### Containers
- `sc.odict()` # flexible container for best-of-all-worlds for lists, dicts, and arrays
- `sc.objdict()` # like an odict, but allows get/set via e.g. foo.bar instead of foo['bar']

### File utilities
- `sc.saveobj()/sc.loadobj()` # efficiently save/load any Python object (via pickling)
- `sc.savejson()/sc.loadjson()` # likewise, for JSONs
- `sc.thisdir()` # get current folder
- `sc.getfilelist()` # easy way to access glob

### Basic utilities
- `sc.dcp()` # shortcut to copy.deepcopy()
- `sc.pr()` # print detailed representation of an object
- `sc.heading()` # print text as a 'large' heading
- `sc.colorize()` # print text in a certain color
- `sc.sigfigs()` # truncate a number to a certain number of significant figures
- `sc.isnumber()` # checks if something is any number type
- `sc.promotetolist()` # converts strings or scalars to lists, for consistent iteration
- `sc.readdate()` # convert strings to dates using common formats
- `sc.tic()/sc.toc()` # simple method for timing durations
- `sc.runcommand()` # simple way of executing a shell command
- `sc.findinds()` # find indices of an array matching a condition
- `sc.findnearest()` # find nearest matching value
- `sc.smooth()` # simple smoothing of 1D or 2D arrays

### Plotting utilities
- `sc.hex2grb()/sc.rgb2hex()` # convert between different color conventions
- `sc.vectocolor()` # map a list of sequential values onto a list of colors
- `sc.gridcolors()` # map a list of qualitative categories onto a list of colors
- `sc.plot3d()/sc.surf3d()` # easy way to render 3D plots
- `sc.boxoff()` # turn off top and right parts of the axes box
- `sc.commaticks()` # convert labels from "10000" and "1e6" to "10,000" and "1,000,0000"
- `sc.SIticks()` # convert labels from "10000" and "1e6" to "10k" and "1m"
- `sc.maximize()` # make the figure fill the whole screen
- `sc.savemovie()` # save a sequence of figures as an MP4 or other movie

### Parallelization utilities
- `sc.parallelize()` # as-easy-as-possible parallelization
- `sc.loadbalancer()` # very basic load balancer


## I'm not convinced.

That's OK. Perhaps you'd be interested in seeing what a script that performs tasks like parallelization, saving and loading files, and 3D plotting looks like when written in [vanilla Python](https://github.com/sciris/sciris/blob/develop/tests/showcase_vanilla.py) compared to [using Sciris](https://github.com/sciris/sciris/blob/develop/tests/showcase.py):

![Sciris showcase](/docs/sciris-showcase-code.png)


## Is Sciris ready yet?

**Yes.** Sciris is available for use, but is still undergoing rapid development. We expect an official launch some time during 2021. If you would like us to let you know when this happens, please email info@sciris.org.

## Installation and run instructions

### 5-second quick start guide

1. Install Sciris: `pip install sciris`

2. Use Sciris: `import sciris as sc`

### 20-second quick start guide

1. Download ScirisWeb (e.g. `git clone http://github.com/sciris/scirisweb`)

2. Install ScirisWeb (which will install Sciris as well): `cd scirisweb; python setup.py develop`

3. Change to the Hello World folder: `cd examples/helloworld`

4. Run the app: `python app.py`

5. Go to `localhost:8080` in your browser

6. Have fun!

### Medium-quick start guide

Note: if you're a developer, you'll likely already have some/all of these packages installed.

1. Install [NodeJS](https://nodejs.org/en/download/) (JavaScript manager)

2. Install [Redis](https://redis.io/topics/quickstart) (database)

3. Install [Anaconda Python](https://www.anaconda.com/download/) (scientific Python environment)

4. Clone and install Sciris: `git clone http://github.com/sciris/sciris`

5. Clone ScirisWeb: `git clone http://github.com/sciris/scirisweb`

6. Once you've done all that, to install, simply run `python setup.py develop` in the root folders of `sciris` and `scirisweb`. This should install Sciris and ScirisWeb as importable Python modules.

To test, open up a new Python window and type `import sciris` (and/or `import scirisweb`)

If you have problems, please email info@sciris.org, or consult the rest of this guide for more information.


### Installing on Linux

The easiest way to install Sciris is by using pip: `pip install scirisweb` (which will also automatically install `sciris`). If you want to install from source, follow these steps:

1. Install Git: `sudo apt install git`

2. Install NodeJS: `sudo apt install nodejs`

3. Install Redis: https://redis.io/topics/quickstart

4. (Optional) Install [Anaconda Python](https://www.anaconda.com/download/) (as of version 0.15, Sciris is only compatible with Python 3), and make sure it's the default Python, e.g.
```
your_computer:~> python
Python 3.7.4 (default, Aug 13 2019, 20:35:49)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
```

5. Clone the Sciris repositories: `git clone http://github.com/sciris/sciris.git` and `git clone http://github.com/sciris/scirisweb.git`.

6. Run `python setup.py develop` in each of the two Sciris folders.

7. To test, open up a new Python window and type `import sciris` and `import scirisweb`. You should see something like:
```
>>> import sciris
>>> import scirisweb
>>>
```


### Installing on Windows

#### Package and library dependencies

Make sure that you have `npm` (included in Node.js installation) and `git` installed on your machine.  
First, install [Anaconda Python](https://www.anaconda.com/download/). In your Python setup, you also need to have the following packages (instructions in parentheses show how to install with Anaconda Python environment already installed). **Note**, these should all be installed automatically when you type `python setup.py develop` and `python setup-web.py develop`.

#### Database dependencies

If you use Redis as your DataStore mode, you will need to have Redis installed
on your computer (as a service).  Redis does not directly support Windows,
but there is a [MicrosoftArchive page on GitHub](https://github.com/MicrosoftArchive/redis)
where you may go for installation directions on your Windows machine.
(For example, it can be installed at [this site](https://github.com/MicrosoftArchive/redis/releases)
, downloading a .msi file).  It
ends up being installed as a service which you can navigate to by going
the Windows Task Manager and going to the Services tab.  Make sure the `Redis`
service is in the Running state.

Most likely, the directory for your Redis executables will be installed at
`C:\Program Files\Redis`.  In that directory, you can double-click the icon
for `redis-cli.exe` to start the redis database command line interface at
the default Redis database (#0).  You can do `keys *` to look at all of the
store key / value pairs in the database, and `exit` exits the interface.  
Most likely, you will want to use a non-default (i.e. `N` is not 0)
database.  To investigate what keys are in, for example, database #2,
while you are within `redis-cli`, you can type `select 2` to switch to that
database.


### Installing on Mac

1. Install Git. This can be done by installing Xcode commandline tools.

            xcode-select --install

2. Install NodeJS. Visit https://nodejs.org/en/download/ and download the Mac version and install.

3. Install Redis: https://redis.io/topics/quickstart or run (Assumming brew is installed)

            brew install redis

4. Install [Anaconda Python 3](https://www.anaconda.com/download/), and make sure it's the default Python, e.g.
```
your_computer:~> python
Python 3.7.4 (default, Aug 13 2019, 20:35:49)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
```

5. Create a directory that will hold Sciris. For reference purposes we will create and refer to that directory as `pyenv`.

6. Clone the Sciris repository into `pyenv`: `git clone http://github.com/sciris/sciris.git`

7. Create a Python virtual environment (venv) inside the directory of your choice. This will be the parent of the Sciris folder.

        `virtualenv venv`

    More information about [python virtual environments](http://docs.python-guide.org/en/latest/dev/virtualenvs/) can be found [here](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
   The project structure should be as follows;
   ```
            -pyenv
                -venv
                -sciris
   ```

8. Get into the virtual environment. While inside the `pyenv` folder, to activate the virtual environment, type:
   ```
        ./venv/bin/activate
   ```

9. Change to the Sciris root folder and type:
   ```
   python setup.py develop
   ```

10. Repeat in the ScirisWeb root folder:
   ```
   python setup.py develop
   ```

11. To test if the if everything is working accordingly, open Python window within the virtual environment and type `import sciris` and `import scirisweb`. If no errors occur, then the import worked.


## Multhreaded deployment

The problem with the simple deployment method described above is that requests are single-threaded. If this is an issue, recommended deployment is using `nginx` to serve the static files, and `gunicorn` to run the Flask app. Note that it is common for an application to call several RPCs with each page load. This means that the multithreaded deployment can result in improved site performance even for a single user. 

### Requirements

You must have nginx (`sudo apt install nginx`) and gunicorn (`pip install gunicorn`) installed. 

### Set up nginx

1. Copy `examples/gunicorn/example_nginx_config` to e.g. `/etc/nginx/sites-enabled/my_app` (can change filename if desired)
2. Edit the copied file to specify
    - The hostname/URL for the site e.g. `my_app.com`
    - The full path to the directory containing `index.html` on the system running `nginx`
    - Change the port in `proxy_pass` line if desired - it must match the port in `launch_gunicorn`
3. Reload or restart `nginx` e.g. `sudo service nginx reload`

For example, this will start it running at `localhost:8188`:

```script
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
```

### Run gunicorn

1. Copy `examples/gunicorn/example_launch_gunicorn` to the folder with your app (e.g. `launch_my_app_gunicorn`), and set the number of workers as desired - usual recommendation is twice the number of CPUs but for applications that are CPU bound (e.g., an RPC call runs a model) then it may be better to reduce it to just the number of CPUs.
2. The example script references the Flask app using `name_of_your_app:flask_app`. The `name_of_your_app` should be importable in Python (either via running Python in the current directory, or installing as a package via `pip`) and `flask_app` is the name of a variable containing the Flask application. So for example, you might have a file `foo.py` containing

```python
app = sw.ScirisApp(__name__, name="My App")
the_app = app.flask_app
```
in which case the `launch_my_app_gunicorn` script should contain `foo:the_app` instead of `name_of_your_app:flask_app`.

3. Run `launch_my_app_gunicorn`. This will need to be kept running to support the site (so run via `nohup` or `screen` etc.).

For example:
```script
cd my_app
screen -S my_app_session
./launch_my_app_gunicorn
<you can now close the terminal>

...

<coming back later, you can restart it with>
screen -R my_app_session
```

Note that for local development, you can add the `--reload` flag to the `gunicorn` command to automatically reload the site. This can be helpful if using the `nginx+gunicorn` setup for local development.


## Examples

In the `examples` and `vue_proto_webapps` directories are contained a number
of working examples of web applications combining Vue, Flask, and Twisted.
These are being used as stepping stones for developing the main framework
based in `user_interface`, `session_manager`, `model_code`, and `bin`.

### Hello World

A very simple test case of Sciris. In the `examples/helloworld` folder, type `python app.py`. If you go to `localhost:8080` in your browser, it should be running a simple Python webapp.

See the directions [here](https://github.com/sciris/scirisweb/tree/develop/examples/helloworld) on how to install and run this example.
