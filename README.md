# Sciris Framework for Scientific Web Application Development

The Sciris framework is intended to be a flexible open source framework
for building web applications based on Vue.js and Python.

Presently we are developing prototypes for using Vue and Python
together.


## Installation and Run Instructions

### Installing on Linux Systems

[needs to be written...]

### Installing on Windows Systems

#### Package and Library Dependencies

Make sure that you have `npm` and `git` installed on your machine.  In 
your Python setup, you also need to have the following packages (instructions
in parentheses show how to install with Anaconda Python environment already 
installed):
* numpy (already installed under Anaconda)
* matplotlib (already installed under Anaconda)
* flask (already installed under Anaconda)
* flask-login (`conda install flask-login`)
* twisted (`conda install twisted`)
* mpld3 (`conda install mpld3`)
* redis (`pip install redis`)

#### Database Dependencies

If you use Redis as your DataStore mode, you will need to have Redis installed 
on your computer (as a service).  Redis does not directly support Windows, 
but there is a [MicrosoftArchive page on GitHub](https://github.com/MicrosoftArchive/redis) 
where you may go for installation directions on your Windows machine. 
(For example, it can be installed at [this site](https://github.com/MicrosoftArchive/redis/releases)
, downloading a .msi file).  It 
ends up being installed as a service which you can navigate to by going 
the Windows Task Manager and going to the Services tab.  Make sure the `Redis` 
service is in the Running state.

(For Linux installations, you can probably use the 
[Redis Quick Start](https://redis.io/topics/quickstart) site directions.)

Most likely, the directory for your Redis executables will be installed at 
`C:\Program Files\Redis`.  In that directory, you can double-click the icon 
for `redis-cli.exe` to start the redis database command line interface at 
the default Redis database (#0).  You can do `keys *` to look at all of the 
store key / value pairs in the database, and `exit` exits the interface.  
Most likely, you will want to use a non-default (i.e. `N` is not 0) 
database.  To investigate what keys are in, for example, database #2, 
while you are within `redis-cli`, you can type `select 2` to switch to that 
database.

#### Initial Build of the Client

After cloning or pulling this repo, use the following steps to do the 
initial build of the app:
* `cd vueinterface`
* `npm install` builds the Node modules the app uses.  This step can take 
a few minutes to complete.
* `npm run build` generates the build version of the app.

#### Running Using the Build Version

* `cd bin`
* `win_buildrun` brings up the server hosting both the 
client / UI and server-side code.
* Now you can go into your browser and navigate to http://localhost:8080.
You may possibly need to refresh/reload, but you should see the UI 
when the app is up.
* If you make changes to the Python `sessionmanager` or `scirismodel` code, 
the Twisted/Flask server will automatically restart so changes can be 
reflected in the app behavior.
* Use `Ctrl-C` to end the Twisted session, taking down the server. (You can 
answer `n` to the "Terminate batch job" query.)

#### Running Using the Webpack Development Server

The process for using the Webpack dev server (which has the benefit of 
allowing hot-reloading of client files when you edit them during development) 
is somewhat different: it does not use Twisted but instead sets up the Flask 
server directly on Port 5000, and the Webpack dev server sets up a server 
for the Vue web-pages on Port 8080, and also sets up a proxy to send the 
RPC requests to Port 5000.  To set this up, in a first terminal window, do 
the following:
* `cd bin`
* `win_devserver` starts the Flask server in Port 5000.
* `Ctrl-C` closes down the Flask server when you are finished using the dev 
server site. (You can answer `n` to the "Terminate batch job" query.)

With the first window running the Flask server, in a second terminal window, 
do the following:
* `cd bin`
* `win_devclient` compiles the code and brings up the Webpack dev server and 
automatically opens a browser window pointed to the web page.
* Presently, changes to `sessionmanager` or `scirismodel` code will not 
cause auto-reloading when the dev-server is used, but potentially this will 
change in the future.
* `Ctrl-C` shuts down the dev server and proxy when you are finished with 
the web site.  (You can answer `n` to the "Terminate batch job" query.)

#### Rebuilding the Build Version After Development Changes

* `cd bin`
* `win_build` generates the (new) build version of the app.

### Installing on Mac OSX Systems

[needs to be written...]


## Vue/Python Code Examples

In the `examples` and `vue_proto_webapps` directories are contained a number 
of working examples of web applications combining Vue, Flask, and Twisted. 
These are being used as stepping stones for developing the main framework 
based in `user_interface`, `session_manager`, `model_code`, and `bin`.

### Hello World

A very simple test case of Sciris. In the `examples/helloworld` folder, type `python server.py`. If you go to `localhost:8080` in your browser, it should be running a simple Python webapp.

### Random Scatterplotter

A slightly more sophisticated version of the previous app, this creates a page with a simple button in the browser.  When the user
presses this in the browser client, the server generates a random matplotlib
scatterplot, which it then returns to the client for display.  This
demonstrates basic communication between a Vue client and a Flask/Twisted
server has been enabled.  See `vue_proto_webapps/scatterplotter_p1/README.md` 
for detailed build and run instructions.
