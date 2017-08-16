# Sciris Framework for Scientific Web Application Development

The Sciris framework is intended to be a flexible open source framework
for building web applications based on Vue.js and Python.

Presently we are developing prototypes for using Vue and Python
together.


## Installation and Run Instructions

### Installing on Linux Systems

[needs to be written...]

### Installing on Windows Systems

#### Package and Library Depedencies

Make sure that you have `npm` and `git` installed on your machine.  In 
your Python setup, you also need to have the following packages:
* numpy
* matplotlib
* flask
* twisted
* mpld3

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
