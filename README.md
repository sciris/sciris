# Sciris Framework for Scientific Web Application Development

The Sciris framework is intended to be a flexible open source framework
for building web applications based on Vue.js and Python.

Presently we are developing prototypes for using Vue and Python
together.


## Installation and Run Instructions

### Linux Systems

[needs to be written...]

### Windows Systems

[needs to be written...]

### Mac OSX Systems

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
