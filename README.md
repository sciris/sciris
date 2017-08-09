# Sciris Framework for Scientific Web Application Development

The Sciris framework is intended to be a flexible open source framework 
for building web applications based on Vue.js and Python.

Presently I (gchadder3) am developing prototypes for using Vue and Python 
together.  

The first is at

## `vue_proto_webapps/scatterplotter_p1` App

This creates a page with a simple button in the browser.  When the user 
presses this in the browser client, the server generates a random matplotlib 
scatterplot, which it then returns to the client for it to display.  This 
demonstrates basic communication between a Vue client and a Flask/Twisted 
server has been enabled.

After cloning or pulling this repo, the following steps start up the app:
* `cd vue_proto_webapps/scatterplotter_p1/vue_client1b`
* `npm install`
* `npm run build`
* `cd ../flask_server`
* `python _twisted_wsgi.py 8080`
* Now you can go into your browser and navigate to `http://localhost:8080'. 
You may possibly need to refresh/reload, but you should see the button 
when the app is up.


