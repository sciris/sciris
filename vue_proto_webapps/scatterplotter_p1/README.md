# Random Scatterplotter

This creates a page with a simple button in the browser.  When the user
presses this in the browser client, the server generates a random matplotlib
scatterplot, which it then returns to the client for display.  This
demonstrates basic communication between a Vue client and a Flask/Twisted
server has been enabled.

## Initial Build of the App

After cloning or pulling this repo, the following steps start up the app:
* `cd vue_proto_webapps/scatterplotter_p1/vue_client`
* `npm install` builds the Node modules the app uses.  This step can take 
significant time.
* `npm run build` generates the build version of the app.

## Running Using the Build Version

* `cd vue_proto_webapps/scatterplotter_p1/flask_server`
* `python _twisted_wsgi.py 8080` brings up the server hosting both the 
client / UI and server-side code.
* Now you can go into your browser and navigate to http://localhost:8080 .
You may possibly need to refresh/reload, but you should see the button
when the app is up.
* Use `Ctrl-C` to end the Twisted session, taking down the server.

## Running Using the Webpack Development Server

The process for using the Webpack dev server (which has the benefit of 
allowing hot-reloading of client files when you edit them during development) 
is somewhat different: it does not use Twisted but instead sets up the Flask 
server directly on Port 5000, and the Webpack dev server sets up a server 
for the Vue web-pages on Port 8080, and also sets up a proxy to send the 
RPC requests to Port 5000.  To set this up, in a first terminal window, does
* `cd vue_proto_webapps/scatterplotter_p1/flask_server`
* `python api.py` starts the Flask server in Port 5000.
* `Ctrl-C` closes down the Flask server when you are finished using the dev 
server site.

Then in a second terminal window, do the following:
* `cd vue_proto_webapps/scatterplotter_p1/vue_client`
* `npm run dev` compiles the code and brings up the Webpack dev server and 
automatically opens a browser window pointed to the web page.
* `Ctrl-C` shuts down the dev server and proxy when you are finished with 
the web site.  (You can answer `n` to the "Terminate batch job" query.)

## Rebuilding the Build Version After Development Changes

* `cd vue_proto_webapps/scatterplotter_p1/vue_client`
* `npm run build` generates the (new) build version of the app.