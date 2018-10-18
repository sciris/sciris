import pylab as pl
import scirisweb as sw
app = sw.ScirisApp(__name__, name="helloworld")

@app.flask_app.route("/")
def hello():
    output = 'Hello world!<br><br>Click <a href="/number">here</a> for a random number.'
    return output

@app.flask_app.route("/number")
def number():
    number = pl.rand()
    output = 'Random number: %s.<br><br>Click <a href="/">here</a> to go back.' % number
    return output

# Run the webapp
if __name__ == "__main__":
    app.run(with_twisted=False)