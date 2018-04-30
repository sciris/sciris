"""
Simple demonstration Flask webapp.
"""

# Imports
from flask import Flask

#import numpy as np
#import matplotlib.pyplot as plt
#import mpld3
#from mpld3 import plugins
#
#fig, ax = plt.subplots()
#points = ax.scatter(np.random.rand(40), np.random.rand(40),
#                    s=300, alpha=0.3)
#
#labels = ["Point {0}".format(i) for i in range(40)]
#tooltip = plugins.PointLabelTooltip(points, labels)
#
#plugins.connect(fig, tooltip)

# Create the app
app = Flask(__name__) 

# Define the API
@app.route('/api', methods=['GET', 'POST'])
def myPage():
#    return mpld3.fig_to_html(fig)
    return '<h1>Hello, Flask!</h1>'

# Run the webapp
if __name__ == "__main__":
    app.run()