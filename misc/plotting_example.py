import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

sc.options(font='Garamond') # Custom font
x = sc.daterange('2022-06-01', '2022-12-31', as_date=True) # Create dates
y = sc.smooth(np.random.randn(len(x))**2*1000) # Create smoothed random numbers
c = sc.vectocolor(y, cmap='turbo') # Set colors proportional to y values
plt.scatter(x, y, c=c) # Plot the data
sc.dateformatter() # Custom date axis formatter
sc.commaticks() # Add commas to y-axis ticks
sc.setylim() # Reset the y-axis limits
sc.boxoff() # Remove the top and right axis spines

plt.show()
sc.savefig('plotting-example.png')