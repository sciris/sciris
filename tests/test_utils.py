import sciris as sc

# Test colorize
sc.colorize(showhelp=True)
sc.colorize('green', 'hi') # Simple example
sc.colorize(['yellow', 'bgblack']); print('Hello world'); print('Goodbye world'); sc.colorize('reset') # Colorize all output in between
bluearray = sc.colorize(color='blue', string=str(range(5)), output=True); print("c'est bleu: " + bluearray)
sc.colorize('magenta') # Now type in magenta for a while
print('this is magenta')
sc.colorize('reset') # Stop typing in magenta