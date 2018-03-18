# To allow graphs to be created without a DISPLAY variable -- WARNING, maybe a cleaner way of doing this?
try:
    import os
    import matplotlib
    if 'DISPLAY' in os.environ:
        if not os.environ['DISPLAY']:
            matplotlib.use('agg')
except:
	print('Sciris initialization: could not set Matplotlib backend, proceeding anyway...')