# To allow graphs to be created without a DISPLAY variable -- WARNING, maybe a cleaner way of doing this?
try:
	import matplotlib
	matplotlib.use('agg')
except:
	print('Sciris initialization: could not set Matplotlib backend, proceeding anyway...')