'''
Save the "DeadClass" object in order to load it in the tests.
'''

import deadclass as dc
import sciris as sc

deadclass = dc.DeadClass(238473)
sc.save('deadclass.obj', deadclass)

print('Done')