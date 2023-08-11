'''
Script to generate test data files for checking load compatibility.

Instructions:

    1. Create a new conda environment, install pypi-timemachine, and start a server.
    For example, for 2022:

        conda create -n tm python=3.9 -y 
        conda activate tm
        pip install pypi-timemachine
        pypi-timemachine 2022-01-01 # shows port being used

    2. In a separate terminal, install Sciris and run this script:

        conda activate tm
        pip install --index-url http://localhost:<PORT> sciris
        python make_archive.py 2022-01-01

This has been run with the following arguments:
    # For the archive
    python make_archive.py 2023-04-19 1
    
    # For the pickles
    python make_archive.py 2021-01-01 0
    python make_archive.py 2022-01-01 0

Version history:
    2023-04-19: Original version
    2023-08-06: Updated to save pickles and additional pandas data
'''

import sys
import sciris as sc
ut = sc.importbypath(sc.thispath() / '..' / 'sc_test_utils.py')

# If using pypi-timemachine, set the corresponding date here
date = '2023-08-11'
as_archive = True

if len(sys.argv) > 1:
    date = sys.argv[1]
if len(sys.argv) > 2:
    as_archive = int(sys.argv[2])

myclass = ut.MyClass(date=date)

if as_archive:
    sc.savearchive(f'archive_{date}.zip', myclass)
else:
    sc.saveobj(f'pickle_{date}.obj', myclass.__dict__)