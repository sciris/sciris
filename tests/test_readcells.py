import sciris as sc

# Load a sample Nutrition databook to try pulling cells from.
wb = sc.Spreadsheet(filename='nutrition_databook.xlsx')

# thang = wb.readcells(method='xlrd', sheetname='Baseline year population inputs')
# thang2 = wb.readcells(method='openpyexcel', sheetname='Baseline year population inputs', wbargs={'data_only': False})
# thang2b = wb.readcells(method='openpyexcel', sheetname='Baseline year population inputs', wbargs={'data_only': True})

# Grab cells using xlrd.
celltest = wb.readcells(method='xlrd', sheetname='Baseline year population inputs', cells=[[46, 2], [47, 2]])

# Grab cells using openpyexcel.  You have to set wbargs={'data_only': True} to pull out cached values instead of
# formula strings.
celltest2 = wb.readcells(method='openpyexcel', wbargs={'data_only': True},
                        sheetname='Baseline year population inputs', cells=[[46, 2], [47, 2]])

print ('Cell test')
print(celltest)
print ('Cell test2')
print(celltest2)

print('Done.')