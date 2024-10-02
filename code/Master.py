"""
Created on Mon Apr  8 11:00:57 2024

@author: rodri
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master_data.py').resolve().parent
if '\\' in str(dirct):
    print(True)
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

my_wd1 = dirct+'/cleaning data'
my_wd2 = dirct+'/empirics data'


## (1) Clean the data for each wave ==================================

# For each wave, runs the files that clean consumption, agriculture, labor and business income, sociodemographic characteristics and wealth.
# Also runs de file that combines together the datasets from the previous runs into the one wave main dataset: dataWAVE.

for wave in ['09','10','11','13','15']:
    print('===========================================================================')
    print('CLEANING WAVE 20'+wave)
    print('===========================================================================')
    print(' ')
    print('----------------------')
    print('Consumption 20'+wave)
    with open(my_wd1+'/data'+wave+'/cons'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
    
    
    print(' ')
    print('----------------------')
    print('Agriculture 20'+wave)
    print('(1) Creates dataset for household agricultural income  inputs, and wealth')
    print('(2) Creates the crop-plot level dataset for the empirical findings section.')
    with open(my_wd1+'/data'+wave+'/agric'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
    
    print(' ')
    print('----------------------')
    print('Non-agric Earnings 20'+wave)
    with open(my_wd1+'/data'+wave+'/labor_bs'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)

    print(' ')
    print('----------------------')
    print('Sociodemographic charaterstics 20'+wave)
    with open(my_wd1+'/data'+wave+'/sociodem'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
  
    
    print(' ')
    print('----------------------')
    print('Wealth 20'+wave)
    with open(my_wd1+'/data'+wave+'/wealth'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
  
    

    print(' ')
    print('----------------------')
    print('household dataset wave'+wave)
    with open(my_wd1+'/data'+wave+'/data'+wave+'.py', 'r') as file:
        code = file.read()
        exec(code)
   
        
        
#%%(2) Create panels: household level and plot level (for crop analysis) ==================================

print('===========================================================================')
print('CREATES THE PANEL DATASETS')
print('===========================================================================')


print(' ')
print('----------------------')
print('Create Household panel, UNPS 2009-2015.')

text = ''''imports the datasets for each wave dataXX.csv and dataXX_rural.csv and outputs
      (1) unbalanced panel all Uganda: panel_UGA.csv
      (2) unbalanced panel rural Uganda: panel_rural_UGA.csv'''

print(text)
with open(my_wd1+'/panel.py', 'r') as file:
    code = file.read()
    exec(code)





#%% (3) Run the results from the data   ============================================================================
# - crop vs yields vs crop selection empricial findings
# - descriptive statistics in the data
# - moments to target (or compare) the calibration of the model.


print('===========================================================================')
print('CIW SUMMARY STATISTICS')
print('===========================================================================')


print(' ')
print('----------------------')
print('Summary Consumption, Income, Wealth and Data Moments')


with open(my_wd2+'/ciw_summary_avgmoments.py', 'r') as file:
    code = file.read()
    exec(code)


print('===========================================================================')
print('RESULTS: RUNS INSURANCE TEST REGRESSIONS')
print('===========================================================================')

with open(my_wd2+'/insurance_tests_UGA.py', 'r') as file:
    code = file.read()
    exec(code)






    
