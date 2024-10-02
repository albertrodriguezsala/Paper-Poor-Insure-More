# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:18:55 2018

@author: Albert
"""

# =============================================================================
#  Panel data UNPS (5 waves): 2009-10 to 2015-16 
# =============================================================================

'''
DESCRIPTION
        imports the datasets for each wave dataXX.csv and dataXX_rural.csv
        and outputs
        unbalanced panel all Uganda: panel_UGA.csv
        unbalanced panel rural Uganda: panel_rural_UGA.csv
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
dirct  = Path('Master_data.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import  data_stats
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

folder =  str(dirct)+'/data/'

import warnings
warnings.filterwarnings('ignore')

# NATIONWIDE----------------
#Import 2009
data09 = pd.read_csv(folder+'data09/data09.csv')
#Import 2010
data10 = pd.read_csv(folder+'data10/data10.csv')
#Import 2011
data11 = pd.read_csv(folder+'data11/data11.csv')
#Import 2013
data13 = pd.read_csv(folder+'data13/data13.csv')
#Import 2015
data15 = pd.read_csv(folder+'data15/data15.csv')


dollars = 2586.89 

data13['HHID'] = data13["hh"].str.slice(0, 6, 1) + data13["hh"].str.slice(10, 12, 1)
del data13['hh'], data15['hh']
data_ids = data13[['HHID','HHID_old']].merge(data15[['HHID']], on='HHID', how='inner')

pd.value_counts(data_ids['HHID'])
data_ids['HHID_old'].fillna(data_ids['HHID'].str.slice(1,8,1), inplace=True)
data_ids.rename(columns={'HHID_old':'hh'}, inplace=True)
pd.value_counts(data_ids['hh'])
data_ids['hh'] = pd.to_numeric(data_ids['hh'])
## Get id back to the data
data13 = data13.merge(data_ids, on='HHID', how='left')
del data13['HHID']
data15 = data15.merge(data_ids, on='HHID', how='left')
del data15['HHID']
# Create panel



counthh_xwave = []
i=0
for data in [data09, data10, data11, data13, data15]:
    i+=1
    counthh = pd.value_counts(data['hh'])
    #print('wave'+str(i)+' count hh ='+str(max(counthh)))
    counthh_xwave.append(counthh)

data13 = data13.drop_duplicates(subset=['hh'],keep='first')
data15 = data15.drop_duplicates(subset=['hh'],keep='first')

panel = data09.append(data10)
panel = panel.append(data11)
panel = panel.append(data13)
panel = panel.append(data15)

panel['y_agric'] = panel['revenue_agr_p_c_district']


panel.to_csv(folder+"panel/panel_UGA.csv", index=False)

