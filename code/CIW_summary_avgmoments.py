# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:50:07 2023

@author: rodri
"""

##### CIW SUMMARY TABLES

import pandas as pd
import numpy as np
import os
pd.options.display.float_format = '{:,.2f}'.format
from pathlib import Path
dirct  = Path('Master.py').resolve().parent
os.chdir(dirct)
import sys
sys.path.append(str(dirct))
from data_functions_albert import remove_outliers, gini, data_stats
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')

else:
    dirct = str(dirct)

my_dirct =  dirct+'/data/panel/'

## IMPORT PANELS: NATIONWIDE AND ONLY RURAL
panel =pd.read_csv(my_dirct+"panel_UGA.csv")





#%% ===========================================================================
# CIW: AVERAGE AND GINI ACROSS WAVES
# =============================================================================


panel = pd.read_csv(my_dirct+"panel_UGA.csv")

panel['wage_total'].replace([0,0.0],np.nan, inplace=True)


panel['wage_total'].describe()



# some extra per capita vars
list_vars =  ['cfood','cnodur', "revenue_agr_p_c_district","bs_profit","profit_lvstk", "wage_total",'asset_value','land_value_hat','wealth_lvstk','farm_capital']
for var in list_vars:
    panel[var+'_cap'] = panel[var]/panel['familysize']


panel_rural = panel.loc[panel['urban'] == 0]
panel_urban = panel.loc[panel['urban'] == 1]


# List of variables
hh_vars = ['ctotal', 'inctotal', 'wtotal']
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']

# Function to compute summary statistics (mean and Gini)
def compute_summary(panel_data, var_list):
    summary = []
    gini_list = []
    for wave in waves:
        # Average
        summary.append(np.mean(panel_data.loc[panel_data['wave'] == wave, var_list], axis=0))
        # Gini
        gini_wave = []
        for var in var_list:
            gini_wave.append(gini(panel_data.loc[panel_data['wave'] == wave, var].replace([np.inf, -np.inf], np.nan).dropna()))
        gini_list.append(gini_wave)
    
    mean_cwi = pd.DataFrame(summary).mean(axis=0).T  # Mean across waves
    sum_ciw = pd.concat([pd.DataFrame(summary), pd.DataFrame([mean_cwi])], ignore_index=True)
    sum_gini = pd.concat([pd.DataFrame(gini_list), pd.DataFrame([pd.DataFrame(gini_list).mean(axis=0)])], axis=0)
    
    sum_ciw.index = waves + ['Average']
    sum_gini.index = waves + ['Average']
    
    return sum_ciw, sum_gini

# Compute for the whole country
sum_ciw, sum_gini = compute_summary(panel,hh_vars)

# Compute for rural households
sum_ciw_rural, sum_gini_rural = compute_summary(panel_rural,hh_vars)

# Compute for urban households
sum_ciw_urban, sum_gini_urban = compute_summary(panel_urban,hh_vars)

# Concatenate the results: National, Rural, Urban
waves_mean_cwi = pd.concat([sum_ciw, sum_ciw_rural, sum_ciw_urban], axis=1, keys=['National', 'Rural', 'Urban'])
waves_sum_gini = pd.concat([sum_gini, sum_gini_rural, sum_gini_urban], axis=1, keys=['National', 'Rural', 'Urban'])

# Format Gini coefficients as (rounded values)
waves_sum_gini = '(' + round(waves_sum_gini, 2).astype(str) + ')'

waves_mean_cwi_rounded = waves_mean_cwi.round(0)
# Output the tables
print('Household Averages:')
print(waves_mean_cwi.to_latex(float_format="%.0f"))

print('Gini Coefficients:')
print(waves_sum_gini.to_latex())

print('======== IN PER CAPITA TERMS ==========')


# List of variables
cap_vars = ['ctotal_cap','inctotal_cap', 'wtotal_cap']
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']


# Compute for the whole country
sum_ciw, sum_gini = compute_summary(panel,cap_vars)

# Compute for rural households
sum_ciw_rural, sum_gini_rural = compute_summary(panel_rural,cap_vars)

# Compute for urban households
sum_ciw_urban, sum_gini_urban = compute_summary(panel_urban,cap_vars)

# Concatenate the results: National, Rural, Urban
waves_mean_cwi = pd.concat([sum_ciw, sum_ciw_rural, sum_ciw_urban], axis=1, keys=['National', 'Rural', 'Urban'])
waves_sum_gini = pd.concat([sum_gini, sum_gini_rural, sum_gini_urban], axis=1, keys=['National', 'Rural', 'Urban'])

# Format Gini coefficients as (rounded values)
waves_sum_gini = '(' + round(waves_sum_gini, 2).astype(str) + ')'

waves_mean_cwi_rounded = waves_mean_cwi.round(0)
# Output the tables
print('Per Capita Averages:')
print(waves_mean_cwi.to_latex(float_format="%.0f"))

print('Per Capita Gini Coefficients:')
print(waves_sum_gini.to_latex())






#%% DECOMPOSING CONSUMPTION, INCOME, WEALTH IN RURAL

print('=========================================')
print(' THE COMPOSITION OF CIW ')



## summary table 
waves = ['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016']

def decomposition_1(data):
    sum_cwi_long = []
    for wave in waves:
        obs_total = min(data.loc[data['wave']==wave, ['ctotal', 'inctotal', 'wtotal']].count())
        mean_cwi_long = (data.loc[data['wave']==wave, ['cfood','cnodur', "revenue_agr_p_c_district","bs_profit","profit_lvstk", "wage_total",'asset_value','land_value_hat','wealth_lvstk','farm_capital']].mean(axis=0)).T
        obs_cwi_long = 100*(data.loc[data['wave']==wave, ['cfood','cnodur', "revenue_agr_p_c_district","bs_profit","profit_lvstk", "wage_total",'asset_value','land_value_hat','wealth_lvstk','farm_capital']].count(axis=0)).T/obs_total
        mean_cwi_long = round(mean_cwi_long.fillna(0), 0).astype(int)
        obs_cwi_long = round(obs_cwi_long, 0).astype(int) 
        sum_cwi_long.append(mean_cwi_long)
        sum_cwi_long.append(obs_cwi_long.astype(str)+'%')
     
    sum_cwi_long_df = (pd.concat(sum_cwi_long, axis=1)).T
    return sum_cwi_long_df

def decomposition_capita(data):
    sum_cwi_long = []
    for wave in waves:
        obs_total = min(data.loc[data['wave']==wave, ['ctotal_cap', 'inctotal_cap', 'wtotal_cap']].count())
        mean_cwi_long = (data.loc[data['wave']==wave, ['cfood_cap','cnodur_cap', "revenue_agr_p_c_district_cap","bs_profit_cap","profit_lvstk_cap", "wage_total_cap",'asset_value_cap','land_value_hat_cap','wealth_lvstk_cap','farm_capital_cap']].mean(axis=0)).T
        obs_cwi_long = 100*(data.loc[data['wave']==wave, ['cfood_cap','cnodur_cap', "revenue_agr_p_c_district_cap","bs_profit_cap","profit_lvstk_cap", "wage_total_cap",'asset_value_cap','land_value_hat_cap','wealth_lvstk_cap','farm_capital_cap']].count(axis=0)).T/obs_total
        mean_cwi_long = round(mean_cwi_long.fillna(0), 0).astype(int)
        obs_cwi_long = round(obs_cwi_long, 0).astype(int) 
        sum_cwi_long.append(mean_cwi_long)
        sum_cwi_long.append(obs_cwi_long.astype(str)+'%')
     
    sum_cwi_long_df = (pd.concat(sum_cwi_long, axis=1)).T
    return sum_cwi_long_df

print(decomposition_1(panel).to_latex())
print(decomposition_1(panel_rural).to_latex())
print(decomposition_1(panel_urban).to_latex())


# agric
(853+776+1010+902+1045)/5


(285+245+349+234+271)/5

# business proportions
(49+47+43+44+46)/5

# labor
(46+42+35*3)/5

(853+776+1010+902+1045)/5

#rural business
(46+44+41+42+41)/5
(619+683+918+929+873)/5

# rural labor
(42+39+30+31+31)/5
(759+775+866+833+1410)/5

#urban business
(60+60+54+58+62)/5
(1265+1574+1604+1893+1518)/5

#urban labor
(58+59+56+52)/5
(1486+1787+1391+1530+1366)/5

(47+46+49+50+51)/5

# land in rural
(3162+2920+3616+2613+2381)/5

# assets in rural
(1251+1341+1438+1575+1357)/5

# assets in urban
(3754+4094+4917+3307+2927)/5

# assets nat
(1806+1881+2086+1944+1689)/5

# land nat
(3104+2925+3648+2550+2326)/5
(68+68+73+75+77)/5

# farm capital nat
(84+81+86+88)/4
(25+21+19+17)/4

#food
fc = 1025+1073+1181+1095+1033
fnc = 714+607+612+606+584

fc/(fc+fnc)


#food
fc = 965+1001+1126+1051+998
fnc = 566+465+500+513+487
fc/(fc+fnc)


#food
fc = 1237+1367+1419+1259+1163
fnc = 1262+1196+949+946


fc/(fc+fnc)








