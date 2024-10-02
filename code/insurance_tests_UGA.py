# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:59:02 2020

@author: rodri
"""


import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

dirct  = Path('Master.py').resolve().parent
if '\\' in str(dirct):
    dirct= str(dirct).replace('\\', '/')
else:
    dirct = str(dirct)

my_wd1 = dirct+'/data/panel/'
folder = dirct+'/results/figures/'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
pd.options.display.float_format = '{:,.2f}'.format
from linearmodels.panel import PanelOLS



import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

plot_title = 0
save=True

#Import Panel
panel_all = pd.read_csv(my_wd1+"panel_UGA.csv")
panel_all.replace(['2009-2010', '2010-2011', '2011-2012', '2013-2014', '2015-2016'],[2009, 2010.0, 2011, 2013, 2015],inplace=True)


panel_0911 = panel_all.loc[(panel_all['wave']==2009) |(panel_all['wave']==2010) |(panel_all['wave']==2011) ]



## Create balanced panel: 
counthh = panel_all.groupby(by="hh")[["hh"]].count()
pd.value_counts(counthh['hh'])
counthh.columns = ["counthh"]
counthh.reset_index(inplace=True)
panel = panel_all.merge(counthh, on="hh", how="left")


#Only those observed across 5 waves 
panelbal = panel_all.loc[panel["counthh"]==5,]
counthh = panelbal.groupby(by="hh")[["hh"]].count()
pd.value_counts(counthh['hh'])
counthh.columns = ["counthh"]
counthh.reset_index(inplace=True)
panelbal = panelbal.merge(counthh, on="hh", how="left")




#%% GENERATE PANEL TO COMPUTE INSURANCE MEASURES

### Choose btw panel, panelbal, panel_0911
panel = panelbal


location = panel.groupby(by=['hh'])[['wave','urban']].mean()
location['rural_allwaves'] = 1*(location['urban']==0)
location['urban_allwaves'] = 1*(location['urban']==1)
location['migrated'] = 1*((location['urban']>0) & (location['urban']<1))
del location['urban'], location['wave']
panel = pd.merge(panel, location, on=['hh'], how='left')

print(panel[['rural_allwaves','urban_allwaves','migrated']].mean())

panelrural = panel.loc[panel['rural_allwaves']==1]
panelurban = panel.loc[panel['urban_allwaves']==1]

print('obs rural:',len(panelrural)/5)
print('obs urban:',len(panelurban)/5)



#panel.reset_index(inplace=True)

panel.dropna(subset=['lnc','lny'], inplace=True)

# Create region-wave averages
regionmeans = panel.groupby(by=["wave","region"])[["lnc"]].mean()
regionmeans.reset_index(inplace=True)
regionmeans.rename(columns={"lnc":"avgc"}, inplace=True)
regionmeans.loc[regionmeans["avgc"]==-np.inf, "avgc"] = np.nan
regionmeans["avgc"].fillna(np.mean(regionmeans["avgc"]), inplace=True)
panel = panel.merge(regionmeans, on=["wave","region"], how="outer")

regionmeansrur = panelrural.groupby(by=["wave","region"])[["lnc"]].mean()
regionmeansrur.reset_index(inplace=True)
regionmeansrur.rename(columns={"lnc":"avgc"}, inplace=True)
regionmeansrur.loc[regionmeansrur["avgc"]==-np.inf, "avgc"] = np.nan
regionmeansrur["avgc"].fillna(np.mean(regionmeansrur["avgc"]), inplace=True)
panelrural = panelrural.merge(regionmeansrur, on=["wave","region"], how="outer")

regionmeansurb = panelurban.groupby(by=["wave","region"])[["lnc"]].mean()
regionmeansurb.reset_index(inplace=True)
regionmeansurb.rename(columns={"lnc":"avgc"}, inplace=True)
regionmeansurb.loc[regionmeansurb["avgc"]==-np.inf, "avgc"] = np.nan
regionmeansurb["avgc"].fillna(np.mean(regionmeansurb["avgc"]), inplace=True)
panelurban = panelurban.merge(regionmeansurb, on=["wave","region"], how="outer")



control0 = panel[['hh','counthh','wave','urban','age','age_sq','familysize','writeread','region2','region3','region4','female','classeduc']]
control0rur = panelrural[['hh','counthh','wave','urban','age','age_sq','familysize','writeread','region2','region3','region4','female','classeduc']]
control0urb = panelurban[['hh','counthh','wave','urban','age','age_sq','familysize','writeread','region2','region3','region4','female','classeduc']]

ciw_avg = panel.groupby(by="hh")[["ctotal","inctotal",'wtotal']].mean()
ciw_avg.reset_index(inplace=True)

ciw_avgrur = panelrural.groupby(by="hh")[["ctotal","inctotal",'wtotal']].mean()
ciw_avgrur.reset_index(inplace=True)

ciw_avgurb = panelurban.groupby(by="hh")[["ctotal","inctotal",'wtotal']].mean()
ciw_avgurb.reset_index(inplace=True)


control = pd.merge(control0,ciw_avg, how='right', on='hh')

controlrur = pd.merge(control0rur,ciw_avgrur, how='right', on='hh')
controlurb = pd.merge(control0urb,ciw_avgurb, how='right', on='hh')

for data in [control, controlrur, controlurb]:
    data['c_quin'] = pd.qcut(data["ctotal"], 5, labels=False)
    data['inc_quin'] = pd.qcut(data["inctotal"], 5, labels=False)
    data['w_quin'] = pd.qcut(data["wtotal"], 5, labels=False)


    
    #Create HH characteristics controls
    cdummies = pd.get_dummies(data["c_quin"])
    cdummies.drop([0.0], axis=1, inplace=True)
    cdummies.columns = ["c2","c3","c4","c5"]
    idummies = pd.get_dummies(data["inc_quin"])
    idummies.drop([0.0], axis=1, inplace=True)
    idummies.columns = ["inc2","inc3","inc4","inc5"]
    wdummies = pd.get_dummies(data["w_quin"])
    wdummies.drop([0.0], axis=1, inplace=True)
    wdummies.columns = ["w2","w3","w4","w5"]
    
    
    data = data.join(cdummies)
    data = data.join(idummies)
    data = data.join(wdummies)
    
    data[['c_quin','inc_quin','w_quin']].corr()
    
    print(np.min(data['counthh']))
    for var, var_name in zip(['c', 'w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        for quintile in [0,1,2,3,4]:
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
        
        



panel.set_index(["hh","wave"],inplace=True)
paneldiff = panel.groupby(by='hh')[ 'avgc', 'lnc', 'lncfood', 'lny'].diff()
paneldiff.columns = ['Δavgc','Δlnc','Δlnfood','Δlny']
paneldiff.reset_index(inplace=True)

paneldiff = paneldiff.merge(control, on=["hh",'wave'], how='left')



#rural
panelrural.set_index(["hh","wave"],inplace=True)
paneldiffrur = panelrural.groupby(by='hh')[ 'avgc', 'lnc', 'lncfood','lny'].diff()
paneldiffrur.columns = ['Δavgc','Δlnc','Δlnfood','Δlny']
paneldiffrur.reset_index(inplace=True)

paneldiffrur = paneldiffrur.merge(controlrur, on=["hh",'wave'], how='left')


#urban

panelurban.set_index(["hh","wave"],inplace=True)
paneldiffurb = panelurban.groupby(by='hh')[ 'avgc', 'lnc', 'lncfood', 'lny'].diff()
paneldiffurb.columns = ['Δavgc','Δlnc','Δlnfood','Δlny']
paneldiffurb.reset_index(inplace=True)

paneldiffurb = paneldiffurb.merge(controlurb, on=["hh",'wave'], how='left')



counthh_xwave = []

for data, loc in zip([paneldiff, paneldiffrur, paneldiffurb],['Nationwide','Rural','Urban']):
    counthh = pd.value_counts(data['hh'])
    print(loc+' count hh ='+str(max(counthh)))
    counthh_xwave.append(counthh)





#%% Compute Townsend coefficient per each quintile and do plot


### NOTES:
#   - Balanced panel: in rural (CW) poor areas insure better, in I relation is flat. In urban (IW) top rich insure better. we cannot reject full-insurance for them. in C relationship is flat.
#   - Panel (09-11:): C poor insure better, W poor insure worse, I no relation... (fucked). Rural no clear relationship (CW). Urban top-rich (CW) insure better.     
#   - Unbalanced panel: In rural C poor insure better, W goes into same direction but not that robust. I no relation. Urban CIW rich insure better.


list_ols = []
list_panelols = []


print('===================================')
print('INSURANCE TEST CRRA SPECIFICATION')
print('===================================')

for data, loc in zip([paneldiff, paneldiffrur, paneldiffurb],['Nationwide','Rural','Urban']):
    print('Insurance test for '+loc+' Uganda --------------------------')
    
    data = data.dropna(subset=['Δlnc','Δlny'])
    data.set_index(['hh','wave'], inplace=True)
    print('fixed effects---------')
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
            
            
        
            #lin_reg = sm.ols(formula='Δlnc ~ +Δlny + Δavgc  +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            lin_reg = PanelOLS.from_formula('Δlnc ~ Δavgc +Δlny     +EntityEffects  ', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='clustered', cluster_entity=True)
            
            list_panelols.append(lin_reg)
            
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            
            params.append(lin_reg.params[1])
            #error = lin_reg.params[2] - lin_reg.conf_int()[0][1]
            error = lin_reg.params[1] -lin_reg.conf_int().iloc[1,0]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
        
        N_q = np.min(obs)
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance levels across '+var_name+' Quintiles '+loc+' in Uganda (+controls)')
      
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)    
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q))
        if save==True:
            fig.savefig(folder+'Insurance_CRRA_FE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
        
       
       
    
    print('No fixed effects--------')
    for var, var_name in zip(['c', 'w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
        
            lin_reg = sm.ols(formula='Δlnc ~ Δavgc +Δlny', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            
            list_ols.append(lin_reg)
            params.append(lin_reg.params[2])
            error = lin_reg.params[2] - lin_reg.conf_int()[0][2]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[2],4))
        
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        N_q = np.min(obs)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance levels across '+var_name+' Quintiles '+loc+' in Uganda')
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)   
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q))
        if save==True:
            fig.savefig(folder+'Insurance_CRRA_NoFE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
        



#%% Insurance tests on food consumption: Quintile 

### NOTES:
#   - Balanced panel: in rural (CW) poor areas insure better, in I Q1 insures better, rest flat. In urban (IW) top rich insure better. we cannot reject full-insurance for them. in C relationship is flat.
#   - Panel (09-11:): C poor insure better, W poor insure worse, I poor not robust. (fucked). Rural no clear relationship C, WI rich insure better, . Urban top-rich (CW) insure better though not linear, bottom poor urb cannot reject full-insurance         
#   - Unbalanced panel: In rural C poor insure better, WI rather flat. Urban I rich insure better. CW rather flat relationship

print('===================================')
print('FOOD INSURANCE TEST CRRA SPECIFICATION')
print('===================================')


for data, loc in zip([paneldiff, paneldiffrur, paneldiffurb],['Nationwide','Rural','Urban']):
    print('Insurance test for '+loc+' Uganda --------------------------')
    
    data = data.dropna(subset=['Δlnc','Δlny'])
    data.set_index(['hh','wave'], inplace=True)
    
    
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
            
            #lin_reg = sm.ols(formula='Δlnc ~ +Δlny + Δavgc  +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            lin_reg = PanelOLS.from_formula('Δlnfood ~ Δavgc +Δlny   +EntityEffects  ', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='clustered', cluster_entity=True)
            
            list_panelols.append(lin_reg)
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            params.append(lin_reg.params[1])
            #error = lin_reg.params[2] - lin_reg.conf_int()[0][1]
            error = lin_reg.params[1] -lin_reg.conf_int().iloc[1,0]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
        
        
        
        
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        N_q = np.min(obs)
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance levels across '+var_name+' Quintiles in '+loc+' Uganda (residuals)')
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)   
        # Set the labels (with LaTeX formatting for math notation)
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q))
        if save==True:
            fig.savefig(folder+'food_CRRA_FE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
        
    

   


#%% Compute Townsend coefficient (exponential utility specification) per each quintile and do plot


print('===================================')
print('INSURANCE TEST STANDARD SPECIFICATION')
print('===================================')

panelcon = panel.merge(control,on=['hh','wave'])
panelconrur = panelrural.merge(controlrur,on=['hh','wave'])
panelconurb = panelurban.merge(controlurb,on=['hh','wave'])



for data, loc in zip([panelcon, panelconrur, panelconurb],['Nationwide','Rural','Urban']):
    print('Insurance test for '+loc+' Uganda --------------------------')
    data = data.dropna(subset=['lnc','lny'])
    data.set_index(['hh','wave'], inplace=True)
    
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
            
            
            #lin_reg = sm.ols(formula='Δlnc ~ +Δlny + Δavgc  +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            lin_reg = PanelOLS.from_formula('lnc ~ avgc +lny   +EntityEffects  ', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='clustered', cluster_entity=True)
            
            list_panelols.append(lin_reg)
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
            params.append(lin_reg.params[1])
            #error = lin_reg.params[2] - lin_reg.conf_int()[0][1]
            error = lin_reg.params[1] -lin_reg.conf_int().iloc[1,0]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            
            
        
        
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        N_q = np.min(obs)
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance across '+var_name+' Quintiles in '+loc+' Uganda. Controls, Exponencial Utility')
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)   
        # Set the labels (with LaTeX formatting for math notation)
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q))  
        
        ax.set_ylim(min(ax.get_ylim()[0],-0.001), ax.get_ylim()[1])
        if save==True:
            fig.savefig(folder+'insurance_EXP_FE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
        
        
    print('===== No fixed effects  =====')
    
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
        
            lin_reg = sm.ols(formula='lnc ~ avgc +lny', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            params.append(lin_reg.params[2])
            error = lin_reg.params[2] - lin_reg.conf_int()[0][2]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
        
        
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        N_q = np.min(obs)
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance levels across '+var_name+' Quintiles in '+loc+' Uganda. Exponencial Utility')
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)   
        # Set the labels (with LaTeX formatting for math notation)
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q)) 
        ax.set_ylim(min(ax.get_ylim()[0],-0.001), ax.get_ylim()[1])
        if save==True:
            fig.savefig(folder+'insurance_EXP_NoFE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
            
            
    print('===== food  =====')    
         ### EXPONENTIAL UTILITY
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
        
            lin_reg = PanelOLS.from_formula('lncfood ~ avgc +lny   +EntityEffects  ', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='clustered', cluster_entity=True)
            params.append(lin_reg.params[1])
            error = lin_reg.params[1] -lin_reg.conf_int().iloc[1,0]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
        
        
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        N_q = np.min(obs)
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance levels across '+var_name+' Quintiles in '+loc+' Uganda')
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)   
        # Set the labels (with LaTeX formatting for math notation)
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q)) 
        ax.set_ylim(min(ax.get_ylim()[0],-0.001), ax.get_ylim()[1])
        if save==True:
            fig.savefig(folder+'food_EXP_FE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
   








#%% BALANCED PANEL

print('=====================')
print(' UNBALANCED PANEL')
print('=====================')

### Choose btw panel, panelbal, panel_0911
panel = panel_all


location = panel.groupby(by=['hh'])[['wave','urban']].mean()
location['rural_allwaves'] = 1*(location['urban']==0)
location['urban_allwaves'] = 1*(location['urban']==1)
location['migrated'] = 1*((location['urban']>0) & (location['urban']<1))
del location['urban'], location['wave']
panel = pd.merge(panel, location, on=['hh'], how='left')

print(panel[['rural_allwaves','urban_allwaves','migrated']].mean())

panelrural = panel.loc[panel['rural_allwaves']==1]
panelurban = panel.loc[panel['urban_allwaves']==1]

panelrural.reset_index(inplace=True)

a=panel.groupby(by="hh")[["hh"]].count() 
b=panelrural.groupby(by="hh")[["hh"]].count() 
c=panelurban.groupby(by="hh")[["hh"]].count() 

print('obs all:',len(a),'obs all more than 1 wave',len(a.loc[a['hh']>1]))
print('obs rural:',len(b),'obs rural more than 1 wave',len(b.loc[b['hh']>1]))
print('obs urban:',len(c),'obs urban more than 1 wave',len(c.loc[c['hh']>1]))


olsc = sm.ols(formula="lnc ~  +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc ", data=panel).fit()
print(olsc.summary())

olsfood = sm.ols(formula="lncfood ~ +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc ", data=panel).fit()
print(olsfood.summary())

olsi = sm.ols(formula="lny ~ +age +age_sq  +familysize +region2 +region3 +region4  +female  +classeduc ", data=panel).fit()
print(olsi.summary())

panel["u_c"] = olsc.resid
panel["u_y"] = olsi.resid
panel["u_food"] = olsfood.resid

#panel.reset_index(inplace=True)

panel.dropna(subset=['lnc','lny'], inplace=True)

# Create region-wave averages
regionmeans = panel.groupby(by=["wave","region"])[["lnc"]].mean()
regionmeans.reset_index(inplace=True)
regionmeans.rename(columns={"lnc":"avgc"}, inplace=True)
regionmeans.loc[regionmeans["avgc"]==-np.inf, "avgc"] = np.nan
regionmeans["avgc"].fillna(np.mean(regionmeans["avgc"]), inplace=True)
panel = panel.merge(regionmeans, on=["wave","region"], how="outer")

regionmeansrur = panelrural.groupby(by=["wave","region"])[["lnc"]].mean()
regionmeansrur.reset_index(inplace=True)
regionmeansrur.rename(columns={"lnc":"avgc"}, inplace=True)
regionmeansrur.loc[regionmeansrur["avgc"]==-np.inf, "avgc"] = np.nan
regionmeansrur["avgc"].fillna(np.mean(regionmeansrur["avgc"]), inplace=True)
panelrural = panelrural.merge(regionmeansrur, on=["wave","region"], how="outer")

regionmeansurb = panelurban.groupby(by=["wave","region"])[["lnc"]].mean()
regionmeansurb.reset_index(inplace=True)
regionmeansurb.rename(columns={"lnc":"avgc"}, inplace=True)
regionmeansurb.loc[regionmeansurb["avgc"]==-np.inf, "avgc"] = np.nan
regionmeansurb["avgc"].fillna(np.mean(regionmeansurb["avgc"]), inplace=True)
panelurban = panelurban.merge(regionmeansurb, on=["wave","region"], how="outer")



control0 = panel[['hh','wave','urban','age','age_sq','familysize','writeread','region2','region3','region4','female','classeduc']]
control0rur = panelrural[['hh','wave','urban','age','age_sq','familysize','writeread','region2','region3','region4','female','classeduc']]
control0urb = panelurban[['hh','wave','urban','age','age_sq','familysize','writeread','region2','region3','region4','female','classeduc']]

ciw_avg = panel.groupby(by="hh")[["ctotal","inctotal",'wtotal']].mean()
ciw_avg.reset_index(inplace=True)

ciw_avgrur = panelrural.groupby(by="hh")[["ctotal","inctotal",'wtotal']].mean()
ciw_avgrur.reset_index(inplace=True)

ciw_avgurb = panelurban.groupby(by="hh")[["ctotal","inctotal",'wtotal']].mean()
ciw_avgurb.reset_index(inplace=True)


control = pd.merge(control0,ciw_avg, how='right', on='hh')

controlrur = pd.merge(control0rur,ciw_avgrur, how='right', on='hh')
controlurb = pd.merge(control0urb,ciw_avgurb, how='right', on='hh')

for data in [control, controlrur, controlurb]:
    data['c_quin'] = pd.qcut(data["ctotal"], 5, labels=False)
    data['inc_quin'] = pd.qcut(data["inctotal"], 5, labels=False)
    data['w_quin'] = pd.qcut(data["wtotal"], 5, labels=False)


    
    #Create HH characteristics controls
    cdummies = pd.get_dummies(data["c_quin"])
    cdummies.drop([0.0], axis=1, inplace=True)
    cdummies.columns = ["c2","c3","c4","c5"]
    idummies = pd.get_dummies(data["inc_quin"])
    idummies.drop([0.0], axis=1, inplace=True)
    idummies.columns = ["inc2","inc3","inc4","inc5"]
    wdummies = pd.get_dummies(data["w_quin"])
    wdummies.drop([0.0], axis=1, inplace=True)
    wdummies.columns = ["w2","w3","w4","w5"]
    
    
    data = data.join(cdummies)
    data = data.join(idummies)
    data = data.join(wdummies)
    
    data[['c_quin','inc_quin','w_quin']].corr()
    
  
    for var, var_name in zip(['c', 'w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        for quintile in [0,1,2,3,4]:
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
        
        



panel.set_index(["hh","wave"],inplace=True)
paneldiff = panel.groupby(by='hh')[ 'avgc', 'lnc', 'lncfood', 'lny'].diff()
paneldiff.columns = ['Δavgc','Δlnc','Δlnfood','Δlny']
paneldiff.reset_index(inplace=True)

paneldiff = paneldiff.merge(control, on=["hh",'wave'], how='left')



#rural
panelrural.set_index(["hh","wave"],inplace=True)
paneldiffrur = panelrural.groupby(by='hh')[ 'avgc', 'lnc', 'lncfood','lny'].diff()
paneldiffrur.columns = ['Δavgc','Δlnc','Δlnfood','Δlny']
paneldiffrur.reset_index(inplace=True)

paneldiffrur = paneldiffrur.merge(controlrur, on=["hh",'wave'], how='left')


#urban

panelurban.set_index(["hh","wave"],inplace=True)
paneldiffurb = panelurban.groupby(by='hh')[ 'avgc', 'lnc', 'lncfood', 'lny'].diff()
paneldiffurb.columns = ['Δavgc','Δlnc','Δlnfood','Δlny']
paneldiffurb.reset_index(inplace=True)

paneldiffurb = paneldiffurb.merge(controlurb, on=["hh",'wave'], how='left')




#%%

list_ols = []
list_panelols = []


print('===================================')
print('INSURANCE TEST CRRA (UNBALANCED PANEL)')
print('===================================')

for data, loc in zip([paneldiff, paneldiffrur, paneldiffurb],['Nationwide','Rural','Urban']):
    print('Insurance test for '+loc+' Uganda --------------------------')
    
    data = data.dropna(subset=['Δlnc','Δlny'])
    data.set_index(['hh','wave'], inplace=True)
    
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
            
            
        
            #lin_reg = sm.ols(formula='Δlnc ~ +Δlny + Δavgc  +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            lin_reg = PanelOLS.from_formula('Δlnc ~ Δavgc +Δlny     +EntityEffects  ', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='clustered', cluster_entity=True)
            
            list_panelols.append(lin_reg)
            
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            
            params.append(lin_reg.params[1])
            #error = lin_reg.params[2] - lin_reg.conf_int()[0][1]
            error = lin_reg.params[1] -lin_reg.conf_int().iloc[1,0]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
        
        N_q = np.min(obs)
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance levels across '+var_name+' Quintiles '+loc+' in Uganda (+controls)')
      
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)    
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q))
        if save==True:
            fig.savefig(folder+'unbal_Insurance_CRRA_FE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
        











print('===================================')
print('INSURANCE TEST STANDARD SPECIFICATION (UNBALANCED PANEL)')
print('===================================')

panelcon = panel.merge(control,on=['hh','wave'])
panelconrur = panelrural.merge(controlrur,on=['hh','wave'])
panelconurb = panelurban.merge(controlurb,on=['hh','wave'])

for data, loc in zip([panelcon, panelconrur, panelconurb],['Nationwide','Rural','Urban']):
    print('Insurance test for '+loc+' Uganda --------------------------')
    data = data.dropna(subset=['lnc','lny'])
    data.set_index(['hh','wave'], inplace=True)
    
    for var, var_name in zip(['c','w','inc'],['Consumption','Wealth','Income']):
        print(var_name+' Distribution')
        params = []
        varname = []
        err_series = []
        obs = []
        for quintile in [0,1,2,3,4]:
            
            
            #lin_reg = sm.ols(formula='Δlnc ~ +Δlny + Δavgc  +age +age_sq  +familysize +region2 +region3 +region4  +female   +classeduc', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='HC1')
            lin_reg = PanelOLS.from_formula('lnc ~ avgc +lny   +EntityEffects  ', data=data.loc[data[var+'_quin']==quintile,]).fit(cov_type='clustered', cluster_entity=True)
            
            obs.append(len(data.loc[data[var+'_quin']==quintile,]))
            print('obs in Q',quintile,'=',len(data.loc[data[var+'_quin']==quintile,]))
            print('Coefficient estimate in Q',quintile,'=',lin_reg.params[1])
            print('Coefficient estimate in Q',quintile,'=',round(lin_reg.params[1],4))
            params.append(lin_reg.params[1])
            #error = lin_reg.params[2] - lin_reg.conf_int()[0][1]
            error = lin_reg.params[1] -lin_reg.conf_int().iloc[1,0]
            err_series.append(error)
            varname.append('Q'+str(quintile+1))
            
            
        
        
        coef_df = pd.DataFrame({'coef': params,
                                'err': err_series,
                                'quint': varname,
                               })
        coef_df
        N_q = np.min(obs)
        fig, ax = plt.subplots(figsize=(8, 5))
        coef_df.plot(x='quint', y='coef', kind='bar', 
                     ax=ax, color='none', 
                     yerr='err', legend=False)
        if plot_title == 1:
            ax.set_title('Insurance across '+var_name+' Quintiles in '+loc+' Uganda. Controls, Exponencial Utility')
        ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
                   marker='s', s=120, 
                   y=coef_df['coef'], color='black')
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
        ax.xaxis.set_ticks_position('none')
        ax
        _ = ax.set_xticklabels(varname, 
                               rotation=0, fontsize=16)   
        # Set the labels (with LaTeX formatting for math notation)
        ax.set_ylabel(r'$\hat{\beta}$')
        ax.set_xlabel(var_name+r' Distribution              $N_Q=$'+str(N_q))  
        
        ax.set_ylim(min(ax.get_ylim()[0],-0.001), ax.get_ylim()[1])
        if save==True:
            fig.savefig(folder+'unbal_insurance_EXP_FE_'+str(loc)+'_along_'+str(var)+'.png', format='png', bbox_inches='tight')
        
























