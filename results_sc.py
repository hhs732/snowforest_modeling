###       /bin/bash runTestCases_docker.sh
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,netcdftime,num2date
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.metrics import mean_squared_error
import itertools
import csv
#%% SWE data
with open("input_SWE.csv") as scvd:
    reader = csv.reader(scvd)
    raw_swe = [r for r in reader]
sc_swe_column = []
for csv_counter1 in range (len (raw_swe)):
    for csv_counter2 in range (2):
        sc_swe_column.append(raw_swe[csv_counter1][csv_counter2])
sc_swe=np.reshape(sc_swe_column,(len (raw_swe),2))
sc_swe = sc_swe[1:]
sc_swe_obs_date = pd.DatetimeIndex(sc_swe[:,0])
sc_swe_obs = [float(value) for value in sc_swe[:,1]]
swe_obs_df = pd.DataFrame(sc_swe_obs, columns = ['observed swe']) 
swe_obs_df.set_index(sc_swe_obs_date,inplace=True)

#max_swe_obs = max(obs_swe['swe_mm'])
#max_swe_date_obs = obs_swe[obs_swe ['swe_mm']== max_swe_obs].index.tolist()    
#%%
hruidxID = list(np.arange(101,105))
hru_num = np.size(hruidxID)
out_names = ['cs']#,'us','uj','cj','cs2','cs3']
paramModel = (np.size(out_names))*(hru_num)
hru_names =[]
for i in out_names:
    hru_names.append(['{}{}'.format(i, j) for j in hruidxID])
hru_names1 = np.reshape(hru_names,(paramModel,1))
hru_names_df = pd.DataFrame (hru_names1)
#%% reading output_swe files
av_ncfiles = ["sagehen1_2015-2016_senatorVariableDecayRate_1.nc", 
              ]
av_all = []
for ncfiles in av_ncfiles:
    av_all.append(Dataset(ncfiles))

for varname in av_all[0].variables.keys():
    var = av_all[0].variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)

#av_sd = []
#for dfs in av_all:
#    av_sd.append(pd.DataFrame(dfs['scalarSnowDepth'][:]))
#av_sd_df = pd.concat (av_sd, axis=1)
#av_sd_df.columns =  hru_names_df[0]

av_swe = []
for dfs in av_all:
    av_swe.append(pd.DataFrame(dfs['scalarSWE'][:]))
av_swe_df = pd.concat (av_swe, axis=1)
av_swe_df.columns = hru_names_df[0]

#%% output time step
TimeSc = av_all[0].variables['time'][:] # get values
t_unitSc = av_all[0].variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = av_all[0].variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSc = num2date(TimeSc, units=t_unitSc, calendar=t_cal)
DateSc = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSc] # -%d %H:%M to display dates as string #i.strftime("%Y-%m-%d %H:%M")        
#%% day of snow disappearance-final output
av_swe_df.set_index(pd.DatetimeIndex(DateSc),inplace=True)
counter = pd.DataFrame(np.arange(0,np.size(av_swe_df['cs101'])),columns=['counter'])
counter.set_index(av_swe_df.index,inplace=True)
av_swe_df2 = pd.concat([counter, av_swe_df], axis=1)
#%%   
av_swe_df4000 = av_swe_df2[:][4000:8737]

zerosnowdate = []
for val in hru_names_df[0]:
    zerosnowdate.append(np.where(av_swe_df4000[val]==0))
zerosnowdate_omg = [item[0] for item in zerosnowdate] #change tuple to array
for i,item in enumerate(zerosnowdate_omg):
    if len(item) == 0:
        zerosnowdate_omg[i] = 3737
for i,item in enumerate(zerosnowdate_omg):
    zerosnowdate_omg[i] = zerosnowdate_omg[i]+4000
        
first_zerosnowdate =[]
for i,item in enumerate(zerosnowdate_omg):
    if np.size(item)>1:
        #print np.size(item)
        first_zerosnowdate.append(item[0])
    if np.size(item)==1:
        first_zerosnowdate.append(item)
    
first_zerosnowdate_df = pd.DataFrame(np.array(first_zerosnowdate)).T
first_zerosnowdate_df.columns = hru_names_df
#first_zerosnowdate_df_obs = pd.DataFrame(np.array([[5985],[6200]]).T,columns=out_names)
first_zerosnowdate_df_obs = pd.DataFrame(np.array([4692]),columns=['2015'])

zerosnowdate_residual=[]
for hru in first_zerosnowdate_df.columns:
    zerosnowdate_residual.append((first_zerosnowdate_df[hru][0]-first_zerosnowdate_df_obs['2015'])/24)

zerosnowdate_residual_df = pd.DataFrame(np.reshape(np.array(zerosnowdate_residual),(np.size(out_names),hru_num)).T, columns=out_names)

#%%

for namefile in out_names:
    x = list(np.arange(1,5))
    fig = plt.figure(figsize=(20,15))
    plt.bar(x,zerosnowdate_residual_df[namefile])
    plt.title(namefile, fontsize=42)
    plt.xlabel('hrus',fontsize=30)
    plt.ylabel('residual dosd (day)', fontsize=30)
    #vax.yaxis.set_label_coords(0.5, -0.1) 
    plt.savefig(namefile)

#%%
DateSc2 = [i.strftime("%Y-%m-%d") for i in tvalueSc]
sax = np.arange(0,np.size(DateSc2))
sa_xticks = DateSc2
safig, saax = plt.subplots(1,1, figsize=(20,15))
plt.xticks(sax, sa_xticks[::1000], rotation=25, fontsize=20)
saax.xaxis.set_major_locator(ticker.AutoLocator())
plt.yticks(fontsize=20)
for hru in av_swe_df.columns:
    plt.plot(av_swe_df[hru])#, sbx, swe_obs2006, 'k--', linewidth=0.5)#, label='wwe', color='maroon') param_nam_list[q] color_list[q]

plt.plot(swe_obs_df, 'k', markersize=10)

plt.title('rainbow_SWE', position=(0.04, 0.88), ha='left', fontsize=40)
plt.xlabel('Time 2015-2016', fontsize=30)
plt.ylabel('SWE(mm)', fontsize=30)
#plt.show()
plt.savefig('swecs.png')
#%%














