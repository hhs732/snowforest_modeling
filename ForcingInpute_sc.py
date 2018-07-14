#%matplotlib inline    /bin/bash runTestCases_docker.sh
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset,netcdftime,num2date
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import csv
#%% hru names
hruidxID = list(np.arange(101,166))
hru_num = np.size(hruidxID)
#%% #Sagehen creek basin forcing data (tower 1)
scFD = Dataset('shT1_force1_2.nc')
for varname in scFD.variables.keys():
    var = scFD.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)
for dimname in scFD.dimensions.keys():
    dim = scFD.dimensions[dimname]
    print(dimname, len(dim), dim.isunlimited())
#%%
print scFD.variables['LWRadAtm'][7:19]
print scFD.variables['SWRadAtm'][7:19]
print scFD.variables['SWRadAtm'].units

#%% sagehen timming
TimeSc = scFD.variables['time'][:] # get values
t_unitS = scFD.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"

try :

    t_cal = scFD.variables['time'].calendar

except AttributeError : # Attribute doesn't exist

    t_cal = u"gregorian" # or standard

tvalueSc = num2date(TimeSc, units=t_unitS, calendar=t_cal)
scDate = [i.strftime("%Y-%m-%d %H:%M") for i in tvalueSc]
#%% sagehen creek forcing data columns=['pptrate','SWRadAtm','LWRadAtm','airtemp','windspd','airpres','spechum']
#Temp and ppt average to test data
temp_sc = scFD.variables['airtemp'][:]
temp_data = pd.DataFrame(temp_sc,index=pd.DatetimeIndex(scDate))
#temp_data=pd.Series(pd.DataFrame(temp_sc),index=pd.DatetimeIndex(scDate))
temp_meanyr=temp_data.resample("A").mean()

ppt_sc = scFD.variables['pptrate'][:]
ppt_data = pd.DataFrame(ppt_sc,index=pd.DatetimeIndex(scDate))
ppt_meanyr=ppt_data.resample("A").sum()
#%% make new nc file
new_fc_sc = Dataset("sagehenCreek_forcing.nc",'w',format='NETCDF3_CLASSIC')
# define dimensions 
hru = new_fc_sc.createDimension('hru', hru_num)
time = new_fc_sc.createDimension('time', None)
# define variables
hruid = new_fc_sc.createVariable('hruId', np.int32,('hru',))
lat = new_fc_sc.createVariable('latitude', np.float64,('hru',))
lon = new_fc_sc.createVariable('longitude', np.float64,('hru',))
ds = new_fc_sc.createVariable('data_step', np.float64)
times = new_fc_sc.createVariable('time', np.float64,('time',))
lwrad = new_fc_sc.createVariable('LWRadAtm', np.float64,('time','hru'), fill_value = -999.0)
swrad = new_fc_sc.createVariable('SWRadAtm', np.float64,('time','hru'), fill_value = -999.0)
airpres = new_fc_sc.createVariable('airpres', np.float64,('time','hru'), fill_value = -999.0)
airtemp = new_fc_sc.createVariable('airtemp', np.float64,('time','hru'), fill_value = -999.0)
pptrate = new_fc_sc.createVariable('pptrate', np.float64,('time','hru'), fill_value = -999.0)
spechum = new_fc_sc.createVariable('spechum', np.float64,('time','hru'), fill_value = -999.0)
windspd = new_fc_sc.createVariable('windspd', np.float64,('time','hru'), fill_value = -999.0)
# give variables units
times.units = 'days since 1990-01-01 00:00:00'
ds.units = 'seconds'
lwrad.units = 'W m-2'
swrad.units = 'W m-2'
airpres.units = 'Pa'
airtemp.units = 'K'
pptrate.units = 'kg m-2 s-1'
spechum.units = 'g g-1'
windspd.units = 'm s-1'
# give variables value type
lwrad.vtype = 'scalarv'
swrad.vtype = 'scalarv'
airpres.vtype = 'scalarv'
airtemp.vtype = 'scalarv'
pptrate.vtype = 'scalarv'
spechum.vtype = 'scalarv'
windspd.vtype = 'scalarv'
# read out to compare with original
#for varname in new_fc_sa.variables.keys():
#    var = new_fc_sa.variables[varname]
#    print (varname, var.dtype, var.dimensions, var.shape)
#%% define hru id, time step (1hr), lat and lon
step = np.array([3600])

lat_sa = np.array(scFD.variables['latitude'][:])
len_lat = np.repeat(lat_sa[:,np.newaxis], hru_num, axis=1); len_lat=len_lat.reshape(hru_num,)

long_sa = np.array(scFD.variables['longitude'][:])
len_lon= np.repeat(long_sa[:,np.newaxis], hru_num, axis=1); len_lon=len_lon.reshape(hru_num,)

#%% assign newly created variables with lists of values from NLDAS and Sagehen data
hruid[:] = hruidxID 
lat[:] = len_lat
lon[:] = len_lon
ds[:] = step

new_ix = np.array(scFD.variables['time'][:])
times[:] = new_ix

lwr_sa = np.array(scFD.variables['LWRadAtm'][:])
lwr_sa_hru = np.repeat(lwr_sa[:,np.newaxis], hru_num, axis=1)
lwrad[:] = lwr_sa_hru

swr_sa = np.array(scFD.variables['SWRadAtm'][:])
swr_sa_hru = np.repeat(swr_sa[:,np.newaxis], hru_num, axis=1)
swrad[:] = swr_sa_hru

ap_sa = np.array(scFD.variables['airpres'][:])
ap_sa_hru = np.repeat(ap_sa[:,np.newaxis], hru_num, axis=1)
airpres[:] = ap_sa_hru

at_sa = np.array(scFD.variables['airtemp'][:])
at_sa_hru = np.repeat(at_sa[:,np.newaxis], hru_num, axis=1) 
airtemp[:] = at_sa_hru

ws_sa = np.array(scFD.variables['windspd'][:])
ws_sa_hru = np.repeat(ws_sa[:,np.newaxis], hru_num, axis=1) 
windspd[:] = ws_sa_hru

sh_sa = np.array(scFD.variables['spechum'][:])
sh_sa_hru = np.repeat(sh_sa[:,np.newaxis], hru_num, axis=1) 
spechum[:] = sh_sa_hru

ppt_sa = np.array(scFD.variables['pptrate'][:])
ppt_sa_hru = np.repeat(ppt_sa[:,np.newaxis], hru_num, axis=1) 
pptrate[:] = ppt_sa_hru
#%%precipitation calibration
#ppt_sa0 = np.array(sa_df['pptrate'])
#ppt_sa1 = []
#for cpi in range (np.size(ppt_sa0)):
#    if at_sa [cpi] <= 274 :
#        ppt_sa1.append(ppt_sa0[cpi]+ppt_sa0[cpi]*(1-((np.exp(4.61-0.04*((ws_sa[cpi])**1.75)))/100)))
#    else: ppt_sa1.append(ppt_sa0[cpi])
#ppt_sa = np.array(ppt_sa1)
#ppt_sa_hru = np.repeat(ppt_sa[:,np.newaxis], hru_num, axis=1) 
#pptrate[:] = ppt_sa_hru

#ppt_sa0 = np.array(sa_df['pptrate'])
#ppt_sa1 = []
#for cpi in range (np.size(ppt_sa0)):
#    if at_sa [cpi] <= 274 :
#        ppt_sa1.append(ppt_sa0[cpi]+ppt_sa0[cpi]*(1-((np.exp(4.61-0.16*(ws_sa[cpi]**1.28)))/100)))
#    else: ppt_sa1.append(ppt_sa0[cpi])
#ppt_sa = np.array(ppt_sa1)
#ppt_sa_hru = np.repeat(ppt_sa[:,np.newaxis], hru_num, axis=1) 
#pptrate[:] = ppt_sa_hru

#%%specific humididty calculations***********************************************
#at0_sb = sbFD.variables['airtemp'][:]
#
#e_t = (ap_sb * sh_sb)/0.622
#p_da = ap_sb - e_t
#e_star_t = 611*(np.exp((17.27*(at0_sb-273.15))/(at0_sb-273.15+237.3)))
#rh = e_t/e_star_t
#
#e_star_t2 = 611*(np.exp((17.27*(at0_sb+2-273.15))/(at0_sb+2-273.15+237.3)))
#e_star_t4 = 611*(np.exp((17.27*(at0_sb+4-273.15))/(at0_sb+4-273.15+237.3)))
#
#e_t2 = rh * e_star_t2
#e_t4 = rh * e_star_t4
#
#p_t2 = p_da + e_t2
#p_t4 = p_da + e_t4
#
#sh_t2 = 0.622 * e_t2 / p_t2
#sh_t4 = 0.622 * e_t4 / p_t4
#%%******************************************************************************
test = new_fc_sc.variables['pptrate'][:]

# close the file to write it
new_fc_sc.close()
#%%
testfd = Dataset("sagehenCreek_forcing.nc")
#print testfd.file_format
# read out variables, data types, and dimensions of original forcing netcdf
for varname in testfd.variables.keys():
    var = testfd.variables[varname]
    print (varname, var.dtype, var.dimensions, var.shape)








