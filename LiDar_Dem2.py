import laspy as ls
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

class K_Means:
    def __init__(self, numOfClusters=2, init_centroids=None):
        self.numOfClusters = numOfClusters
        self.centroids={}        
        for i in range(self.numOfClusters):
            self.centroids[i] = init_centroids[i]

    def fit(self,data,cols,cole):
        self.classifications = {}

        for i in range(self.numOfClusters):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset[cols:cole]-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

#%%
infile = ls.file.File("lidardata\lasvegTest.las", mode="r")
#infileGrd = ls.file.File("lidardata\sagehen_testveg.las", mode="r")

# Grab all of the points from the file.
point_records = infile.points

# Grab just the X dimension from the file, and scale it.
def scaled_x_dimension(las_file):
    x_dimension = las_file.X
    scale = las_file.header.scale[0]
    offset = las_file.header.offset[0]
    return(x_dimension*scale + offset)

scaled_x = scaled_x_dimension(infile)
#%%
# Find out what the point format looks like.
pointformat = infile.point_format
for spec in infile.point_format:
    print(spec.name)

#Lets take a look at the header also.
headerformat = infile.header.header_format
for spec in headerformat:
    print(spec.name)
    
#%%# Grab the scaled x, y, and z dimensions and stick them together in an nx3 numpy array
coords = np.vstack((infile.x, infile.y, infile.z)).T
#coordsGrd = np.vstack((infileGrd.x, infileGrd.y, infileGrd.z)).T
#%% calculating the nearest neighbors of a set of points, you might want to use a highly optimized package like FLANN 
dataset = np.vstack([infile.X, infile.Y, infile.Z]).T
#datasetGrd = np.vstack([infileGrd.X, infileGrd.Y, infileGrd.Z]).T
minLat,maxLat = np.min(dataset[:,0]),np.max(dataset[:,0])
minLon, maxLon = np.min(dataset[:,1]),np.max(dataset[:,1])
#%%we’re interested only in the last return from each pulse in order to do ground detection. 
num_returns = infile.num_returns
return_num = infile.return_num
ground_points = infile.points[num_returns == return_num]
#print("%i points out of %i were ground points." % (len(ground_points),len(infile)))
#%% list and array of groundpoints from lidar file
groundPoints_ls = ground_points.tolist()
groundPoints_arr = []
for i in range (len(groundPoints_ls)):
    GPlist = np.array(groundPoints_ls[i])
    groundPoints_arr.append(GPlist[0,0:3])
groundPoints_arr = np.array(groundPoints_arr)
groundPoints_lidar = groundPoints_arr.copy()

#%% implementing my Kmean---------------no need to run
##Number of clusters
#k = np.size(groundPoints_lidar[:,0])
## Number of training data
#n = np.size(dataset[:,0])
## Number of features in the data
##c = dataset.shape[1]
#centers = groundPoints_lidar.copy()
#
#clusters = np.zeros(n)
#distances = np.zeros((n,k))
## Measure the distance to every center
#for i in range(k):
#    distances[:,i] = np.linalg.norm(dataset - centers[i], axis=1)
## Assign all training data to closest center
#clusters = np.argmin(distances, axis = 1)
#%%new metnod (class) for Kmean---------------no need to run
#centroids=groundPoints_lidar.copy()  
## instantiate a class
#clf = K_Means(numOfClusters=k,init_centroids=centroids)
## fit kmean class to data
#clf.fit(dataset)
## get classification 
#classes = clf.classifications
#%% DEM file (.tif) reading (test)
#import gzip
#with gzip.open("lidardata\sagehen_demveg.tin.gz", 'rb') as f:
#     for line in f:        
#         print(line)
driver = gdal.GetDriverByName('GTiff')
filename = "lidardata\demvegTest.tif" #path to raster
demset = gdal.Open(filename)
lyr = gdal.GDALDEMProcessingOptions_swigregister(demset)

dem_hs_ds = gdal.DEMProcessing('', demset, 'hillshade', format='MEM')

band = demset.GetRasterBand(1)
BandType=gdal.GetDataTypeName(band.DataType)
minBand = band.GetMinimum()
maxBand = band.GetMaximum()
minMax = band.ComputeRasterMinMax(True)

elevation = band.ReadAsArray()
elevation[elevation == -9999.] = 1960

minelev = elevation[elevation > 1960] 
#plt.imshow(elevation, cmap='gist_earth')

cols = demset.RasterXSize
rows = demset.RasterYSize
print("Size is {} x {} x {}".format(demset.RasterXSize,demset.RasterYSize,demset.RasterCount))

geotransform = demset.GetGeoTransform()
x0, dx, dxdy, y0, dydx, dy = demset.GetGeoTransform()
nrows, ncols = elevation.shape
x1 = x0 + dx * ncols
y1 = y0 + dy * nrows
extent=[x0, x1, y1, y0]
plt.imshow(elevation, cmap='gist_earth', extent=extent)
plt.savefig('dem_bareearth_test2.png')
#%% whole sagehen DEM
#filename = "lidardata\sagehenDem.tif" #path to raster
#demset1 = gdal.Open(filename)
#
#band1 = demset1.GetRasterBand(1)
#elevation1 = band1.ReadAsArray()
#elevation1[elevation1 >10000] = 1900
#
#x01, dx1, dxdy1, y01, dydx1, dy1 = demset1.GetGeoTransform()
#nrows1, ncols1 = elevation1.shape
#x11 = x01 + dx1 * ncols1
#y11 = y01 + dy1 * nrows1
#extent1=[x01, x11, y11, y01]
#plt.imshow(elevation1, cmap='gist_earth', extent=extent1)
#plt.savefig('sagehenDem.png')
#%% creating centroid from Dem files (test)
latitude =[]
for x in range (ncols):
    latitude.append(x+x0)
longitude = []
for y in range (nrows):
    longitude.append(y0-y)

latitude_rp = np.tile(latitude, nrows)
longitude_rp = np.repeat(longitude, ncols)
elevation_rp = np.reshape(elevation,(nrows*ncols)).T
dem_groundPoints = np.vstack([latitude_rp,longitude_rp,elevation_rp]).T
#%% classification with new grountpoints from dem file
#demGP = dem_groundPoints.copy()
#demGP[:,2]=0
centroids_new=dem_groundPoints[:,0:2]  
k1 = np.size(dem_groundPoints[:,0])
# instantiate a class
clf1 = K_Means(numOfClusters=k1,init_centroids=centroids_new)
# fit kmean class to data
clf1.fit(coords,0,2)
# get classification 
classes1 = clf1.classifications
#%% test classification
testClass1 = []
classes_rplc = []#[x if np.size(x)!= 0 else [0,0,0] for x in classes1]
for cls in range (len (dem_groundPoints)):
    if len(classes1[cls])==0:
        nokhale = [np.array([0,0])]
        classes_rplc.append(nokhale)
    else: classes_rplc.append(classes1[cls])

    testClass1.append(abs(classes_rplc[cls]-dem_groundPoints[cls]))

failclass=[]
for xyidx in range (len(testClass1)):
    for xycl in range (len(testClass1[xyidx])):
        if ((testClass1[xyidx][xycl][0]>0.5) or (testClass1[xyidx][xycl][1]>0.5)):
            failclass.append(xyidx)
            break
        
#%%classification based on DEM geometery
#adx = []
#ady = []
#for cd in coords:
#    adx.append(abs(cd[0]-dem_groundPoints[0,0]))
#    ady.append(abs(cd[1]-dem_groundPoints[0,1]))

#%% ploting
coordinatelidar = coords.copy()
datasetLidar = dataset.copy()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(20,15))
ax = Axes3D(fig)
ax.scatter(dem_groundPoints[:, 0], dem_groundPoints[:, 1], dem_groundPoints[:, 2])
#ax.scatter(coordinatelidar[:, 0], coordinatelidar[:, 1], coordinatelidar[:, 2])
#ax.scatter([x[0] for x in classes1[675]], [x[1] for x in classes1[675]], [x[2] for x in classes1[675]])
for flcl in failclass:
    ax.scatter([x[0] for x in classes1[flcl]], [x[1] for x in classes1[flcl]])
#ax.scatter([x[0] for x in classes1[25]], [x[1] for x in classes1[25]])

plt.savefig('classificationtest.png')
#%% vegtation classification from DEM2014 and las2014
vegClass1 = []
classes_rplc = []#[x if np.size(x)!= 0 else [0,0,0] for x in classes1]
for cls in range (len (dem_groundPoints)):
    if len(classes1[cls])==0:
        nokhale = [np.array([0,0,0])]
        classes_rplc.append(nokhale)
    else: classes_rplc.append(classes1[cls])

    vegClass1.append(classes_rplc[cls]-dem_groundPoints[cls])
#all tree classification
allTreeClass = []
numTreeClass = []
for vgcl in range (len (dem_groundPoints)):
    for pnt in range (len (vegClass1[vgcl])):
        if vegClass1[vgcl][pnt][2]>2:
            #print 'hooorrraaa tree'
            numTreeClass.append(vgcl)
            allTreeClass.append(vegClass1[vgcl])
            break
        
negVegTreeClass = []
numNegVegClass = []
for tcl in range (len (allTreeClass)):
    for pn in range (len (allTreeClass[tcl])):
        if allTreeClass[tcl][pn][2]<-0.15:
            numNegVegClass.append(tcl)
            negVegTreeClass.append(allTreeClass[tcl])
            break        
# trees with low branches
lowVegTreeClass = []
numlowVegClass = []
for tcl in range (len (allTreeClass)):
    for pn in range (len (allTreeClass[tcl])):
        if allTreeClass[tcl][pn][2]>0.15 and allTreeClass[tcl][pn][2]<2:
            numlowVegClass.append(tcl)
            lowVegTreeClass.append(allTreeClass[tcl])
            break
# trees with no low blanches
allTreeClassNum = list(np.arange(len(allTreeClass)))
numNolowVegClass = list(set(allTreeClassNum)-set(numlowVegClass))
nolowVegTreeClass = []
for indx in numNolowVegClass:
    nolowVegTreeClass.append(allTreeClass[indx])    
# open space (no trees, no return between 0.15 -2)
# all low veg
allLowVegClass = []
numallLowVegClass = []
for lvgcl in range (len (dem_groundPoints)):
    for crdnt in range (len (vegClass1[lvgcl])):
        if vegClass1[lvgcl][crdnt][2]<2 and vegClass1[lvgcl][crdnt][2]>0.15:
            numallLowVegClass.append(lvgcl)
            allLowVegClass.append(vegClass1[lvgcl])
            break
#open places
numNotopen = list(set(numallLowVegClass).union(set(numTreeClass)))
allClassNum = list(np.arange(len(vegClass1)))
numopen =  list(set(allClassNum)-set(numNotopen))
allOpenClass = []
for idop in numopen:
    allOpenClass.append(vegClass1[idop])
#%%
#dem_groundPoints_d1 = dem_groundPoints[dem_groundPoints[:,0]>(minLat/100.)-1]
#dem_groundPoints_d2 = dem_groundPoints_d1[dem_groundPoints_d1[:,0]<(maxLat/100.)+1]
#dem_groundPoints_d3 = dem_groundPoints_d2[dem_groundPoints_d2[:,1]<(maxLon/100.)+1]

#a = np.array([0, 1, 2])
#eleva= np.tile(a, (4, 3))
#elev_rp = np.reshape(eleva,(36)).T
import struct

#Y
#Z
#intensity
#flag_byte
#raw_classification
#scan_angle_rank
#user_data
#pt_src_id
#gps_time
#file_sig ???????????????????
#file_source_id
#global_encoding
#proj_id_1 ??????????????
#proj_id_2   ????????????????
#proj_id_3   ?????????
#proj_id_4    ???????????/
#version_major
#version_minor
#system_id
#software_id
#created_day
#created_year
#header_size
#data_offset
#num_variable_len_recs
#data_format_id
#data_record_length
#point_records_count
#point_return_count
#x_scale
#y_scale
#z_scale
#x_offset
#y_offset
#z_offset
#x_max
#x_min
#y_max
#y_min
#z_max
#z_min




































