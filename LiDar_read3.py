import laspy as ls
import numpy as np
import scipy
from scipy.spatial.kdtree import KDTree
import matplotlib.pyplot as plt

class K_Means:
    def __init__(self, numOfClusters=2, init_centroids=None):
        self.numOfClusters = numOfClusters
        self.centroids={}        
        for i in range(self.numOfClusters):
            self.centroids[i] = init_centroids[i]

    def fit(self,data):
        self.classifications = {}

        for i in range(self.numOfClusters):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)


    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

#%%
infile = ls.file.File("lidardata\sagehen_testveg.las", mode="r")
infileGrd = ls.file.File("lidardata\sagehen_testGrd.las", mode="r")

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

#%%
# Grab the scaled x, y, and z dimensions and stick them together in an nx3 numpy array
coords = np.vstack((infile.x, infile.y, infile.z)).T
coordsGrd = np.vstack((infileGrd.x, infileGrd.y, infileGrd.z)).T

#%% calculating the nearest neighbors of a set of points, you might want to use a highly optimized package like FLANN 
dataset = np.vstack([infile.X, infile.Y, infile.Z]).T
datasetGrd = np.vstack([infileGrd.X, infileGrd.Y, infileGrd.Z]).T

#%%we’re interested only in the last return from each pulse in order to do ground detection. 
#We can easily figure out which points are the last return by finding out for which points return_num is equal to num_returns.
# Grab the return_num and num_returns dimensions
num_returns = infile.num_returns
return_num = infile.return_num
ground_points = infile.points[num_returns == return_num]
print("%i points out of %i were ground points." % (len(ground_points),len(infile)))

num_returnsG = infileGrd.num_returns
return_numG = infileGrd.return_num
ground_pointsGrd = infileGrd.points[num_returnsG == return_numG]
#%%
groundPoints_ls = ground_points.tolist()
#groundPoints_arr = np.array(groundPoints_ls)
groundPoints_arr = []
for i in range (len(groundPoints_ls)):
    GPlist = np.array(groundPoints_ls[i])
    groundPoints_arr.append(GPlist[0,0:3])
groundPoints_arr = np.array(groundPoints_arr)

#%%
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(figsize=(20,15))
#ax = Axes3D(fig)
#ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2])
#plt.savefig('3DallPoints.png')
#%% implementing Kmean
#Number of clusters
k = np.size(groundPoints_arr[:,0])
# Number of training data
n = np.size(dataset[:,0])
# Number of features in the data
#c = dataset.shape[1]

centers = groundPoints_arr.copy()

clusters = np.zeros(n)
distances = np.zeros((n,k))

# Measure the distance to every center
for i in range(k):
    distances[:,i] = np.linalg.norm(dataset - centers[i], axis=1)
# Assign all training data to closest center
clusters = np.argmin(distances, axis = 1)

#%%new metnod (class) for Kmean
centroids=groundPoints_arr.copy()  

# instantiate a class
clf = K_Means(numOfClusters=k,init_centroids=centroids)

# fit kmean class to data
clf.fit(dataset)

# get classification 
classes = clf.classifications

#%% DEM file (.tif) reading
#import gzip
#with gzip.open("lidardata\sagehen_demveg.tin.gz", 'rb') as f:
#     for line in f:        
#         print(line)

from osgeo import gdal
demfile = gdal.Open("lidardata\output.tin.tif", gdal.GA_ReadOnly)
lyr = gdal.GDALDEMProcessingOptions_swigregister(demfile)

print("Driver: {}/{}".format(demfile.GetDriver().ShortName,demfile.GetDriver().LongName))
print("Size is {} x {} x {}".format(demfile.RasterXSize,demfile.RasterYSize,demfile.RasterCount))
print("Projection is {}".format(demfile.GetProjection()))

geotransform = demfile.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

band = demfile.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))
      
min0 = band.GetMinimum()
max0 = band.GetMaximum()
if not min or not max:
    (min,max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min,max))
      
if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))
      
if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

scanline = band.ReadRaster(xoff=0, yoff=0, xsize=band.XSize, ysize=1,
                           buf_xsize=band.XSize, buf_ysize=1,
                           buf_type=gdal.GDT_Float32)
import struct
tuple_of_floats = struct.unpack('f' * band.XSize, scanline)



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




































