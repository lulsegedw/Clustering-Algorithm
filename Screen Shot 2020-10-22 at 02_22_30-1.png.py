#%%
# By: Lul Woreta
# Course: Data Mining
# 11/08/2020
#
# The purpose of this work is to find the top 20 US cities based on the density of zip codes in an area.
# A cluster approach will be used by running the DBSCAN (Density-based spatial clustering of applications with noise)
# algorithm, since it is a density based clustering algorithm that produces a partitional clustering, 
# in which the number of clusters is automatically determined by the algorithm.
# The results will be plotted to visualize and validate the results.

from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
import matplotlib as mpl
from matplotlib.colors import rgb2hex
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
from sklearn.cluster import DBSCAN
# %%

#URL of the dataset to obtain the zip codes
#https://public.opendatasoft.com/explore/dataset/us-zip-code-latitude-and-longitude/export/?location=2,43.32518,-69.78516&basemap=jawg.streets

df = pd.read_csv('us-zip-code-latitude-and-longitude.csv', sep=';')

#Remove the data of non-contiguos states (Alaska and Hawaii)
df.drop(df[df['State'] == 'AK'].index, inplace=True)
df.drop(df[df['State'] == 'HI'].index, inplace=True)

#Remove duplicates rows
df.drop_duplicates(inplace=True)


# %%
# To use de DBSCAN algorithm we need to determine the epsilon (Eps) and minimum points (MinPts) parameters.
#
# Eps defines the radius of neighborhood around a point x. This values was approximated first by searching the smallest
# city in size on the Internet, and then calculating its aproximate radius using Google maps. The first value used was
# 0.27, but then was adjusted to 0.29 which produced better results.
#
# MinPts is the minimum number of neighbors within “Eps” radius. We can aggregate by Cities and have an idea of 
# what are the most dense and set an appropriate value for this parameter.

zipCounts = df.groupby(['City'], as_index=False).count()
sorted_cities = zipCounts[['City', 'Zip']].sort_values(by=['Zip'], ascending=False).head(20)
sorted_cities

# It can be seen that the city in the 20th position has 82 zip codes, so this value was chosen first as the MinPts 
# parameter. In the subsequents test this value changed to 80, which produced better results.


# %%
# The DBSCAN algorithm runs using the function with the same name from the Scikit-learn library
# The algorithm uses the coordinates of each zip code to obtain the clusters
zipCoords = df[['Latitude', 'Longitude']]
model = DBSCAN(eps=0.29, min_samples=80)
model.fit(zipCoords)


# %%
# The resulting model classify each row to a specific cluster with a label
labels = model.labels_
unique, counts = np.unique(labels, return_counts=True)
# Here can be checked the number of clusters automatically created.
len(counts)


# %%
# The number of clusters obtained are then ordered in descending order by the number 
# of zip codes thy have and stored the first 20 clusters in the f20cities data frame
# Here the noise points are removed, as they are located at the beggining of the array
clusters = pd.DataFrame({'label':unique[1:], 'counts':counts[1:]})
f20cities = clusters.sort_values(by=['counts'], ascending=False).head(20)


# %%
# A United States map is created using the Basemap library 
map1 = Basemap(llcrnrlon = -125, 
            llcrnrlat = 23,
            urcrnrlon = -66,
            urcrnrlat = 50,
            projection ='mill',
            resolution = 'c')

map1.drawcountries(color='blue', linewidth=2.0)
map1.drawstates()
map1.drawcoastlines()
map1.drawlsmask(ocean_color="aqua")

# For each cluster in the first 20 cities with more zip codes
# every zip code is ploted with the same color using its map coordinate
for i, cluster_row in f20cities.iterrows():
   rgb = [rd.random(), rd.random(), rd.random()]
   clusteri = zipCoords[labels == cluster_row['label']]
   for j, zip_loc in clusteri.iterrows():
      x, y = map1(zip_loc['Longitude'], zip_loc['Latitude'])
      map1.plot(x, y, marker='s',color=rgb)

plt.show()


# %%
map2 = Basemap(llcrnrlon = -125, 
            llcrnrlat = 23,
            urcrnrlon = -66,
            urcrnrlat = 50,
            projection ='mill',
            resolution = 'c')

map2.drawcountries(color='blue', linewidth=2.0)
map2.drawstates()
map2.drawcoastlines()
map2.drawlsmask(ocean_color="aqua")

# For each group the obtained by grouping the dataset by Cities, one city (the first occurrence),
# is selected to represent the city and it is plotted to have a reference of the precise location
# of this city and compare the result with the clustering algorithm
for i, cluster_row in sorted_cities.iterrows():
   rgb = [rd.random(), rd.random(), rd.random()]
   clusteri = df[df.City == cluster_row['City']]
   reference = clusteri.iloc[0]
   x, y = map2(reference['Longitude'], reference['Latitude'])
   map2.plot(x, y, marker='o', markersize=10, label=reference['City'], color=rgb)

fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), prop=fontP)
plt.show()
# %%
# As can be inspected, 10 out of 20 of the total clusters match, that is, a 50%.
# There are some cases in which to cities are merged in one cluster, that is the case of San Diego
# and Los Angeles or Columbia and Atlanta. This happen because this cities are closed to each other
# and the zip codes are located uniformly throughout the cities.
# On the other hand, large cities may have the zip codes sparsely located and they are not correctly
# clustered.
# %%
