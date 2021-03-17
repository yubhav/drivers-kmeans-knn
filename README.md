# drivers-kmeans-knn

## Problem Statement
Lithionpower is the largest provider of electric vehicle (e-vehicle) batteries. It provides battery on a rental model to e-vehicle drivers. Drivers rent battery typically for a day and then replace it with a charged battery from the company. Lithionpower has a variable pricing model based on driver's driving history. As the life of a battery depends on factors such as overspeeding, distance driven per day, etc. You as a ML expert have to create a cluster model based upon KNN, where drivers can be grouped together based on the driving data.

Drivers will be incentivized based on the cluster, so grouping has to be accurate.

id: unique id of the driver
mean_dist_day: mean distance driven by driver per day
mean_over_speed_perc: mean percentage of time when driver exceeded the 5 mph over speed limit
increase in profits: up to 15-20% as drivers with poor history will be charged more

## Method

 - added a new field "profit" to the dataset as it was missing
 - used k-means for clustering (number of clusters taken as four)
 - used knn for classification
 - printed classification report
 - appropriate comments have been added in the file before each step

## References
The original driver-data.csv dataset was taken from https://github.com/Suhong-Liang/K-Mean-Clustering
It contains 4000 records