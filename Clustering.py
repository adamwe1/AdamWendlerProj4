#Clustering.py
#Adam Wendler
#CMSC471
#Project 4
#Preformes K-means clustering on data from file then plots
#clusters are differentiated based on scatter color, 
#centroids are marked with an X

#KNOWN ERROR: Due to the low number of colors matplotlib.pyplot.scatter supports,
#and because there is no constrain rules programed against it,
#neighboring clusters may be assigned the same color.  However, the centroids will
#always apear near the middle of the cluster, so the
#boundries of the cluster may be approximated

import random

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean



#reads file give filepath and converts strings to int
def readFile(filePath):
    values = []
    #open file
    f = open(filePath, "r")
    
    #turns into ordered pairs
    for line in f:
        s = line.split()
        values.append([float(s[0]),float(s[1])])

    f.close()
    return values


#lables dataset
def getLables(dataSet, centroids, k):
    lables = []

    jj=0
    for aa in dataSet:
        distances = []
        #formulate from each centroid
        for kk in centroids:
            distances.append(np.sqrt((np.array(aa) - np.array(kk))**2).sum(axis=0))   
        #lable
        lable = np.argmin(distances)
        lables.append(lable)

    return lables


#new centroids for KMeans
def getCentroids(dataSet, lables, K, xran, yran):
    labledData=[]
    centroids = []
    ii=0
    while ii<K:
        ii+=1
        labledData.append([])

    ii=0
    axxs = []
    ayys = []
    #cluster data
    for aa in dataSet:
        labledData[lables[ii]].append(aa)
        ii+=1   

    
    for jj in labledData:
        #randomize if none in cluster
        if not jj:
            cX = random.randrange(xran[0], xran[1], 1)
            cY = random.randrange(yran[0], yran[1], 1)
            centroids.append([cX,cY])

        #centroid = mean of cluster members
        else:
            centroids.append(np.mean(jj, axis=0))

    #print(centroids)
    return centroids
    

#stops KMeans loop
def stoper(old, new, iterations):
    #Maximum iterations
    if iterations > 10000:
        return True
    #first iterations
    if old == None:
        return False
    #compair lists
    newT =[tuple(lst) for lst in new]
    oldT =[tuple(lst) for lst in old]
    return set(oldT) == set(newT)


#Kmeans
def kMeans(dataSet, K, xran, yran):
    centroids = []
    ii=0
    #randomlly initilize Centroids
    while ii < K:
        ii+=1
        cX = random.randrange(xran[0], xran[1], 1)
        cY = random.randrange(yran[0], yran[1], 1)
        centroids.append([cX,cY])
    old = None
    ii=0
    #training loop
    while not stoper(old, centroids, ii):
        #save old centroids
        old = centroids
        ii+=1
        #get lables from Centroids
        lables = getLables(dataSet, centroids, K)
        #predict centroids using lables
        centroids = getCentroids(dataSet, lables, K, xran, yran)
    
    #lables = getLables(dataSet, centroids, K)
    return centroids, lables


def main():
    #error check
    argl = len(sys.argv)
    if argl < 3:
        print("Please include the number of clusters\n and a file with points")
        return 0

    #get command line arguments
    k = int(sys.argv[1])
    filePath = sys.argv[2]

    if(k==0):
        print("Cannot split into 0 clusters.")
        return 0

    colors = ['b','g','r','c','m','y']

    #extend colors list as need be
    ii=0
    while len(colors) < k:
        colors.append(colors[ii])
        ii+=1

    #initialize clusters
    clusters = []
    while len(clusters)<k:
        clusters.append([])



    X = readFile(filePath)

    ii=0
    axxs = []
    ayys = []
    for aa in X:
        axxs.append(aa[0])
        ayys.append(aa[1])
        ii+=1   

    x_min = int(min(axxs))
    x_max = int(max(axxs))
    y_min = int(min(ayys))
    y_max = int(max(ayys))

    centroids, lables = kMeans(X, k,[x_min,x_max],[y_min,y_max])

    ii=0
    for aa in X:
        clusters[lables[ii]].append(aa)
        ii+=1   



    #create graph
    plt.figure(1)
    plt.clf()

    #plot each cluster seperatlly 
    ii=0
    for aa in clusters:
        xxs = []
        yys = []
        for bb in aa:
            xxs.append(bb[0])
            yys.append(bb[1])
        plt.scatter(xxs, yys, color = colors[ii])
        ii+=1

    #find and plot centroids for each cluster
    for aa in centroids:
        plt.scatter(aa[0], aa[1], marker = "x", s = 169, linewidths = 3, color = "k")

    plt.title("K-means")
    plt.xlim(min(x_min,-10)*6/5,max(x_max,10)*6/5)    
    plt.ylim(min(y_min,-10)*6/5,max(y_max,10)*6/5)
    plt.show()
    return 0




main()
