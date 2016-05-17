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

import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



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


def main():
    #error check
    argl = len(sys.argv)
    if argl < 3:
        print("Please include the number of clusters\n and a file with points")
        return 0

    #get command line arguments
    k = int(sys.argv[1])
    filePath = sys.argv[2]
    colors = ['b','g','r','c','m','y']

    #extend colors list as need be
    ii=0
    while len(colors) < k:
        colors.append(colors[ii])
        ii+=1
    
    #create arrays for cluster data
    clusters = []
    while len(clusters) < k:
        clusters.append([])

    #read file, return to X
    X = readFile(filePath)
    #create KMeans model with k clusters
    meanMachine = KMeans(k)
    meanMachine.fit(X)

    #save predictions
    lables = meanMachine.predict(X)
    
    #find max x and y
    ii=0
    axxs = []
    ayys = []
    for aa in X:
        axxs.append(aa[0])
        ayys.append(aa[1])
        clusters[lables[ii]].append(aa)
        ii+=1   

    x_min = min(axxs)
    x_max = max(axxs)
    y_min = min(ayys)
    y_max = max(ayys)

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
    centroids = meanMachine.cluster_centers_
    for aa in centroids:
        plt.scatter(aa[0], aa[1], marker = "x", s = 169, linewidths = 3, color = "k")

    plt.title("K-means")
    plt.xlim(x_min*6/5,x_max*6/5)    
    plt.ylim(y_min*6/5,y_max*6/5)
    plt.show()





main()
