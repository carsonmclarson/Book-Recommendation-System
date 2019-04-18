# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
import networkx
from operator import itemgetter
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# Read the data from the amazon-books.txt;
# populate amazonProducts nested dicitonary;
# key = ASIN; value = MetaData associated with ASIN
fhr = open('./amazon-books.txt', 'r', encoding='utf-8', errors='ignore')
amazonBooks = {}
fhr.readline()
for line in fhr:
    cell = line.split('\t')
    MetaData = {}
    MetaData['Id'] = cell[0].strip() 
    ASIN = cell[1].strip()
    MetaData['Title'] = cell[2].strip()
    MetaData['Categories'] = cell[3].strip()
    MetaData['Group'] = cell[4].strip()
    MetaData['SalesRank'] = int(cell[5].strip())
    MetaData['TotalReviews'] = int(cell[6].strip())
    MetaData['AvgRating'] = float(cell[7].strip())
    MetaData['DegreeCentrality'] = int(cell[8].strip())
    MetaData['ClusteringCoeff'] = float(cell[9].strip())
    amazonBooks[ASIN] = MetaData
fhr.close()

# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=networkx.read_weighted_edgelist(fhr)
fhr.close()

print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'

# get book metadata
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks[purchasedAsin]['Title'])
print ("SalesRank = ", amazonBooks[purchasedAsin]['SalesRank'])
print ("TotalReviews = ", amazonBooks[purchasedAsin]['TotalReviews'])
print ("AvgRating = ", amazonBooks[purchasedAsin]['AvgRating'])
print ("DegreeCentrality = ", amazonBooks[purchasedAsin]['DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks[purchasedAsin]['ClusteringCoeff'])
    

#Get the depth-1 ego network of purchasedAsin from copurchaseGraph,

n = purchasedAsin
G = copurchaseGraph
purchasedAsinEgoGraph = networkx.Graph()
purchasedAsinEgoGraph = networkx.ego_graph(G, n, radius=1)



pos = networkx.spring_layout(G)
plt.figure(figsize=(10,10))
networkx.draw_networkx_labels(G,pos,font_size=20)
networkx.draw(G, pos=pos, node_size=1500, node_color='r', edge_color='r', style='dashed')
networkx.draw(purchasedAsinEgoGraph, pos=pos, node_size=10, node_color='b', edge_color='b', style='solid')
plt.show()

#Use the island method on purchasedAsinEgoGraph to only retain edges with 
#threshold >= 0.5, lower threshold to 0 for results with fewer than 5 neighbors
purchasedAsinNeighbors = []
threshold = 0.5

purchasedAsinEgoTrimGraph = networkx.Graph()
for f, t, e in purchasedAsinEgoGraph.edges(data=True):
    if e['weight'] >= threshold:
        purchasedAsinEgoTrimGraph.add_edge(f, t, weight=e['weight'])
n = purchasedAsin
purchasedAsinNeighbors = [i for i in purchasedAsinEgoTrimGraph.neighbors(n)]

if len(purchasedAsinNeighbors) < 5:  #drop threshold if fewer than 5 neighbors in list
    threshold = .0
    for f, t, e in purchasedAsinEgoGraph.edges(data=True):
        if e['weight'] >= threshold:
            purchasedAsinEgoTrimGraph.add_edge(f, t, weight=e['weight'])
    n = purchasedAsin
    purchasedAsinNeighbors = [i for i in purchasedAsinEgoTrimGraph.neighbors(n)]

#Find the list of neighbors of the purchasedAsin in the 
purchasedAsinNeighbors = []
n = purchasedAsin
purchasedAsinNeighbors = [i for i in purchasedAsinEgoTrimGraph.neighbors(n)]

#composite measure to make Top Five book recommendations.
metrics = []
for asin in purchasedAsinNeighbors:
    metrics.append(
    [asin,
    (amazonBooks[asin]['Title']),
    (amazonBooks[asin]['SalesRank']),
    (amazonBooks[asin]['TotalReviews']),
    (amazonBooks[asin]['AvgRating']),
    (amazonBooks[asin]['DegreeCentrality']),
    (amazonBooks[asin]['ClusteringCoeff'])])
df = pd.DataFrame(metrics, columns = ['ASIN','Title','SalesRank','totalReviews','AvgRating','DegreeCentrality','ClusteringCoeff'])
df = df.set_index(['ASIN'])
 
#remove books with no reviews
df = df[df['totalReviews']>0] 

min_max_scaler = preprocessing.MinMaxScaler()

RatingNorm = min_max_scaler.fit_transform(df[['AvgRating']])

#use inverse of sales rank to get highest ranked books normalized to 1
RankNorm = min_max_scaler.fit_transform(1/df[['SalesRank']])

DegreeNorm = min_max_scaler.fit_transform(df[['DegreeCentrality']])

df['SalesRankNorm'] = RankNorm
df['AvgRatingNorm'] = RatingNorm
df['DegreeNorm'] = DegreeNorm


df['CompositeScore'] = (df['DegreeNorm']*2+df['ClusteringCoeff']+df['AvgRatingNorm']+df['SalesRankNorm'])

df = df.sort_values(['CompositeScore'],ascending = False)

# Print Top 5 recommendations (ASIN, and associated Title, Sales Rank, 
# TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff)
df.drop(['CompositeScore','DegreeNorm','AvgRatingNorm','SalesRankNorm'],axis=1, inplace=True)
print(df.head())


