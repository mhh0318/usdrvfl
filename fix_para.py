#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/3 18:44
@author: merci
"""
from sklearn import cluster
from utils import *
from relabel_voting import *
import argparse
import numpy as np
from sklearn import datasets
import time

parser = argparse.ArgumentParser()
parser.add_argument("--k", help="nearest k-est neighbor for Laplacian Graph", default=5)
parser.add_argument("--normalized", help="normalized the Laplacian Graph", default=False)
parser.add_argument("--degree", help="degree of the iterated Laplacian", default=1)
parser.add_argument("--N", help="the number of neurons in the hidden layer", default=1000)
parser.add_argument("--C", help="regularized parameter", default=0.1)
parser.add_argument("--L", help="the number of layers in RVFLNN", default=3)
parser.add_argument("--activation", help="activation function", default='relu')
args = parser.parse_args()
L_range = [1,5,10,50,150,250]
# L_range = [1]
args.L = L_range[-1]
data,label,classes = ETH80()
args.N = 1000
args.C = 2.0

Embed = 16



'''
begin = time.time()
kmeans_usrvfl = cluster.KMeans(n_clusters=classes, random_state=100).fit(data)
pred = relabel_cluster(np.stack((label, kmeans_usrvfl.labels_), axis=0))
cost_kmean = time.time()-begin


begin = time.time()
G_train = create_Graph(data, args)
L_train = G_train.laplacian()
network = RVFL(L_train, Embed, args)

shallow_elm = network.usrvfl(data, functionalink=False)
kmeans_uselm = cluster.KMeans(n_clusters=classes, random_state=0).fit(shallow_elm)
pred = relabel_cluster(np.stack((label, kmeans_uselm.labels_), axis=0))
cost_elm = time.time()-begin


begin = time.time()
G_train = create_Graph(data, args)
L_train = G_train.laplacian()
network = RVFL(L_train, Embed, args)

embeds_rvfl = network.usrvfl(data)
kmeans_uselm = cluster.KMeans(n_clusters=classes, random_state=0).fit(embeds_rvfl[0])
pred = relabel_cluster(np.stack((label, kmeans_uselm.labels_), axis=0))
cost_rvfl = time.time()-begin
'''
for l in L_range:
    args.L = l
    begin = time.time()
    G_train = create_Graph(data, args)
    L_train = G_train.laplacian()
    network = RVFL(L_train, Embed, args)

    embeds_rvfl = network.usrvfl(data)
    re = []
    ensembel = []
    for i in range(args.L):
        kmeans_usrvfl = cluster.KMeans(n_clusters=classes, random_state=100).fit(embeds_rvfl[i])
        ensembel.append(kmeans_usrvfl.labels_)
    for i in L_range:
        relabeled_clusters = relabel_cluster(ensembel)
        usrvfl_result = confusion_matrix(label, voting(relabeled_clusters, i))
        re.append(usrvfl_result[1])
    cost_drvfl = time.time()-begin
    a = np.array(max(re))
    print(l,cost_drvfl)


'''
begin = time.time()
kmeans_usrvfl = cluster.SpectralClustering(n_clusters=classes, random_state=100).fit(data)
pred = relabel_cluster(np.stack((label, kmeans_usrvfl.labels_), axis=0))
cost_SC = time.time()-begin
'''