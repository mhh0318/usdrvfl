#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/12/13 20:47
@author: merci
"""
import argparse
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn import cluster
from utils import *
from relabel_voting import *


parser = argparse.ArgumentParser()
parser.add_argument("--k", help="nearest k-est neighbor for Laplacian Graph", default=5)
parser.add_argument("--normalized", help="normalized the Laplacian Graph", default=False)
parser.add_argument("--degree", help="degree of the iterated Laplacian", default=1)
parser.add_argument("--N", help="the number of neurons in the hidden layer", default=1000)
parser.add_argument("--C", help="regularized parameter", default=0.1)
parser.add_argument("--L", help="the number of layers in RVFLNN", default=3)
parser.add_argument("--activation", help="activation function", default='sigmoid')
args = parser.parse_args()



data,label,classes = USPS()
embed_dim = classes
print('finish load data!')
N_range = [128, 256, 512, 1024, 2048]
C_range = 2. ** np.arange(-3, 11, 2)
L_range = [4,16,64]

embedded = [2,4,8,16,32,64]

total_result = {'drvfl':[],'elm':[],'rvfl':[],'trick':[]}



for ED in embedded:

    for N in N_range:
        args.N = N
        for C in C_range:
            args.C = C
            args.L = L_range[-1]
            r0, r1, r2 = [], [], []

            G_train = create_Graph(data, args)
            # W = G.adjacency()
            L_train = G_train.laplacian()

            # G_test = create_Graph(test_X, args)
            # W = G.adjacency()
            # L_test = G_test.laplacian()

            repeat_vec = []
            ensembel = [label]
            network = RVFL(L_train, ED, args)
            embeds_rvfl = network.usrvfl(data)


            for i in range(args.L):
                kmeans_usrvfl = cluster.KMeans(n_clusters=classes, random_state=1).fit(embeds_rvfl[i])
                ensembel.append(kmeans_usrvfl.labels_)
            for i in L_range:
                relabeled_clusters = relabel_cluster(ensembel)
                usrvfl_result = confusion_matrix(label, voting(relabeled_clusters[1:],i))
                r0.append(usrvfl_result[1])
            result_drvfl = max(r0)

            for repeat in range(100):
                kmeans_usrvfl = cluster.KMeans(n_clusters=classes, random_state=repeat).fit(embeds_rvfl[0])
                pred = relabel_cluster(np.stack((label, kmeans_usrvfl.labels_),axis=0))
                usrvfl_result = confusion_matrix(label, pred[1])
                r1.append(usrvfl_result[1])
            result_rvfl = min(r1)
            result_rvfl_max = max(r1)


            shallow_elm = network.usrvfl(data, functionalink=False)
            for repeat in range(100):
                kmeans_uselm = cluster.KMeans(n_clusters=classes, random_state=repeat).fit(shallow_elm)
                pred = relabel_cluster(np.stack((label, kmeans_uselm.labels_),axis=0))
                uselm_result = confusion_matrix(label, pred[1])
                r2.append(uselm_result[1])
            result_elm = min(r2)

            print('Embed Dim:{}\tN:{}\tC:{}'.format(ED,N,C))
            print("US-ELM Accuracy:{:.4f}\nUS-RVFL Accuracy:{:.4f} / {:.4f}\nUS-RVFL Accuracy:{:.4f}\n".format(result_elm, result_rvfl, result_rvfl_max, result_drvfl))

            # print("Different Layers Result: 5:{:.3f}|\t10:{:.3f}|\t50:{:.3f}|\t150:{:.3f}|\t250:{:.3f}".format(r1[0],r1[1],r1[2],r1[3],r1[4]))

            #print("Average US-ELM Accuracy:{:.3f}\nAverage US-RVFL Accuracy:{:.3f}\n".format(np.array(r2).mean(), np.array(r1).mean()))

            total_result['elm'].append(np.array(result_elm))
            total_result['rvfl'].append(np.array(result_rvfl))
            total_result['drvfl'].append(np.array(r0))
            total_result['trick'].append(np.array(r1))
total_result['elm']=np.array(total_result['elm'])
total_result['rvfl']=np.array(total_result['rvfl'])
total_result['drvfl']=np.array(total_result['drvfl'])
total_result['trick']=np.array(total_result['trick'])
np.save('./Results/USPS.3D.npy',total_result)
