#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/12/13 20:47
@author: merci
"""
import numpy as np
from scipy import sparse
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import scipy.io as scio

from sklearn.preprocessing import normalize

class create_Graph:
    def __init__(self, data, args):
        self.X = data
        self.args = args

    def adjacency(self):
        n_sample,n_D = self.X.shape
        dist= squareform(pdist(self.X,metric='euclidean'))
        topk = np.argsort(dist,axis=-1)[:,1:self.args.k+1]
        A = np.zeros((n_sample,n_sample))
        np.put_along_axis(A,indices=topk,values=1,axis=-1)
        return A+(A!=A.T)*A.T

        #return A

    def laplacian(self):
        A = self.adjacency()
        sparse_A = sparse.csr_matrix(A)
        laplacian_A = sparse.csgraph.laplacian(sparse_A, normed=self.args.normalized)
        if self.args.degree > 1:
            laplacian_A = laplacian_A**self.args.degree
        return laplacian_A.A

class RVFL:
    def __init__(self, Graph, classes, args):

        self.args = args
        self.Graph = Graph
        self.classes = classes

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def solution(self, x, L):
        n_sample, n_D = x.shape
        if n_D<n_sample:
            A = np.eye(n_D) + self.args.C*(x.T.dot(L).dot(x))
            B = x.T.dot(x)
            W, V = sparse.linalg.eigs(A, k=self.classes+1, M=B, sigma=0, which='LM')
            V = np.real(V)
            idx = np.argsort(W)
            norm_term = x.dot(V[:,idx[1:]])
            weight = V[:,idx[1:]]*np.sqrt(1. / sum(norm_term**2))
        else:
            B = x.dot(x.T)
            A = np.eye(n_sample) + self.args.C*(L.dot(B))
            # BUGs in linalg.eigs
            W, V = sparse.linalg.eigs(A, k=self.classes+1, M=B, sigma=0, which= 'LM')
            V = np.real(V)
            idx = np.argsort(W)
            norm_term = B.dot(V[:, idx[1:]])
            weight = x.T.dot(V[:, idx[1:]]) * np.sqrt(1. / sum(norm_term**2))
        return weight

    def usrvfl(self, data, functionalink=True):
        self.X = data
        A = []
        beta = []
        weights = []
        biases = []
        A_input = self.X
        n_sample, n_D = self.X.shape
        rand_seed = np.random.RandomState(1)

        for i in range(self.args.L):
            if i == 0:
                # w = 2 * rand_seed.rand(n_D, self.args.N) - 1
                w = 2 * rand_seed.rand(self.args.N, n_D) - 1
            else:
                w = 2 * rand_seed.rand(self.args.N, self.args.N) - 1
            w = w.T
            b = rand_seed.rand(1, self.args.N)
            weights.append(w)
            biases.append(b)

            if self.args.activation == 'relu':
                A_ = np.matmul(A_input, w)
                # layer normalization
                A_mean = np.mean(A_, axis=0)
                A_std = np.std(A_, axis=0)
                A_ = (A_ - A_mean) / A_std
                A_ = A_ + np.repeat(b, n_sample, 0)
                #A_ = self.relu(A_)
                A_ = self.selu(A_)
            elif self.args.activation == 'kernel':
                A_ = 1. / (1 + np.exp(-np.dot(A_input, w)))
            elif self.args.activation == 'sigmoid':
                A_ = np.matmul(A_input, w)
                # layer normalization
                A_mean = np.mean(A_, axis=0)
                A_std = np.std(A_, axis=0)
                A_ = (A_ - A_mean) / A_std
                A_ = A_ + np.repeat(b, n_sample, 0)
                # A_ = self.relu(A_)
                A_ = self.sigmoid(A_)

            if functionalink ==True:
                A_merge = np.concatenate([self.X, A_, np.ones((n_sample, 1))], axis=1)
            else:
                A_merge = A_
                beta_ = self.solution(A_merge, self.Graph)
                beta.append(beta_)
                embed = A_merge.dot(beta_)
                # embed = H.dot(beta_)
                return embed

            beta_ = self.solution(A_merge, self.Graph)
            beta.append(beta_)
            embed = A_merge.dot(beta_)
            # embed = H.dot(beta_)
            A.append(embed)
            A_input = A_

        return A

def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    cm = np.zeros((n_labels,n_labels))
    for n in range(len(y_pred)):
        i = y_pred[n]
        j = y_true[n]
        cm[i,j] += 1
    cm /= cm.sum(axis=0)
    accuracy = np.diagonal(cm).mean()
    return cm, accuracy


