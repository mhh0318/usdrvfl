#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/12/16 14:11
@author: merci
"""
import numpy as np
import networkx
from networkx.algorithms.core import core_number
import community
from collections import namedtuple


def community_ecg(self, weight='weight', ens_size=16, min_weight=0.05):

    W = {k: 0 for k in self.edges()}
    ## Ensemble of level-1 Louvain
    for i in range(ens_size):
        d = community.generate_dendrogram(self, weight=weight, randomize=True)
        l = community.partition_at_level(d, 0)
        for e in self.edges():
            W[e] += int(l[e[0]] == l[e[1]])
    ## vertex core numbers
    core = core_number(self)
    ## set edge weights
    for e in self.edges():
        m = min(core[e[0]], core[e[1]])
        if m > 1:
            W[e] = min_weight + (1 - min_weight) * W[e] / ens_size
        else:
            W[e] = min_weight

    networkx.set_edge_attributes(self, W, 'W')
    part = community.best_partition(self, weight='W')
    P = namedtuple('Partition', ['partition', 'W', 'CSI'])
    w = list(W.values())
    CSI = 1 - 2 * np.sum([min(1 - i, i) for i in w]) / len(w)
    p = P(part, W, CSI)
    return p


community.ecg = community_ecg