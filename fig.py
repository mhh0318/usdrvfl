#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/2/17 19:18
@author: merci
"""
import numpy as np
import matplotlib.pyplot as plt
import os
root = './Results/af'

def cat_result():
    newdict = {'rvfl':[],'drvfl':[],'elm':[]}
    a = np.load('/home/hu/usedrvfl/pollen.npy',allow_pickle=True)
    b = np.load('/home/hu/usedrvfl/pollen2000.npy',allow_pickle=True)
    aa = a.item().get('rvfl')
    bb = b.item().get('rvfl')
    newdict['rvfl'] = np.concatenate((aa.reshape(12,-1),bb.reshape(6,-1)),0)
    aa = a.item().get('elm')
    bb = b.item().get('elm')
    newdict['elm'] = np.concatenate((aa.reshape(12,-1),bb.reshape(6,-1)),0)
    aa = a.item().get('drvfl')
    bb = b.item().get('drvfl')
    newdict['drvfl'] = np.concatenate((aa.reshape(12,-1,3),bb.reshape(6,-1,3)),0)
    np.save('./Results/pollen.npy',newdict)

def comparision_elm_rvfl():
    plt.figure(num=(32),figsize = (10,5))
    for i,file in enumerate(os.listdir(root)):
        plt.subplot(3,2,1+i)
        tmp = np.load(os.path.join(root,file),allow_pickle=True)
        if file == 'mnist05.npy':
            rvfl = tmp.item().get('rvfl')[:72].reshape(18,-1).max(0)
            elm = tmp.item().get('elm')[:72].reshape(18,-1).max(0)
            rvfl = np.append(rvfl, tmp.item().get('rvfl')[72:].max())
            elm = np.append(elm, tmp.item().get('elm')[72:].max())
        elif file == 'umnist.npy':
            rvfl = tmp.item().get('rvfl')[:72].reshape(18,-1).max(0)
            elm = tmp.item().get('elm')[:72].reshape(18,-1).max(0)
            rvfl = np.append(rvfl, tmp.item().get('rvfl')[72:].max())
            elm = np.append(elm, tmp.item().get('elm')[72:].max())
        else:
            rvfl = np.array(tmp.item().get('rvfl')).reshape(18,-1).max(0)
            elm = np.array(tmp.item().get('elm')).reshape(18,-1).max(0)
            if rvfl.mean()<elm.mean():
                rvfl,elm = elm,rvfl
        plt.plot(rvfl[:5],'^-',label='usRVFL',alpha=0.6)
        plt.plot(elm[:5],'o-',label='US-ELM',alpha=0.6)
        plt.xticks([0, 1, 2, 3, 4],[2,4,8,16,32])
        # plt.xticks([0, 1, 2, 3],[2,4,8,16])
        plt.yticks([np.array([elm,rvfl]).min().round(2),np.array([elm,rvfl]).max().round(2)])
        plt.xlabel('Embedded Dims')
        plt.ylabel('Accuracy')
        plt.title("{}".format(file.split('.')[0].upper()))
        plt.legend()
        print(i,file)
    plt.tight_layout()
    plt.show()

def plot3d():
    fig = plt.figure(num=(43),figsize = (9,12))
    for i,file in enumerate(os.listdir(root)):
        ax = fig.add_subplot(4,3,1+i, projection='3d')
        tmp = np.load(os.path.join(root, file), allow_pickle=True)
        rvfl = tmp.item().get('rvfl').reshape(5,7)
        # elm = tmp.item().get('elm').reshape(6,3,6)
        x = np.arange(0, 7)
        y = np.arange(0, 5)
        X, Y = np.meshgrid(x, y)
        print(i, file)
        ax.plot_surface(X, Y, rvfl, rstride = 1, cstride = 1, cmap='rainbow')
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_xticklabels([-3, -1, 1, 3, 5, 7, 9])
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels([128, 256, 512, 1024, 2048])
        ax.set_xlabel('$\log(\lambda)$')
        ax.set_ylabel('N')
        ax.set_zlabel('ACC(%)')
        ax.set_title("{}".format(file.split('.')[0].upper()))
    fig.canvas.draw()
    plt.tight_layout()
    plt.show()

def plotlayers():
    plt.figure( figsize=(10, 5))
    heights = []
    for i, file in enumerate(os.listdir(root)):
        print(i, file)
        tmp = np.load(os.path.join(root, file), allow_pickle=True)
        if file == 'mnist05.npy':
            pass
        elif file == 'umnist.npy':
            rvfl = tmp.item().get('rvfl')[:90].reshape(18, -1)
            drvfl = tmp.item().get('drvfl')[:90].reshape(18, -1, 3)
            rvfl = np.append(rvfl.max(0), tmp.item().get('rvfl')[90:])
            drvfl = np.vstack((drvfl.max(0), tmp.item().get('drvfl')[90:]))
            height = np.concatenate((rvfl.reshape(6, -1), drvfl), 1)
            heights.append(height)
        else:
            rvfl = np.array(tmp.item().get('rvfl')).reshape(18, -1)
            drvfl = np.array(tmp.item().get('drvfl')).reshape(18, -1, 3)
            height = np.concatenate((rvfl.max(0).reshape(6,-1),drvfl.max(0)),1)
            heights.append(height)
    X = np.arange(5)+1
    k = np.array(heights).max(1)
    plt.bar(X-0.3, k[:,0], alpha=0.9, width = 0.2, label = 'usRVFL')
    plt.bar(X-0.1, k[:,1], alpha=0.9, width = 0.2, label = 'usdRVFL with 4 layer')
    plt.bar(X+0.1, k[:,2], alpha=0.9, width = 0.2, label = 'usdRVFL with 16 layer')
    plt.bar(X+0.3, k[:,3], alpha=0.9, width = 0.2, label = 'usdRVFL with 32 layer')

    plt.xticks([1, 2, 3, 4, 5],['USPST','UMIST','POLLEN','ETH80','ORL'])
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

rvfls = []
drvfls = []
fig = plt.figure(num=(12), figsize=(10, 5))
for i, file in enumerate(os.listdir(root)):
    print(i, file)
    tmp = np.load(os.path.join(root, file), allow_pickle=True)
    rvfl = tmp.item().get('rvfl').reshape(6,-1)
    drvfl = tmp.item().get('drvfl').reshape(6,-1)
    rvfls.append(rvfl)
    drvfls.append(drvfl)
rvfl = np.array(rvfls).max(1)
drvfl = np.array(drvfls).max(1)
X = np.arange(6) + 1
plt.subplot(121)
plt.bar(X-0.3, rvfl[:,0], alpha=0.9, width = 0.2, label = 'ReLU')
plt.bar(X-0.1, rvfl[:,1], alpha=0.9, width = 0.2, label = 'SeLU')
plt.bar(X+0.1, rvfl[:,2], alpha=0.9, width = 0.2, label = 'Sigmoid')
plt.bar(X+0.3, rvfl[:,3], alpha=0.9, width = 0.2, label = 'Gaussian Kernel')
plt.xticks([1, 2, 3, 4, 5, 6],['ETH80','ORL','POLLEN','UMIST','MNIST05','USPS'])
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.legend(loc="upper right")
plt.subplot(122)
plt.bar(X-0.3, drvfl[:,0], alpha=0.9, width = 0.2, label = 'ReLU')
plt.bar(X-0.1, drvfl[:,1], alpha=0.9, width = 0.2, label = 'SeLU')
plt.bar(X+0.1, drvfl[:,2], alpha=0.9, width = 0.2, label = 'Sigmoid')
plt.bar(X+0.3, drvfl[:,3], alpha=0.9, width = 0.2, label = 'Gaussian Kernel')
plt.xticks([1, 2, 3, 4, 5, 6],['ETH80','ORL','POLLEN','UMIST','MNIST05','USPS'])
plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()