#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2016 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, July 2016
"""
Analyse all coherence data.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cPickle
from matplotlib import cm
import matplotlib as mpl

#%% init
sens = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
        'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
groups = ['Control goup', 'ADHD group']
eyes = ['closed eyes', 'open eyes']
bands = ['delta','theta','alpha','beta']
fs = 250
eeg_no_ch = len(sens)

#%% load
with open('coherence_master.pickle', "rb") as f:
    coherence_master = cPickle.load(f)
# coherence_master = np.zeros((2,30,2,eeg_no_ch,eeg_no_ch, 5))
# these are: groups, people, eyes, ..., bands
#    if "open"  eyes = 1
#    if "Norm"  group = 0

#%% plot example coherence
fig, ax = plt.subplots(nrows=2, ncols=2)
for band in range(4):
    plt.subplot(2,2,band+1)
    plt.title(bands[band])
    plt.imshow(coherence_master[0,7,1,:,:,band], aspect='auto', \
               origin='lower', \
               extent=[0, eeg_no_ch, 0, eeg_no_ch],\
               cmap='plasma')
               
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                          norm=norm, orientation='vertical', 
                          ticks=[tick for tick in np.arange(0.5,1.1,0.1)])   

#%% plot example topographic map
eeg_1020 = img.imread('21ch_eeg.png')

el_centers_dict = {"Cz": (197,181), "C3" : (134,181),
"C4" : (261,181), "T3" : (70,181), "T4" : (324,181),
"Fz" : (197,117), "F3" : (146,116), "F4" : (250,116),
"F7" : (95,107), "F8" : (300,107), "Fp1" : (156,61),
"Fp2" : (239,61), "O1" : (157,301), 
"O2" : (238,301), "Pz" : (197,245),
"P3" : (146,245), "P4" : (250,245), "T5" : (95,255),
"T6" : (300,255)} 

el_centers = np.array([el_centers_dict[item] for item in sens])
fig, ax = plt.subplots(nrows=2, ncols=2)
for band in range(4):
    plt.subplot(2,2,band+1)
    plt.title(bands[band])
    for i in range(eeg_no_ch):     
        for j in range(i):  
            coh = coherence_master[0,7,1,i,j,band]
            if coh > 0.5:
                coh = (coh - 0.5)*2
                col = cm.inferno(coh)
                plt.plot([el_centers[i,0], el_centers[j,0]], 
                         [el_centers[i,1], el_centers[j,1]], 
                         color=col, alpha=coh*.99, linewidth=coh*8)
    plt.imshow(eeg_1020)
    ax = plt.gca()            
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                          norm=norm, orientation='vertical', 
                          ticks=[tick for tick in np.arange(0.5,1.1,0.1)])          
    
#%% average coherence
# example averaging
#for i in range(30):
#    coherence_master[0,i,1,:,:,:] = coherence_master[0,7,1,:,:,:]
#    coherence_master[0,i,0,:,:,:] = coherence_master[0,7,0,:,:,:]
coherence_avg = np.mean(coherence_master, axis=1)

# %% plot average coherence
for g in range(2):
    for e in range(2):
        plt.figure()
        plt.suptitle(groups[g]+' '+eyes[e], fontsize=16)
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.title(bands[i])
            plt.imshow(coherence_avg[g,e,:,:,i], aspect='auto', \
                       origin='lower', \
                       extent=[0, eeg_no_ch, 0, eeg_no_ch],\
                       cmap='plasma')
            plt.title(bands[i])

#%% hemispheres
hemisphere_l = [ind for ind,s in enumerate(sens) if s in 'Fp1F7F3T3C3T5P3O1']
hemisphere_lz = [ind for ind,s in enumerate(sens) if s in 'Fp1F7F3T3C3T5P3O1CzFzPz']
hemisphere_r = [ind for ind,s in enumerate(sens) if s in 'Fp2F8F2T4C4T6P4O2']
hemisphere_rz = [ind for ind,s in enumerate(sens) if s in 'Fp2F8F2T4C4T6P4O2CzFzPz'] 

#%% plot intra heisphere coherence
for g in range(2):
    for e in range(2):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.suptitle(groups[g]+', '+eyes[e], fontsize=16)
        for band in range(4):
            plt.subplot(2,2,band+1)
            plt.title(bands[band])
            for i in hemisphere_l:     
                for j in hemisphere_l:  
                    coh = coherence_avg[g,e,i,j,band]
                    if coh > 0.5:
                        coh = (coh - 0.5)*2
                        col = cm.inferno(coh)
                        plt.plot([el_centers[i,0], el_centers[j,0]], 
                                 [el_centers[i,1], el_centers[j,1]], 
                                 color=col, alpha=coh*.99, linewidth=coh*8)
            for i in hemisphere_r:     
                for j in hemisphere_r:  
                    coh = coherence_avg[g,e,i,j,band] 
                    if coh > 0.5:
                        coh = (coh - 0.5)*2
                        col = cm.inferno(coh)
                        plt.plot([el_centers[i,0], el_centers[j,0]], 
                                 [el_centers[i,1], el_centers[j,1]], 
                                 color=col, alpha=coh*.99, linewidth=coh*8)
            plt.imshow(eeg_1020)
            ax = plt.gca()            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                  norm=norm, orientation='vertical', 
                                  ticks=[tick for tick in np.arange(0.5,1.1,0.1)])        

#%% inter heisphere coherence
for g in range(2):
    for e in range(2):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        plt.suptitle(groups[g]+', '+eyes[e], fontsize=16)
        for band in range(4):
            plt.subplot(2,2,band+1)
            plt.title(bands[band])
            for i in hemisphere_lz:     
                for j in hemisphere_rz:  
                    coh = coherence_avg[g,e,i,j,band]
                    if coh > 0.5:
                        coh = (coh - 0.5)*2
                        col = cm.inferno(coh)
                        plt.plot([el_centers[i,0], el_centers[j,0]], 
                                 [el_centers[i,1], el_centers[j,1]], 
                                 color=col, alpha=coh*.99, linewidth=coh*8)
            plt.imshow(eeg_1020)
            ax = plt.gca()            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=0.5, vmax=1)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                  norm=norm, orientation='vertical', 
                                  ticks=[tick for tick in np.arange(0.5,1.1,0.1)])        
