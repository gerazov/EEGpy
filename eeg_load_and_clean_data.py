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
Loading of EEG files, artifact removal and saving.

"""
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import edfplus
from sklearn.decomposition import FastICA
import matplotlib.image as img
import scipy.interpolate as inpl
import os
import sys
import cPickle

#%% load data
eeg_folder = 'eegs/'

t_diff = 20  # in sec - for taking eyes_open and closed

filenames = os.listdir(eeg_folder)
filenames = sorted(filenames)

for i_file, filename in enumerate(filenames):
    print('#############################')
    print('Processing file %d/%d: ' % (i_file+1, len(filenames)), filename)
    print('#############################')
    sys.stdout.flush()
    
    eegs_orig, fs, sens, t, ann = edfplus.load_edf(eeg_folder + filename)
    t_mid = t[-1]/2
    open_eyes = np.logical_and(t > t_mid+t_diff, t < t[-1]-t_diff)
    closed_eyes = np.logical_and(t < t_mid-t_diff, t > t_diff)
    eegs_len = eegs_orig.shape

#%% filter data
    print('Filtering ... ', end='')
    sys.stdout.flush()     
    f_l = .5
    f_h = 30
    # highpass > 0.5 Hz
    order = 5
    b, a = sig.iirfilter(order, f_l / (fs/2),
                         btype='high', 
                         ftype='butter')
    eegs = sig.lfilter(b, a, eegs_orig, axis=1)
    
    # lowpass < 48 Hz
    order = 5
    b, a = sig.iirfilter(order, f_h / (fs/2),
                         btype='low', 
                         ftype='butter')
    eegs = sig.lfilter(b, a, eegs, axis=1)
    
    ## plot filter                     
    #w, h = sig.freqz(b,a)
    #f_filt = w/pi * fs/2
    #plt.figure()
    #plt.plot(f_filt, 20*np.log10(np.abs(h)))
    #plt.grid()
    print('done.')

#%% delete chanells and montage
    print('Montage ... ', end='')
    sys.stdout.flush()
    # take away ECG and Fpz and Oz
    for i in range(len(sens))[::-1]:
        if sens[i] in "ECGFpzOzLABEL":
            eegs = np.delete(eegs, (i), axis=0)
            del sens[i]

    # Montage
    eegs_mean = np.mean(eegs,0)
    eegs = eegs - eegs_mean
    
    # take away AA 
    for i in range(len(sens))[::-1]:
        if sens[i] in "AA":
            eegs = np.delete(eegs, (i), axis=0)
            del sens[i]

    eeg_no_ch = eegs.shape[0]
    print('done.')
    #%% separete open and closed eyes
    print('Extracting open and closed eyes... ', end='')
    sys.stdout.flush()    
    eegs_open = eegs[:,open_eyes]
    t_open = t[open_eyes]
    eegs_closed = eegs[:,closed_eyes]
    t_closed = t[closed_eyes]
#%% plot time domain
    plt.figure(figsize=[16,10])
    plt.grid()
    for i, eeg in enumerate(eegs):
        plt.plot(t,eeg + (eeg_no_ch - 1 - i)*50)
    plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
    plt.axis([0, t[-1],-50, eeg_no_ch*50])
    plt.tight_layout()
    plt.legend(sens, fontsize=12)
    from matplotlib.patches import Rectangle
    ca = plt.gca()
    ca.add_patch(Rectangle((t_open[0], -50), t_open[-1] - t_open[0], 
                           eeg_no_ch*50+50, facecolor="red", alpha=0.2))
    ca.add_patch(Rectangle((t_closed[0], -50), t_closed[-1] - t_closed[0], 
                           eeg_no_ch*50+50, facecolor="blue", alpha=0.2))
    plt.xlabel('Time [s]')
    plt.ylabel(r'Amp [$\mu$V]')
    plt.tight_layout()
    
    print('done.')
    plt.show()

#%% ICA Artifact rejection
    print('ICA blind source separation ... ', end='')
    sys.stdout.flush()
    ica = FastICA(n_components=eeg_no_ch,max_iter=1000)
    eegs_ica_sig = ica.fit_transform(eegs_open.T)
    eegs_ica_mix = ica.mixing_

    # check decomposition
    recon = np.dot(eegs_ica_sig, eegs_ica_mix.T) + ica.mean_
    try:
        assert np.allclose(eegs_open.T, recon)
    except:
        print('WARNING: Assertion failed!')
            
# a0 b0 c0 d0     mix_ya0 mix_ya1 mix_ya2 mix_ya3 '   ya0 yb0 yc0 yd0
# a1 b1 c1 d1  X  mix_yb0 mix_yb1 mix_yb2 mix_yb3   = ya1 yb1 yc1 yd1
# a2 b2 c2 d2     mix_yc0 mix_yc1 mix_yc2 mix_yc3     ya2 yb2 yc2 yd2
#                 mix_yd0 mix_yd1 mix_yd2 mix_yd3

    ica_max = np.max(eegs_ica_sig, 0)
    ica_max_sort_ind = np.argsort(ica_max)[::-1]  # from largest to smallest
    print('done.')
    
#%% plot ICA
    plt.figure(figsize=[16,10])
    plt.grid()
    for i, eeg in enumerate(eegs_open):
        plt.plot(t_open,eeg + (eeg_no_ch - 1 - i)*50)
    plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
    plt.axis([t_open[0], t_open[-1],-50, eeg_no_ch*50])
#    plt.axis([300, 350,-50, eeg_no_ch*50])
    plt.tight_layout()
    plt.legend(sens, fontsize=12)
    plt.xlabel('Time [s]')
    plt.ylabel(r'Amp [$\mu$V]')
    plt.tight_layout()
        
    # time domain
    plt.figure(figsize=[16,10])
    plt.grid()
    plots = []
    for i, max_ind in enumerate(ica_max_sort_ind):
        eeg = eegs_ica_sig.T[max_ind,:]
        plot, = plt.plot(t_open, eeg*1000 + (eeg_no_ch - 1 - i)*50, label='ICA %d' % max_ind)
        plots.append(plot)
    plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
    plt.legend(handles = plots, fontsize=12)
    plt.axis([t_open[0], t_open[-1], -50, eeg_no_ch*50])
#    plt.axis([300, 350,-50, eeg_no_ch*50])
    plt.xlabel('Time [s]')
    plt.ylabel(r'Amp [$\mu$V]')
    plt.tight_layout()

#%%  plot ICA topographies
    eeg_1020 = img.imread('21ch_eeg.png')
    grid_x, grid_y = np.mgrid[0:eeg_1020.shape[0],
                              0:eeg_1020.shape[1]]
    
    el_centers_dict = {"Cz": (197,181), "C3" : (134,181),
    "C4" : (261,181), "T3" : (70,181), "T4" : (324,181),
    "Fz" : (197,117), "F3" : (146,116), "F4" : (250,116),
    "F7" : (95,107), "F8" : (300,107), "Fp1" : (156,61),
    "Fpz" : (197,53), "Fp2" : (239,61), "O1" : (157,301), 
    "Oz" : (197,308), "O2" : (238,301), "Pz" : (197,245),
    "P3" : (146,245), "P4" : (250,245), "T5" : (95,255),
    "T6" : (300,255)} 
    
    el_centers = [el_centers_dict[item] for item in sens]
    el_centers = np.fliplr(el_centers)
    
    plt.figure(figsize=[12,12])
    for i, max_ind in enumerate(ica_max_sort_ind):
        values = eegs_ica_mix.T[max_ind, :]
        topograph_t0 = inpl.griddata(el_centers, np.abs(values), 
                                 (grid_x, grid_y),
                                method='cubic')
        plt.subplot(4,5,i+1)
        plt.imshow(topograph_t0)
        plt.imshow(eeg_1020)
        plt.title('ICA %d' % max_ind, fontsize=12)
        plt.gca().set_xticklabels([''])
        plt.gca().set_yticklabels([''])
        
#%% remove blinking
    plt.show()
    ica_blink = 100
    while ica_blink is not None:    
        ica_blink = raw_input('Enter ICA component to remove [None]> ') or None
        if ica_blink is not None:
            ica_blink = [int(n) for n in ica_blink.split(",") ]
            eegs_ica_sig[:,ica_blink] = 0
            eegs_open_clean = np.dot(eegs_ica_sig, eegs_ica_mix.T) + ica.mean_
            eegs_open_clean = eegs_open_clean.T
        
            plt.figure(figsize=[16,10])
            plt.grid()
            for i, eeg in enumerate(eegs_open):
                plt.plot(t_open,eeg + (eeg_no_ch - 1 - i)*50)
            plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
            plt.axis([t_open[0], t_open[-1],-50, eeg_no_ch*50])
            plt.tight_layout()
            plt.legend(sens)
            plt.title('before')
            
            plt.figure(figsize=[16,10])
            plt.grid()
            for i, eeg in enumerate(eegs_open_clean):
                plt.plot(t_open,eeg + (eeg_no_ch - 1 - i)*50)
            plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
            plt.axis([t_open[0], t_open[-1],-50, eeg_no_ch*50])
#            plt.axis([300, 350,-50, eeg_no_ch*50])
            plt.tight_layout()
            plt.legend(sens, fontsize=12)
            plt.xlabel('Time [s]')
            plt.ylabel(r'Amp [$\mu$V]')
            plt.tight_layout()
            plt.title('after')
            
        #% save
            plt.show()
            ok = raw_input('Is the result ok? [y]/n >') or 'y'
            if ok is 'y':
                ica_blink = None
        else:
            eegs_open_clean = eegs_open

#%% Pickle cleaned eeg
    pickles_dir = 'pickles/'
    pickle_name = pickles_dir + \
                  os.path.basename(os.path.normpath(eeg_folder)) + \
                  os.path.splitext(filename)[0] + 'open.pickle'
    with open(pickle_name.replace(' ',''), "wb") as f:
        cPickle.dump(eegs_open_clean, f, 2)

#%% same for closed eyes
    print('ICA closed eye blind source separation ... ', end='')
    sys.stdout.flush()
    ica = FastICA(n_components=eeg_no_ch,max_iter=1000)
    eegs_ica_sig = ica.fit_transform(eegs_closed.T)
    eegs_ica_mix = ica.mixing_

    # check decomposition
    recon = np.dot(eegs_ica_sig, eegs_ica_mix.T) + ica.mean_
    try:
        assert np.allclose(eegs_closed.T, recon)
    except:
        print('WARNING: Assertion failed!')
            
# a0 b0 c0 d0     mix_ya0 mix_ya1 mix_ya2 mix_ya3 '   ya0 yb0 yc0 yd0
# a1 b1 c1 d1  X  mix_yb0 mix_yb1 mix_yb2 mix_yb3   = ya1 yb1 yc1 yd1
# a2 b2 c2 d2     mix_yc0 mix_yc1 mix_yc2 mix_yc3     ya2 yb2 yc2 yd2
#                 mix_yd0 mix_yd1 mix_yd2 mix_yd3

    ica_max = np.max(eegs_ica_sig, 0)
    ica_max_sort_ind = np.argsort(ica_max)[::-1]  # form largest to smallest
    print('done.')

#%% plot ICA
    plt.figure(figsize=[16,10])
    plt.grid()
    for i, eeg in enumerate(eegs_closed):
        plt.plot(t_closed,eeg + (eeg_no_ch - 1 - i)*50)
    plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
    plt.axis([t_closed[0], t_closed[-1],-50, eeg_no_ch*50])
    plt.tight_layout()
    plt.legend(sens)
    plt.title('before')

    plt.figure(figsize=[16,10])
    plt.grid()
    plots = []
    for i, max_ind in enumerate(ica_max_sort_ind):
        eeg = eegs_ica_sig.T[max_ind,:]
        plot, = plt.plot(eeg*1000 + (eeg_no_ch - 1 - i)*50, label='ICA %d' % max_ind)
        plots.append(plot)
    plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
    plt.legend(handles = plots)
    plt.axis([0, eeg.shape[0], -50, eeg_no_ch*50])
    plt.tight_layout()

#%%  plot ICA topographies
    eeg_1020 = img.imread('21ch_eeg.png')
    grid_x, grid_y = np.mgrid[0:eeg_1020.shape[0],
                              0:eeg_1020.shape[1]]
    
    el_centers_dict = {"Cz": (197,181), "C3" : (134,181),
    "C4" : (261,181), "T3" : (70,181), "T4" : (324,181),
    "Fz" : (197,117), "F3" : (146,116), "F4" : (250,116),
    "F7" : (95,107), "F8" : (300,107), "Fp1" : (156,61),
    "Fpz" : (197,53), "Fp2" : (239,61), "O1" : (157,301), 
    "Oz" : (197,308), "O2" : (238,301), "Pz" : (197,245),
    "P3" : (146,245), "P4" : (250,245), "T5" : (95,255),
    "T6" : (300,255)} 
    
    el_centers = [el_centers_dict[item] for item in sens]
    el_centers = np.fliplr(el_centers)
        
    plt.figure(figsize=[12,12])
    for i, max_ind in enumerate(ica_max_sort_ind):
        values = eegs_ica_mix.T[max_ind, :]
        topograph_t0 = inpl.griddata(el_centers, np.abs(values), 
                                 (grid_x, grid_y),
                                method='cubic')
        plt.subplot(4,5,i+1)
        plt.imshow(topograph_t0)
        plt.imshow(eeg_1020)
        plt.title('ICA %d' % max_ind, fontsize=12)
        plt.gca().set_xticklabels([''])
        plt.gca().set_yticklabels([''])

#%% remove blinking
    plt.show()
    ica_blink = 100
    while ica_blink is not None:    
        ica_blink = raw_input('Enter ICA component to remove [None]> ') or None
        if ica_blink is not None:
            ica_blink = [int(n) for n in ica_blink.split(",") ]
            eegs_ica_sig[:,ica_blink] = 0
            eegs_closed_clean = np.dot(eegs_ica_sig, eegs_ica_mix.T) + ica.mean_
            eegs_closed_clean = eegs_closed_clean.T
            
        #%% plot clean
            plt.figure(figsize=[16,10])
            plt.grid()
            for i, eeg in enumerate(eegs_closed):
                plt.plot(t_closed,eeg + (eeg_no_ch - 1 - i)*50)
            plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
            plt.axis([t_closed[0], t_closed[-1],-50, eeg_no_ch*50])
            plt.tight_layout()
            plt.legend(sens)
            plt.title('before')
            
            plt.figure(figsize=[16,10])
            plt.grid()
            for i, eeg in enumerate(eegs_closed_clean):
                plt.plot(t_closed,eeg + (eeg_no_ch - 1 - i)*50)
            plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
            plt.axis([t_closed[0], t_closed[-1],-50, eeg_no_ch*50])
            plt.tight_layout()
            plt.legend(sens)
            plt.title('after')
            
        #%% save
            plt.show()
            ok = raw_input('Is the result ok? [y]/n >') or 'y'
            if ok is 'y':
                ica_blink = None
        else:
            eegs_closed_clean = eegs_closed

#%% Pickle it        
    pickles_dir = 'pickles/'
    pickle_name = pickles_dir + \
                  os.path.basename(os.path.normpath(eeg_folder)) + \
                  os.path.splitext(filename)[0] + 'closed.pickle'
    with open(pickle_name.replace(' ',''), "wb") as f:
        cPickle.dump(eegs_closed_clean, f, 2)

