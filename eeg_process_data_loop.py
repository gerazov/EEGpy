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
Loading of EEG data, finding coherence.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import scipy.interpolate as inpl
import os
import cPickle
from scipy import fftpack as fp 
import re

#%% init
sens = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
        'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
fs = 250
eeg_no_ch = len(sens)
coherence_master = np.zeros((2,30,2,eeg_no_ch,eeg_no_ch, 5))
# these are: groups, people, eyes, ..., bands
#%% main file loop
pickle_folder = 'pickles/'
filenames = os.listdir(pickle_folder)
filenames = sorted(filenames)
#filename = 'EEGVozrasniADHD01closed.pickle'

for i_file, filename in enumerate(filenames):
#    print('#############################')
    print('Processing file %d/%d: ' % (i_file+1, len(filenames)), filename)
#    print('#############################')
    person_no = int(re.findall(r'\d+', filename)[0]) - 1
#    print(person_no)
    if "open" in filename:
        eyes = 1
    else:
        eyes = 0
    if "Norm" in filename:
        group = 0
    else:
        group = 1
    
    with open(pickle_folder + filename, "rb") as f:
        eegs = cPickle.load(f)
    
    eeg_len = eegs.shape[1]
    t = np.arange(0,eeg_len/fs, 1/fs)
    
    
    #%% detect artifacts
    artifact_mask = np.ones(eeg_len)
    
    # 1. Aplitude of raw signal is above 100 uV
    artifact_mask_100uv = artifact_mask * np.any(np.abs(eegs) > 100, 0)
    
    # 2. Amplitude of slow waves (<1Hz) are > 50uV
    f_l = 1
    order = 5
    b, a = sig.iirfilter(order, f_l / (fs/2),
                         btype='low', 
                         ftype='butter')
                         
    eegs_1hz = sig.lfilter(b, a, eegs, axis=1)
    artifact_mask_1hz = artifact_mask * \
                        np.any(np.abs(eegs_1hz) > 50, 0)
    
    # 3. High frequency activity (20-35Hz) is > 35uV
    f_l = 20
    f_h = 35
    order = 9
    b, a = sig.iirfilter(order, np.array([f_l, f_h]) / (fs/2),
                         btype='bandpass', 
                         ftype='butter')
    
    eegs_20_35hz = sig.lfilter(b, a, eegs, axis=1)
    artifact_mask_20_35hz = artifact_mask * \
                            np.any(np.abs(eegs_20_35hz) > 35, 0)
    
    ## plot filter                     
    #w, h = sig.freqz(b,a)
    #f_filt = w/pi * fs/2
    #plt.figure()
    #plt.plot(f_filt, 20*np.log10(np.abs(h)))
    #plt.grid()
    
    # all artifacts detected
    artifact_mask = artifact_mask_100uv + artifact_mask_1hz + artifact_mask_20_35hz
    artifact_mask[artifact_mask > 1] = 1
    
    #%% dilatate mask
    dil_t = 2  # sec
    dil_smpl = round(dil_t/2 * fs)
    b = np.ones(dil_smpl) 
    
    artifact_mask_dil = sig.filtfilt(b, 1, artifact_mask)
    artifact_mask_dil[artifact_mask_dil>1] = 1
    
    #plt.figure()
    #plt.plot(artifact_mask)
    #plt.plot(artifact_mask_dil)
    
    #%% plot artifacts
#    plt.figure(figsize=[15,10])
#    plt.grid()
#    for i, eeg in enumerate(eegs):
#        plt.plot(t,eeg + (eeg_no_ch - 1 - i)*50)
#    plt.yticks(np.arange(-50, eeg_no_ch*50, 50))
#    plt.axis([t[0], t[-1],-50, eeg_no_ch*50])
#    plt.tight_layout()
#    plt.legend(sens)
#   
##    # plot artifacts
#    plt.plot(t, artifact_mask_100uv*eeg_no_ch*50, 'r', linewidth=5, alpha=0.5)
#    plt.plot(t, artifact_mask_1hz*eeg_no_ch*50, 'b', linewidth=5, alpha=0.5)
#    plt.plot(t, artifact_mask_20_35hz*eeg_no_ch*50, 'g', linewidth=5, alpha=0.5)
#    plt.plot(t, artifact_mask_dil*eeg_no_ch*50, 'k', linewidth=10, alpha=0.5)
#    plt.show()

    
    #%% windowing
    epoch_t = 4  # sec
    epoch = int(np.round(epoch_t * fs))
    epoch_h = int(epoch/2)
    hop = epoch_h   # hop size
    wins = sig.get_window('hann', epoch) * np.ones((eeg_no_ch,epoch))
    poz = epoch_h  # position of middle of window 
    pad = np.zeros((eeg_no_ch, epoch_h))
    eegs_pad = np.hstack((pad, eegs, pad))
    while poz < eegs_pad.shape[1] - epoch_h:
        frames = eegs_pad[:,poz-epoch_h : poz+epoch_h] * wins
    #    f_frame, frame_spec = das.get_spectrum(frame, fs)
        if poz == epoch_h:
            eegs_frames = np.expand_dims(frames, axis=2) 
        else:
            eegs_frames = np.dstack((eegs_frames, 
                                    np.expand_dims(frames, axis=2)))
    #                      np.array([frame_spec]).T))
        poz += hop
    
    no_frames = eegs_frames.shape[2]
    
    #%% remove frames with artifacts
    t_frames = np.arange(no_frames) * hop /fs
    artifact_mask_frames = inpl.interp1d(t, artifact_mask_dil, kind='nearest')(t_frames)
    #plt.figure()
    #plt.plot(t, artifact_mask_dil)
    #plt.plot(t_frames, artifact_mask_frames+0.5)
    eegs_frames_clean = np.delete(eegs_frames,
                                  np.where(artifact_mask_frames), axis=2)
    
    #%% calculate spectres
    nfft = int(2**np.ceil(np.log(epoch)/np.log(2)))
    nfft2 = int(nfft/2)
    eegs_frames_fft = fp.fft(eegs_frames_clean, nfft, axis=1)
    eegs_frames_fft = eegs_frames_fft[:,:nfft2+1,:] / epoch *2
#    eegs_frames_fft = np.abs(eegs_frames_fft)**2
    f = np.linspace(0, fs/2, nfft2+1)
    
    #%% test spectrogram
    plt.figure()
    plt.imshow(20*np.log10(np.abs(eegs_frames_fft[15,:,:])), aspect='auto', \
               origin='lower', \
               extent=[0, t_frames[-1], 0, f[-1]],\
               vmin=-30,vmax=0, cmap='viridis')
    plt.axis([0, t_frames[-1], 0, 30])           
    cbar = plt.colorbar()
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    cbar.ax.set_ylabel('Amplitude [dB]')
    plt.show()
    
    #%% calculate coherence
    no_pairs = eeg_no_ch * (eeg_no_ch-1) / 2
    coherence = np.zeros((eeg_no_ch, eeg_no_ch, f.size))
    
    #for i in range(eegs.shape[0]):     
    for i in range(eeg_no_ch): 
        for j in range(eeg_no_ch): 
            i_fft = eegs_frames_fft[i,:,:]
            j_fft = eegs_frames_fft[j,:,:]
            i_fft_sum = np.sum(np.abs(i_fft)**2, axis=1)
            j_fft_sum = np.sum(np.abs(j_fft)**2, axis=1)
            ij_fft = i_fft * np.conj(j_fft)
            ij_fft_sum = np.sum(ij_fft, axis=1) 
            ij_fft_sum = np.abs(ij_fft_sum)**2
            coherence[i,j,:] = ij_fft_sum / (i_fft_sum * j_fft_sum)
    
    #%% plot 
#    plt.figure()
#    plt.plot(f, coherence[2,1,:])    
#    plt.show()
    
    #%% calculate coherence in a given band 
    fbands = [[0,4], [4,8], [8,15], [15,31]]
    bands = ['delta','theta','alpha','beta']
    no_bands = len(bands)
    coherence_bands = np.zeros((eeg_no_ch, eeg_no_ch, no_bands+1))
    
    for i in range(no_bands):
        fstart = np.where(f >= fbands[i][0])[0][0]
        fstop = np.where(f < fbands[i][1])[0][-1]
        coherence_bands[:,:,i] = np.max(coherence[:,:,fstart:fstop], axis=2)
    
    coherence_bands[:,:,no_bands] = np.max(coherence_bands[:,:,:no_bands])
    # %% plot coherence
    plt.figure()
    for i in range(no_bands):
        plt.subplot(2,2,i+1)
        plt.imshow(coherence_bands[:,:,i], aspect='auto', \
                   origin='upper', \
                   extent=[0, eeg_no_ch, 0, eeg_no_ch],\
                   cmap='viridis')
    plt.show()
    #           vmin=0,vmax=1, cmap='viridis')
    #plt.axis([0, t_frames[-1], 0, 30])           
    #cbar = plt.colorbar()
    #plt.xlabel('time [s]')
    #plt.ylabel('frequency [Hz]')
    #cbar.ax.set_ylabel('Amplitude [dB]')

#%% consolidate master coherence
    coherence_master[group, person_no, eyes, :, :, :]  = coherence_bands

#%% save
with open('coherence_master.pickle', "wb") as f:
    cPickle.dump(coherence_master, f, 2)
    