# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""
from scipy import signal

# Highpass filter
def Butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='highpass', output='ba')
    y = signal.filtfilt(b, a, data, axis=0, padtype='odd', padlen=data.shape[0]-1, method='pad')
    return y, b, a

# Lowpass filter
def Butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', output='ba')
    y = signal.filtfilt(b, a, data, axis=0, padtype='odd', padlen=data.shape[0]-1, method='pad')
    return y, b, a

# Bandpass filter
def Butter_bandpass_filter(data, cutoff_low, cutoff_high, fs, order):
    nyq = 0.5*fs
    normal_cutoff_low = cutoff_low/nyq
    normal_cutoff_high = cutoff_high/nyq
    b, a = signal.butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', output='ba')
    y = signal.filtfilt(b, a, data, axis=0, padtype='odd', padlen=data.shape[0]-1, method='pad')
    return y, b, a