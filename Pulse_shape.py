# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""

import numpy as np
from scipy import signal
# Filter
from Filter import Butter_bandpass_filter

def f_omega(time, omega, t_imp, x2, hpf, lpf, order):
    
    # Filter (bandwidth of the interferometer)
    dt = np.abs(time[1]-time[0]) # Time step (µs)
    sf_bpf, b_bpf, a_bpf = Butter_bandpass_filter(np.arange(0,10,0.1), hpf, lpf, 1/dt, order) # Bandpass
    w_bpf, bpf_det = signal.freqz(b_bpf, a_bpf, worN = len(omega[:,0]), whole=False) # whole=True if np.fft.fft
    bpf_det = (bpf_det*np.ones((len(x2),1))).T    

    # Gaussian pulse in the Fourier domain
    f_omega = 1./np.sqrt(2*np.pi)*np.exp(-(omega*t_imp)**2/(16*np.log(2)))* \
                np.exp(1j*omega*(time[0]))*bpf_det
                    
    return f_omega

def g_k2(x1, x2, k2, a_s, a_d, theta, x1_shift, x2_shift):
    
    # Filter (gaussian spot of the interferometer)
    interferometer = np.exp(-(k2*a_d)**2/(16*np.log(2))) 
    # Gaussian pulse in the Fourier domain
    g_k2 = 1./(np.sqrt(2*np.pi)*np.cos(theta))* \
            np.exp(-(k2/np.cos(theta)*a_s)**2/(16*np.log(2)))* \
            np.exp(-1j*k2*np.tan(theta)*x1_shift)*np.exp(1j*k2*x2_shift)*interferometer

    return g_k2
