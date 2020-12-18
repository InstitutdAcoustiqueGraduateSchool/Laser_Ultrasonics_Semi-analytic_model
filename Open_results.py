# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# Filter
from Filter import Butter_bandpass_filter

plt.rcParams.update({'font.size':24})
plt.rcParams.update({'figure.autolayout': True})
plt.close('all')

def Open_results(path_simu, id_save, x1_plot, pow_kn1, pow_kt1, dt, dx2, hpf, lpf, order):
    
    #%% Open data (Temporal domain)

    data = np.load(path_simu+id_save+'_kn'+str(pow_kn1)+'_kt'+str(pow_kt1)+ '.npz', 'r')
    # Read '.npz' file
    time_data = data['time']
    x2_data = data['position']
    front_data = data['front']
    t_norm = 0.25 # (µs)   
    idt_norm = np.argmin(np.abs(time_data-t_norm))
    max_norm = np.where(np.max(front_data[idt_norm::,:])!=0., np.max(front_data[idt_norm::,:]), 1.)
    
    # Plot B-scan 
    plt.figure()
    plt.title('Simulated B-scan')
    plt.imshow(front_data*1e6, origin='lower',
                   extent=[x2_data[0],x2_data[-1],time_data[0],time_data[-1]],
                   cmap='seismic',aspect='auto', vmin=-max_norm*1e6, vmax=max_norm*1e6)
    plt.xlabel('Position (mm)')
    plt.ylabel('Time ($\mu$s)')
    cbar = plt.colorbar()
    cbar.set_label('$u_1$ (nm)')
    plt.savefig('b-scan.svg')
    
    # Plot Temporal signal at x2=0
    x2_pos = 0.
    ind_x2_data = np.argmin(np.abs(x2_data-x2_pos))
    
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(time_data, front_data[:,ind_x2_data]*1e6,'r')
    plt.title('Temporal sig. at $x_2=0$')
    plt.ylim([-max_norm*1e6, max_norm*1e6])
    plt.xlabel('Time ($\mu$s)')
    plt.ylabel('$u_1$ (nm)')

    # Plot FFT signal at x2=0
    freq_data = np.fft.fftfreq(len(time_data), dt) # (MHz)
    id_fmin = np.argmin(np.abs(freq_data-0.)) 
    id_fmax = np.argmin(np.abs(freq_data-20.))
    hann_win = np.hanning(len(time_data)) # Hann window
    FFT_data = np.abs(np.fft.fft(front_data[:,ind_x2_data]*hann_win))
    max_FFT_data = np.max(FFT_data[id_fmin:id_fmax+1])
    FFT_data_norm = FFT_data/max_FFT_data

    fig, ax1 = plt.subplots(1,1)
    ax1.plot(freq_data[id_fmin:id_fmax+1], FFT_data_norm[id_fmin:id_fmax+1], 'r')
    plt.title('FFT signal at $x_2=0$')
    plt.xlabel('Freq. (MHz)')
    plt.ylabel('Norm. Amp.')
    
    #%% Open data (Fourier domain)
    data_F = np.load(path_simu+id_save+'_Fourier_kn'+str(pow_kn1)+'_kt'+str(pow_kt1)+ '.npz', 'r')
    # Read '.npz' file
    omega_data_F = data_F['omega'][:,0]
    k2_data_F = data_F['k2'][0,:]
    front_data_F = data_F['front']
    
    freq_dc = np.real(omega_data_F[1::])/2/np.pi # Frequency (MHz)
    k2_dc = np.fft.fftshift(k2_data_F) # Wavenumber (rad.mm-1)
    disperion_curves = np.abs(np.fft.fftshift(front_data_F[1::,:],axes=1))**2
    disperion_curves_dB = 20*np.log10(disperion_curves/np.max(disperion_curves))
    
    id_fmin_dc = np.argmin(np.abs(freq_dc-0.)) # MHz
    id_fmax_dc = np.argmin(np.abs(freq_dc-10.)) # MHz
    id_k2min_dc = np.argmin(np.abs(k2_dc+10.)) # rad/mm
    id_k2max_dc = np.argmin(np.abs(k2_dc-10.)) # rad/mm
    
    plt.figure()
    plt.title('$f-k$ diagram')
    plt.imshow(disperion_curves_dB[id_fmin_dc:id_fmax_dc+1, id_k2min_dc:id_k2max_dc+1], origin='lower',
                   extent=[k2_dc[id_k2min_dc], k2_dc[id_k2max_dc], 
                           freq_dc[id_fmin_dc], freq_dc[id_fmax_dc]],
                   aspect='auto', vmin=-150., vmax=-3.)
    plt.xlabel('$k_2$ (rad.mm$^{-1}$)')
    plt.ylabel('Freq. (MHz)')
    cbar = plt.colorbar()
    cbar.set_label('Norm. $|\hat{u}_1|^2$ (dB)')
    plt.savefig('f-k.svg')
