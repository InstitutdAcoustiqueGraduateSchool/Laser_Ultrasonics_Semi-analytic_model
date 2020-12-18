# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""

import numpy as np

def Omega_k2(time, x2):
    
    dt = np.abs(time[1]-time[0]) # Step t (µs)
    dx2 = np.abs(x2[1]-x2[0]) # Step x2 (mm)
    # Shift in the imaginary plane with a constant delta (R. L. Weaver et al., "Transient ultrasonic waves in a viscoelastic plate: Applications to materials characterization," J. Acoust. Soc. Am. 85(6), 2262-2267, 1989.)
    delta = 3*np.log(10)/time[-1] # Criteria based on the article of E. Kausel and J. M. Roesset, "Frequency domain analysis of undamped systems", J. Eng. Mech. 118:721-734, 1992.
    # Angular frequency (omega)
    omega = 2*np.pi*(np.fft.rfftfreq(len(time), dt)*np.ones((len(x2),1))).T-1j*delta # (rad/µs) Matrix [len(omega),len(k2)]
    omega[0,:] = np.finfo(np.float64).eps-1j*delta # Machine limits for floating point np.float64
    # Wavenumber (k2)
    k2 = 2*np.pi*np.fft.fftfreq(len(x2), dx2)*np.ones((len(omega[:,0]),1)) # (rad/mm) Matrix [len(omega),len(k2)]
    k2[:,0] = np.finfo(np.float64).eps # Machine limits for floating point np.float64
    
    return omega, k2, delta