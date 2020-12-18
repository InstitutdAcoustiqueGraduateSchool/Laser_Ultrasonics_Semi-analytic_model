# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""
import numpy as np
# Pulse shape (in the Fourier domain)
from Pulse_shape import f_omega, g_k2

def Coef_EM(medium, dh, n0, mu_0, theta_0, kopt_0, n1, mu_1, kopt_1, n2, mu_2, 
            kopt_2, EMc):
    
    #%% Thicknesses 
    h_1 = medium[0][5] # Medium 1
    h_2 = medium[1][5] # Medium 2
    H = h_1+h_2+dh # Total thickness
    #%% theta_1, theta_2
    theta_1 = np.arcsin(kopt_0/kopt_1*np.sin(theta_0))
    theta_2 = np.arcsin(kopt_0/kopt_2*np.sin(theta_0))
    #%% Beta_1, beta_2
    beta_1 = 2*(np.real(kopt_1)*np.imag(np.cos(theta_1))+ \
                np.imag(kopt_1)*np.real(np.cos(theta_1)))
    beta_2 = 2*(np.real(kopt_2)*np.imag(np.cos(theta_2))+ \
                np.imag(kopt_2)*np.real(np.cos(theta_2)))
    #%% gamma_1, gamma_2 
    gamma_1 = (np.real(kopt_1)*np.real(np.cos(theta_1))- \
               np.imag(kopt_1)*np.imag(np.cos(theta_1)))
    gamma_2 = (np.real(kopt_2)*np.real(np.cos(theta_2))- \
               np.imag(kopt_2)*np.imag(np.cos(theta_2)))
    #%% Electromagnetic boundary conditions between medium 1 and medium 2
    L11, L12 = EMc[0,:] # Electromagnetic coupling matrix
    L21, L22 = EMc[1,:] # Electromagnetic coupling matrix
    #%% R_0, T_m1, R_m1, T_m2, R_m2, T_H
    n3=n0; mu_3=mu_0; kopt_3=kopt_0; theta_3=theta_0 # assumption that medium 0 and medium III have similar electromagnetic properties
    A = np.array([[np.cos(theta_0), np.cos(theta_1), -np.cos(theta_1)*np.exp(-beta_1*h_1/2), 0, 0, 0], 
                  [n0/mu_0, -n1/mu_1, -n1/mu_1*np.exp(-beta_1*h_1/2), 0, 0, 0],
                  [0, -np.cos(theta_1)*np.exp(1j*gamma_1*h_1)*np.exp(-beta_1*h_1/2), 
                   np.cos(theta_1)*np.exp(-1j*gamma_1*h_1),
                   (L11*np.cos(theta_2)+L12*n2/mu_2)*np.exp(1j*gamma_2*(h_1+dh)),
                   (-L11*np.cos(theta_2)+L12*n2/mu_2)*np.exp(-1j*gamma_2*(h_1+dh))*np.exp(-beta_2*h_2/2), 0],
                  [0, -n1/mu_1*np.exp(1j*gamma_1*h_1)*np.exp(-beta_1*h_1/2), -n1/mu_1*np.exp(-1j*gamma_1*h_1),
                   (L21*np.cos(theta_2)+L22*n2/mu_2)*np.exp(1j*gamma_2*(h_1+dh)),
                   (-L21*np.cos(theta_2)+L22*n2/mu_2)*np.exp(-1j*gamma_2*(h_1+dh))*np.exp(-beta_2*h_2/2), 0],
                  [0, 0, 0, -np.cos(theta_2)*np.exp(1j*gamma_2*H)*np.exp(-beta_2*h_2/2),
                   np.cos(theta_2)*np.exp(-1j*gamma_2*H), np.cos(theta_3)*np.exp(1j*kopt_3*np.cos(theta_3)*H)],
                  [0, 0, 0, n2/mu_2*np.exp(1j*gamma_2*H)*np.exp(-beta_2*h_2/2), 
                   n2/mu_2*np.exp(-1j*gamma_2*H), -n3/mu_3*np.exp(1j*kopt_3*np.cos(theta_3)*H)]])
    B = np.array([[np.cos(theta_0)], [-n0], [0], [0], [0], [0]], dtype=complex)
    Rb_0, Rf_m1, Rb_m1, Rf_m2, Rb_m2, Rf_H = np.linalg.solve(A, B)
    
    return beta_1, beta_2, theta_1, theta_2, Rf_m1, Rb_m1, Rf_m2, Rb_m2

def Coupling_EM(theta_0, kopt_0, n_i, mu_i, h_i):
    
    kopt_i = kopt_0*n_i # optical wavenumber (complex value)
    theta_i = np.arcsin(kopt_0/kopt_i*np.sin(theta_0))
    gamma_i = kopt_i*np.cos(theta_i)*h_i
    EMc_i = np.array([[np.cos(gamma_i), -1j*np.sin(gamma_i)*np.cos(theta_i)*mu_i/n_i],
                      [-1j*np.sin(gamma_i)*n_i/(mu_i*np.cos(theta_i)), np.cos(gamma_i)]])
    
    return EMc_i

def Power_densities_Q(I0, beta_1, beta_2, mu_1, mu_2, n1, n2, theta_1, theta_2,
                      Rf_m1, Rb_m1, Rf_m2, Rb_m2, time, omega, k2, t_imp, x1, x2, 
                      hpf, lpf, order, a_s, a_d, h_1, h_2, dh):
    
    H = h_1+h_2+dh # Total thickness (mm)
    # Medium 1
    Qf_m1 = I0*beta_1/mu_1*np.real(np.conj(n1)*np.cos(theta_1))*np.abs(Rf_m1)**2* \
        f_omega(time, omega, t_imp, x2, hpf, lpf, order)* \
        g_k2(x1, x2, k2, a_s, a_d, np.real(theta_1), 0, 0)
    Qb_m1 = I0*beta_1/mu_1*np.real(np.conj(n1)*np.cos(theta_1))*np.abs(Rb_m1)**2* \
            f_omega(time, omega, t_imp, x2, hpf, lpf, order)*\
            g_k2(x1, x2, k2, a_s, a_d, -np.real(theta_1), h_1, h_1*np.tan(np.real(theta_1)))
    # Medium 2
    Qf_m2 = I0*beta_2/mu_2*np.real(np.conj(n2)*np.cos(theta_2))*np.abs(Rf_m2)**2* \
            f_omega(time, omega, t_imp, x2, hpf, lpf, order)*\
            g_k2(x1, x2, k2, a_s, a_d, np.real(theta_2), h_1+dh, h_1*np.tan(np.real(theta_1)))
    Qb_m2 = I0*beta_2/mu_2*np.real(np.conj(n2)*np.cos(theta_2))*np.abs(Rb_m2)**2* \
            f_omega(time, omega, t_imp, x2, hpf, lpf, order)*\
            g_k2(x1, x2, k2, a_s, a_d, -np.real(theta_2), H, h_1*np.tan(np.real(theta_1))+h_2*np.tan(np.real(theta_2)))
    
    Q = np.array([Qf_m1, Qb_m1, Qf_m2, Qb_m2])
    
    return Q
