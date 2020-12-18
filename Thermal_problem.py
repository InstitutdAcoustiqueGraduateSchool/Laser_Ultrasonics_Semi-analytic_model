# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""
import numpy as np
import matplotlib.pyplot as plt

def Coefficients_tf_tb(omega, k2, medium, Rc, Tp, gamma_m1, lambda11_m1, 
                       gamma_m2, lambda11_m2, beta, h, T_ext, theta_1, theta_2, dh):
    
    # Elastic coefficients, density and thickness of the medium
    c11_1, c12_1, c22_1, c66_1, rho_1, h_1 = medium[0]
    c11_2, c12_2, c22_1, c66_1, rho_2, h_2 = medium[1]
    H = h_1+dh+h_2 # Total thickness (mm)
    beta_1, beta_2 = beta
    Tpf_beta_1, Tpb_beta_1, Tpf_beta_2, Tpb_beta_2 = Tp
    # a_t, b_t
    a_t = np.zeros((4,4, len(omega[:, 0]), len(k2[0, :])), complex)
    b_t = np.zeros((4,1, len(omega[:, 0]), len(k2[0, :])), complex)
    zero = np.zeros((len(omega[:, 0]), len(k2[0, :])), complex)
    one = np.ones((len(omega[:, 0]), len(k2[0, :])), complex)
    h = h*one # Heat transfer coefficient
    delta_omega = np.zeros((len(omega[:, 0]), len(k2[0, :])), complex)
    delta_omega[0,:] = np.ones((len(k2[0, :])), complex)
    delta_k2 = np.zeros((len(omega[:, 0]), len(k2[0, :])), complex)
    delta_k2[:,0] = np.ones((len(omega[:, 0])), complex)
    a_t = np.array([[-lambda11_m1*gamma_m1-h, (lambda11_m1*gamma_m1-h)*np.exp(-gamma_m1*h_1), zero, zero],
                    [-lambda11_m1*gamma_m1*np.exp(-gamma_m1*h_1), lambda11_m1*gamma_m1, 
                     lambda11_m2*gamma_m2, -np.exp(-gamma_m2*h_2)*lambda11_m2*gamma_m2],
                    [(-lambda11_m1*gamma_m1*Rc+one)*np.exp(-gamma_m1*h_1), lambda11_m1*gamma_m1*Rc+one, 
                     -one, -np.exp(-gamma_m2*h_2)],
                     [zero, zero, (-lambda11_m2*gamma_m2+h)*np.exp(-gamma_m2*h_2), lambda11_m2*gamma_m2+h]])
    b_t = np.array([[Tpf_beta_1*(lambda11_m1*(beta_1-1j*k2*np.tan(theta_1))+h)+Tpb_beta_1*(-lambda11_m1*(beta_1-1j*k2*np.tan(theta_1))+h)*np.exp(-beta_1*h_1)-h*T_ext*2*np.pi*delta_omega*delta_k2], 
                    [lambda11_m1*(beta_1-1j*k2*np.tan(theta_1))*(Tpf_beta_1*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)-Tpb_beta_1*np.exp(-1j*k2*np.tan(theta_1)*h_1))- 
                     lambda11_m2*(beta_2-1j*k2*np.tan(theta_2))*(Tpf_beta_2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))-Tpb_beta_2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh)))], 
                    [Tpf_beta_1*(lambda11_m1*Rc*(beta_1-1j*k2*np.tan(theta_1))-one)*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)+ 
                     Tpb_beta_1*(-lambda11_m1*Rc*(beta_1-1j*k2*np.tan(theta_1))-one)*np.exp(-1j*k2*np.tan(theta_1)*h_1)+
                     Tpf_beta_2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))+Tpb_beta_2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh))],
                    [Tpf_beta_2*(lambda11_m2*(beta_2-1j*k2*np.tan(theta_2))-h)*np.exp(-beta_2*h_2)*np.exp(1j*k2*np.tan(theta_2)*H)+
                     Tpb_beta_2*(-lambda11_m2*(beta_2-1j*k2*np.tan(theta_2))-h)*np.exp(-1j*k2*np.tan(theta_2)*H)+h*T_ext*2*np.pi*delta_omega*delta_k2]])
    #%% Solve np.linalg.solve
    A = np.swapaxes(np.swapaxes(a_t,0,2),1,3)
    B = np.swapaxes(np.swapaxes(b_t,0,2),1,3)  
    sol = np.linalg.solve(A, B)
    Thf_m1, Thb_m1_star, Thf_m2_star, Thb_m2_star = np.swapaxes(np.swapaxes(sol,0,2),1,3)[:,0,:,:]

    return Thf_m1, Thb_m1_star, Thf_m2_star, Thb_m2_star

def Coef_Temp(omega, k2, medium, Rc, beta, Q, h, T_ext, th_prop_m1, th_prop_m2,
              theta_1, theta_2, dh):

    #%% Medium 1
    beta_1 = beta[0]
    c11_1, c12_1, c22_1, c66_1, rho_1, h_1 = medium[0] 
    cp_1, lambda11_m1, lambda22_m1, C_alpha1_m1, C_alpha2_m1 = th_prop_m1    
    D11_m1 = lambda11_m1/(rho_1*cp_1)
    D22_m1 = lambda22_m1/(rho_1*cp_1)
    gamma_m1 = np.sqrt((1j*omega+D22_m1*k2**2)/D11_m1)   
    #%% Medium 2
    beta_2 = beta[1]
    c11_2, c12_2, c22_1, c66_1, rho_2, h_2 = medium[1]   
    cp_2, lambda11_m2, lambda22_m2, C_alpha1_m2, C_alpha2_m2 = th_prop_m2 
    D11_m2 = lambda11_m2/(rho_2*cp_2)
    D22_m2 = lambda22_m2/(rho_2*cp_2)  
    gamma_m2 = np.sqrt((1j*omega+D22_m2*k2**2)/D11_m2) 
    #%% Particular solution medium 1
    Qf_m1, Qb_m1 = Q[0:2]
    Tpf_beta_1 = -Qf_m1/(rho_1*cp_1*((beta_1-1j*k2*np.tan(theta_1))**2*D11_m1-(1j*omega+D22_m1*k2**2)))
    Tpb_beta_1 = -Qb_m1/(rho_1*cp_1*((beta_1-1j*k2*np.tan(theta_1))**2*D11_m1-(1j*omega+D22_m1*k2**2)))
    #%% Particular solution medium 2
    Qf_m2, Qb_m2 = Q[2::]
    Tpf_beta_2 = -Qf_m2/(rho_2*cp_2*((beta_2-1j*k2*np.tan(theta_2))**2*D11_m2-(1j*omega+D22_m2*k2**2)))
    Tpb_beta_2 = -Qb_m2/(rho_2*cp_2*((beta_2-1j*k2*np.tan(theta_2))**2*D11_m2-(1j*omega+D22_m2*k2**2)))
    #%% Coefficients tf_lm1, tb_lm1_star, tf_lm2, tb_lm2_star
    Tp = Tpf_beta_1, Tpb_beta_1, Tpf_beta_2, Tpb_beta_2
    Thf_m1, Thb_m1_star, \
    Thf_m2_star, Thb_m2_star = Coefficients_tf_tb(omega, k2, medium, Rc, Tp,
                                                  gamma_m1, lambda11_m1, 
                                                  gamma_m2, lambda11_m2,
                                                  beta, h, T_ext, theta_1, theta_2, dh)
    t_gamma = gamma_m1, gamma_m2
    Th = Thf_m1, Thb_m1_star, Thf_m2_star, Thb_m2_star 
    
    return Tp, t_gamma, Th

def Temp_Complete_solution(x1, k2, time, x2, delta, medium, dh, beta, Tp, 
                           t_gamma, Th, T_ext, theta_1, theta_2):
    
    # Elastic coefficients, density and thickness of the medium
    c11_1, c12_1, c22_1, c66_1, rho_1, h_1 = medium[0]
    c11_2, c12_2, c22_1, c66_1, rho_2, h_2 = medium[1]
    H = h_1+h_2+dh
    # beta, t_gamma, Tp, Th
    beta_1, beta_2 = beta
    Tpf_beta_1, Tpb_beta_1, Tpf_beta_2, Tpb_beta_2 = Tp
    gamma_m1, gamma_m2 = t_gamma 
    Thf_m1, Thb_m1_star, Thf_m2_star, Thb_m2_star = Th
    #%% Complete solution 
    # Medium 1
    if x1<=h_1: 
        Thc = Thf_m1*np.exp(-gamma_m1*x1)+Thb_m1_star*np.exp(gamma_m1*(x1-h_1)) # Homogeneous solution
        Tpc = Tpf_beta_1*np.exp(-beta_1*x1)*np.exp(1j*k2*np.tan(theta_1)*x1)+Tpb_beta_1*np.exp(beta_1*(x1-h_1))*np.exp(-1j*k2*np.tan(theta_1)*x1) # Particular solution
        Tc = Thc+Tpc # Complete solution
    # Sublayer(s)
    if x1>h_1 and x1<h_1+dh: 
        Tc = np.zeros((np.shape(Thf_m1)), complex) 
    # Medium 2
    if x1>=h_1+dh: 
        Thc = Thf_m2_star*np.exp(-gamma_m2*(x1-(h_1+dh)))+Thb_m2_star*np.exp(gamma_m2*(x1-H))# Homogeneous solution
        Tpc = Tpf_beta_2*np.exp(-beta_2*(x1-(h_1+dh)))*np.exp(1j*k2*np.tan(theta_2)*x1)+Tpb_beta_2*np.exp(beta_2*(x1-H))*np.exp(-1j*k2*np.tan(theta_2)*x1) # Particular solution
        Tc = Thc+Tpc # Complete solution
    delta_t = (np.exp(delta*time)*np.ones((len(x2),1))).T
    # Double Inverse Fourier transform
    Tc_temporal = T_ext+np.real(1./(2*np.pi)*delta_t*np.fft.irfft( \
                          np.fft.fftshift(np.fft.fft(Tc, axis=1, norm='ortho'),
                          axes=1), axis=0, norm='ortho'))
#    # Plot
#    id_tmax = np.argmin(np.abs(time-np.floor(time[-1])))
#    plt.figure()
#    plt.imshow(Tc_temporal[0:id_tmax,:]-273.15,origin='lower',
#               extent=[x2[0],x2[-1],time[0],time[id_tmax]], aspect='auto')
#    plt.xlabel('Position (mm)')
#    plt.ylabel('Time ($\mu$s)')
#    plt.title('Temperature (°C)')

    return Tc_temporal
