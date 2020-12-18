# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""

import numpy as np
import time as duration
from Dot_prod import dot4444, dot2444, dot2441, dot4441

def Homogeneous_solution(omega, k2, medium):
    
    # Elastic coefficients, density and thickness of the medium
    c11, c12, c22, c66, rho, h = medium
    # Terms of the matrix M
    a11 = rho*omega**2-k2**2*c66
    a22 = rho*omega**2-k2**2*c22
    a12 = k2*(c12+c66)
    # Coefficients of the quadratic equation 
    a_coef = c11*c66
    b_coef = -(c11*a22+c66*a11+a12**2)
    c_coef = a11*a22
    # Solutions of the quadratic equation
    det = b_coef**2-4*a_coef*c_coef 
    pow2_k1_t = (-b_coef+np.sqrt(det))/(2*a_coef)
    pow2_k1_l = (-b_coef-np.sqrt(det))/(2*a_coef)
    k1_t = np.sqrt(pow2_k1_t) # real part: positive, imaginary part: negative 
    k1_l = np.sqrt(pow2_k1_l) # real part: positive, imaginary part: negative 
    # Homogeneous solution
    u1h_l = a22-k1_l**2*c66
    u1h_t = a22-k1_t**2*c66
    u2h_l = -k1_l*a12 
    u2h_t = -k1_t*a12 
    
    return  u1h_l, u1h_t, u2h_l, u2h_t, k1_l, k1_t

def Particular_solution_gamma(omega, k2, medium_i, th_prop_i, d_i, T_i):   
    
    # Thermal properties
    cp_1, lambda1_mi, lambda2_mi, C_alpha1_mi, C_alpha2_mi = th_prop_i
    # Elastic coefficients, density and thickness of the medium
    c11_i, c12_i, c22_i, c66_i, rho_i, h_i = medium_i
    # Terms of the matrix M
    a11 = rho_i*omega**2-k2**2*c66_i
    a22 = rho_i*omega**2-k2**2*c22_i
    a12 = k2*(c12_i+c66_i)
    # Mat a_t
    xi_11 = a11+c11_i*d_i**2
    xi_12 = 1j*a12*d_i
    xi_22 = a22+c66_i*d_i**2
    a_t = np.array([[xi_11, xi_12],[xi_12, xi_22]])
    # Vec b_t
    b_t = np.array([[-C_alpha1_mi*d_i*T_i], [-1j*k2*C_alpha2_mi*T_i]])
    #%% Solve np.linalg.solve
    A = np.swapaxes(np.swapaxes(a_t,0,2),1,3)
    B = np.swapaxes(np.swapaxes(b_t,0,2),1,3)  
    sol = np.linalg.solve(A, B)
    up_1, up_2 = np.swapaxes(np.swapaxes(sol,0,2),1,3)[:,0,:,:]
    
    return up_1, up_2

def Particular_solution_beta(omega, k2, medium_i, th_prop_i, d_i, T_i, theta_i):   
    
    # Thermal properties
    cp_1, lambda1_mi, lambda2_mi, C_alpha1_mi, C_alpha2_mi = th_prop_i
    # Elastic coefficients, density and thickness of the medium
    c11_i, c12_i, c22_i, c66_i, rho_i, h_i = medium_i
    # Terms of the matrix M
    a11 = rho_i*omega**2-k2**2*c66_i
    a22 = rho_i*omega**2-k2**2*c22_i
    a12 = k2*(c12_i+c66_i)
    # Mat a_t
    xi_11 = a11+c11_i*(d_i-1j*k2*np.tan(theta_i))**2
    xi_12 = 1j*a12*(d_i-1j*k2*np.tan(theta_i))
    xi_22 = a22+c66_i*(d_i-1j*k2*np.tan(theta_i))**2
    a_t = np.array([[xi_11, xi_12],[xi_12, xi_22]])
    # Vec b_t
    b_t = np.array([[-C_alpha1_mi*(d_i-1j*k2*np.tan(theta_i))*T_i], [-1j*k2*C_alpha2_mi*T_i]])
    #%% Solve np.linalg.solve
    A = np.swapaxes(np.swapaxes(a_t,0,2),1,3)
    B = np.swapaxes(np.swapaxes(b_t,0,2),1,3)  
    sol = np.linalg.solve(A, B)
    up_1, up_2 = np.swapaxes(np.swapaxes(sol,0,2),1,3)[:,0,:,:]
    
    return up_1, up_2

def Boundary_conditions_Particular_solutions_Source_Term(omega, k2, medium, beta, 
                                                         th_prop_m1, th_prop_m2, t_gamma, Tp, Th, 
                                                         upf_beta_1, upb_beta_1, upf_gamma_m1, upb_gamma_m1, 
                                                         upf_beta_2, upb_beta_2, upf_gamma_m2, upb_gamma_m2,
                                                         theta_1, theta_2, dh):
    #%% beta, t_gamma, Tp, Th
    beta_1, beta_2 = beta
    gamma_m1, gamma_m2 = t_gamma 
    Tpf_beta_1, Tpb_beta_1, Tpf_beta_2, Tpb_beta_2 = Tp
    Thf_m1, Thb_m1_star, Thf_m2_star, Thb_m2_star = Th
    #%% Particular solution medium 1
    u1pf_beta_m1, u2pf_beta_m1 = upf_beta_1
    u1pb_beta_m1, u2pb_beta_m1 = upb_beta_1
    u1pf_gamma_m1, u2pf_gamma_m1 = upf_gamma_m1
    u1pb_gamma_m1, u2pb_gamma_m1 = upb_gamma_m1
    #%% Particular solution medium 2
    u1pf_beta_m2, u2pf_beta_m2 = upf_beta_2
    u1pb_beta_m2, u2pb_beta_m2 = upb_beta_2
    u1pf_gamma_m2, u2pf_gamma_m2 = upf_gamma_m2
    u1pb_gamma_m2, u2pb_gamma_m2 = upb_gamma_m2
    #%% Elastic coefficients, density and thickness of the medium
    c11_1, c12_1, c22_1, c66_1, rho_1, h_1 = medium[0]
    c11_2, c12_2, c22_2, c66_2, rho_2, h_2 = medium[1]
    H = h_1+dh+h_2 # Total thickness (mm)
    #%% Thermal properties of medium 1 and medium 2
    cp_1, lambda1_m1, lambda2_m1, C_alpha1_m1, C_alpha2_m1 = th_prop_m1 
    cp_2, lambda1_m2, lambda2_m2, C_alpha1_m2, C_alpha2_m2 = th_prop_m2
    #%% Limit conditions - Particular solutions
    # Free surface x1=0, Medium 1
        # f: forward exp(j(omega*t-k*x1)), example: up1_gammaf_m1 
        # b: backward exp(j(omega*t+k*x1)), example: up1_gammab_m1 
    u1p0_m1 = u1pf_beta_m1+u1pb_beta_m1*np.exp(-beta_1*h_1)+\
                u1pf_gamma_m1+u1pb_gamma_m1*np.exp(-gamma_m1*h_1)
    du1p0_m1 = -(beta_1-1j*k2*np.tan(theta_1))*u1pf_beta_m1+(beta_1-1j*k2*np.tan(theta_1))*u1pb_beta_m1*np.exp(-beta_1*h_1)- \
                gamma_m1*u1pf_gamma_m1+gamma_m1*u1pb_gamma_m1*np.exp(-gamma_m1*h_1)
    u2p0_m1 = u2pf_beta_m1+u2pb_beta_m1*np.exp(-beta_1*h_1)+\
                u2pf_gamma_m1+u2pb_gamma_m1*np.exp(-gamma_m1*h_1)
    du2p0_m1 = -(beta_1-1j*k2*np.tan(theta_1))*u2pf_beta_m1+(beta_1-1j*k2*np.tan(theta_1))*u2pb_beta_m1*np.exp(-beta_1*h_1)- \
                gamma_m1*u2pf_gamma_m1+gamma_m1*u2pb_gamma_m1*np.exp(-gamma_m1*h_1)
    T0_m1 = Thf_m1+Thb_m1_star*np.exp(-gamma_m1*h_1)+Tpf_beta_1+Tpb_beta_1*np.exp(-beta_1*h_1)
    sigma11ps0_m1 = c11_1*du1p0_m1-1j*k2*c12_1*u2p0_m1-C_alpha1_m1*T0_m1
    sigma12ps0_m1 = c66_1*(-1j*k2*u1p0_m1+du2p0_m1)
    # Interface 1, x1=h1, Medium 1    
    u1ph1_m1 = u1pf_beta_m1*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)+u1pb_beta_m1*np.exp(-1j*k2*np.tan(theta_1)*h_1)+\
                u1pf_gamma_m1*np.exp(-gamma_m1*h_1)+u1pb_gamma_m1
    du1ph1_m1 = -(beta_1-1j*k2*np.tan(theta_1))*u1pf_beta_m1*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)+\
                (beta_1-1j*k2*np.tan(theta_1))*u1pb_beta_m1*np.exp(-1j*k2*np.tan(theta_1)*h_1)- \
                gamma_m1*u1pf_gamma_m1*np.exp(-gamma_m1*h_1)+gamma_m1*u1pb_gamma_m1
    u2ph1_m1 = u2pf_beta_m1*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)+u2pb_beta_m1*np.exp(-1j*k2*np.tan(theta_1)*h_1)+\
                u2pf_gamma_m1*np.exp(-gamma_m1*h_1)+u2pb_gamma_m1
    du2ph1_m1 = -(beta_1-1j*k2*np.tan(theta_1))*u2pf_beta_m1*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)+\
                (beta_1-1j*k2*np.tan(theta_1))*u2pb_beta_m1*np.exp(-1j*k2*np.tan(theta_1)*h_1)- \
                gamma_m1*u2pf_gamma_m1*np.exp(-gamma_m1*h_1)+gamma_m1*u2pb_gamma_m1
    Th1_m1 = Thf_m1*np.exp(-gamma_m1*h_1)+Thb_m1_star+Tpf_beta_1*np.exp(-beta_1*h_1)*np.exp(1j*k2*np.tan(theta_1)*h_1)+Tpb_beta_1*np.exp(-1j*k2*np.tan(theta_1)*h_1)
    sigma11psh1_m1 = c11_1*du1ph1_m1-1j*k2*c12_1*u2ph1_m1-C_alpha1_m1*Th1_m1
    sigma12psh1_m1 = c66_1*(-1j*k2*u1ph1_m1+du2ph1_m1)
    # Interface 2, x1=h1+dh, Medium 2
    u1ph1dh_m2 = u1pf_beta_m2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))+\
                u1pb_beta_m2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh))+ \
                u1pf_gamma_m2+u1pb_gamma_m2*np.exp(-gamma_m2*h_2)
    du1ph1dh_m2 = -(beta_2-1j*k2*np.tan(theta_2))*u1pf_beta_m2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))+\
                    (beta_2-1j*k2*np.tan(theta_2))*u1pb_beta_m2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh))- \
                    gamma_m2*u1pf_gamma_m2+gamma_m2*u1pb_gamma_m2*np.exp(-gamma_m2*h_2)
    u2ph1dh_m2 = u2pf_beta_m2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))+\
                    u2pb_beta_m2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh))+\
                    u2pf_gamma_m2+u2pb_gamma_m2*np.exp(-gamma_m2*h_2)
    du2ph1dh_m2 = -(beta_2-1j*k2*np.tan(theta_2))*u2pf_beta_m2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))+\
                    (beta_2-1j*k2*np.tan(theta_2))*u2pb_beta_m2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh))-\
                    gamma_m2*u2pf_gamma_m2+gamma_m2*u2pb_gamma_m2*np.exp(-gamma_m2*h_2)
    Th1dh_m2 = Thf_m2_star+Thb_m2_star*np.exp(-gamma_m2*h_2)+\
                Tpf_beta_2*np.exp(1j*k2*np.tan(theta_2)*(h_1+dh))+Tpb_beta_2*np.exp(-beta_2*h_2)*np.exp(-1j*k2*np.tan(theta_2)*(h_1+dh))
    sigma11psh1dh_m2 = c11_2*du1ph1dh_m2-1j*k2*c12_2*u2ph1dh_m2-C_alpha1_m2*Th1dh_m2
    sigma12psh1dh_m2 = c66_2*(-1j*k2*u1ph1dh_m2+du2ph1dh_m2)
    # Free surface x1=H, Medium 2
    u1pH_m2 = u1pf_beta_m2*np.exp(-beta_2*h_2)*np.exp(1j*k2*np.tan(theta_2)*H)+\
                u1pb_beta_m2*np.exp(-1j*k2*np.tan(theta_2)*H)+ \
                u1pf_gamma_m2*np.exp(-gamma_m2*h_2)+u1pb_gamma_m2
    du1pH_m2 = -(beta_2-1j*k2*np.tan(theta_2))*u1pf_beta_m2*np.exp(-beta_2*h_2)*np.exp(1j*k2*np.tan(theta_2)*H)+\
                (beta_2-1j*k2*np.tan(theta_2))*u1pb_beta_m2*np.exp(-1j*k2*np.tan(theta_2)*H)- \
                gamma_m2*u1pf_gamma_m2*np.exp(-gamma_m2*h_2)+gamma_m2*u1pb_gamma_m2
    u2pH_m2 = u2pf_beta_m2*np.exp(-beta_2*h_2)*np.exp(1j*k2*np.tan(theta_2)*H)+\
                u2pb_beta_m2*np.exp(-1j*k2*np.tan(theta_2)*H)+\
                u2pf_gamma_m2*np.exp(-gamma_m2*h_2)+u2pb_gamma_m2
    du2pH_m2 = -(beta_2-1j*k2*np.tan(theta_2))*u2pf_beta_m2*np.exp(-beta_2*h_2)*np.exp(1j*k2*np.tan(theta_2)*H)+\
                (beta_2-1j*k2*np.tan(theta_2))*u2pb_beta_m2*np.exp(-1j*k2*np.tan(theta_2)*H)-\
                gamma_m2*u2pf_gamma_m2*np.exp(-gamma_m2*h_2)+gamma_m2*u2pb_gamma_m2
    TH_m2 = Thf_m2_star*np.exp(-gamma_m2*h_2)+Thb_m2_star+Tpf_beta_2*np.exp(-beta_2*h_2)*np.exp(1j*k2*np.tan(theta_2)*H)+\
            Tpb_beta_2*np.exp(-1j*k2*np.tan(theta_2)*H)
    sigma11psH_m2 = c11_2*du1pH_m2-1j*k2*c12_2*u2pH_m2-C_alpha1_m2*TH_m2
    sigma12psH_m2 = c66_2*(-1j*k2*u1pH_m2+du2pH_m2)

    return sigma11ps0_m1, sigma12ps0_m1, sigma11psH_m2, sigma12psH_m2, \
            u1ph1_m1, u2ph1_m1, sigma11psh1_m1, sigma12psh1_m1, \
            u1ph1dh_m2, u2ph1dh_m2, sigma11psh1dh_m2, sigma12psh1dh_m2

def Boundary_conditions_Homogeneous_solutions(omega, k2, medium, sh_m1, sh_m2):
            
    #%% Homogeneous solution medium 1
    u1h_lm1, u1h_tm1, u2h_lm1, u2h_tm1, k1_lm1, k1_tm1 = sh_m1
    #%% Homogeneous solution medium 2
    u1h_lm2, u1h_tm2, u2h_lm2, u2h_tm2, k1_lm2, k1_tm2 = sh_m2
    #%% Elastic coefficients, density and thickness of the medium
    c11_1, c12_1, c22_1, c66_1, rho_1, h_1 = medium[0]
    c11_2, c12_2, c22_2, c66_2, rho_2, h_2 = medium[1]
    #%% Medium 1
    rl_m1 = c11_1*k1_lm1*u1h_lm1-c12_1*k2*u2h_lm1
    rt_m1 = c11_1*k1_tm1*u1h_tm1-c12_1*k2*u2h_tm1
    ml_m1 = c66_1*(-1j*k2*u1h_lm1+1j*k1_lm1*u2h_lm1)
    mt_m1 = c66_1*(-1j*k2*u1h_tm1+1j*k1_tm1*u2h_tm1)
    sl1 = (1.-np.exp(-1j*k1_lm1*h_1))/(2.*1j)
    st1 = (1.-np.exp(-1j*k1_tm1*h_1))/(2.*1j)
    cl1 = (1.+np.exp(-1j*k1_lm1*h_1))/2.
    ct1 = (1.+np.exp(-1j*k1_tm1*h_1))/2.
    #%% Medium 2
    rl_m2 = c11_2*1j*k1_lm2*u1h_lm2-c12_2*1j*k2*u2h_lm2
    rt_m2 = c11_2*1j*k1_tm2*u1h_tm2-c12_2*1j*k2*u2h_tm2
    ml_m2 = c66_2*(-1j*k2*u1h_lm2+1j*k1_lm2*u2h_lm2)
    mt_m2 = c66_2*(-1j*k2*u1h_tm2+1j*k1_tm2*u2h_tm2)   
    el_m2 = np.exp(-1j*k1_lm2*h_2)
    et_m2 = np.exp(-1j*k1_tm2*h_2)  
    #%% Limit conditions - Homogeneous solutions
    # Free surface, x1 = 0, medium 1
    psi_0_m1_star = np.array([[rl_m1*sl1, rt_m1*st1, rl_m1*cl1, rt_m1*ct1],
                              [ml_m1*cl1, mt_m1*ct1, -ml_m1*sl1, -mt_m1*st1]])
    # Interface 1, x1 = h1, medium 1
    psi_h1_m1_star = np.array([[u1h_lm1*cl1, u1h_tm1*ct1, u1h_lm1*sl1, u1h_tm1*st1],
                               [1j*u2h_lm1*sl1, 1j*u2h_tm1*st1, -1j*u2h_lm1*cl1, -1j*u2h_tm1*ct1],
                               [-rl_m1*sl1, -rt_m1*st1, rl_m1*cl1, rt_m1*ct1],
                               [ml_m1*cl1, mt_m1*ct1, ml_m1*sl1, mt_m1*st1]])
    psi_h1_m1_u_star = psi_h1_m1_star[0:2,:,:,:]
    psi_h1_m1_sigma_star = psi_h1_m1_star[2::,:,:,:] 
    # Interface 2, x1 = h1+dh, medium 2
    psi_h1dh_m2_star = np.array([[u1h_lm2*el_m2, u1h_lm2, u1h_tm2*et_m2, u1h_tm2],
                                 [u2h_lm2*el_m2, -u2h_lm2, u2h_tm2*et_m2, -u2h_tm2],
                                 [rl_m2*el_m2, -rl_m2, rt_m2*et_m2, -rt_m2],
                                 [ml_m2*el_m2, ml_m2, mt_m2*et_m2, mt_m2]])
    # Free surface, x1 = H, medium 2
    psi_H_m2_star = np.array([[rl_m2, -rl_m2*el_m2, rt_m2, -rt_m2*et_m2],
                              [ml_m2, ml_m2*el_m2, mt_m2, mt_m2*et_m2]])
    # Stress: Interface 1 (x1 = h1) + Stress: Free surface (x1 = 0)
    gamma_psi_h_star = np.zeros((4,4, len(omega[:, 0]), len(k2[0, :])),complex)
    gamma_psi_h_star[0:2,:,:,:] = psi_h1_m1_sigma_star
    gamma_psi_h_star[2::,:,:,:] = psi_0_m1_star
    #%% Inverse of Matrix gamma_psi_h_star
    gamma_psi_h_star_swap_axis = np.swapaxes(np.swapaxes(gamma_psi_h_star,0,2),1,3)
    inv_gamma_psi_h_star_swap_axis = np.linalg.inv(gamma_psi_h_star_swap_axis)
    inv_gamma_psi_h_star = np.swapaxes(np.swapaxes(inv_gamma_psi_h_star_swap_axis,0,2),1,3)
    
    return psi_h1_m1_u_star, gamma_psi_h_star, inv_gamma_psi_h_star, psi_h1dh_m2_star, psi_H_m2_star

def Interfacial_layer(omega, pow_kn, pow_kt):
    
    KN = np.power(10, pow_kn) #+1j*np.real(omega)*0.001 # kN/mm3 (multiply by 1e12 to obtain N/m3)
    KT = np.power(10, pow_kt) #+1j*np.real(omega)*0.001 # kN/mm3 (multiply by 1e12 to obtain N/m3)
    one = np.ones(np.shape(omega), complex) 
    zero = np.zeros(np.shape(omega), complex)
    K = np.array([[one, zero, -one/KN, zero],
                  [zero, one, zero, -one/KT],
                  [zero, zero, one, zero],
                  [zero, zero, zero, one]])
    return K

def Perfect_coupling(omega):
    
    one = np.ones(np.shape(omega), complex) 
    zero = np.zeros(np.shape(omega), complex)
    K = np.array([[one, zero, zero, zero],
                  [zero, one, zero, zero],
                  [zero, zero, one, zero],
                  [zero, zero, zero, one]])
    return K

def Orthotropic_layer(omega, k2, medium_i, sh_i): 
    
    # Elastic coefficients, density and thickness of the medium
    c11_i, c12_i, c22_i, c66_i, rho, h_i = medium_i
    u1_l_i, u1_t_i, u2_l_i, u2_t_i, k1_l_i, k1_t_i = sh_i 
    #%% rl, rt, ml, mt 
    rl = c11_i*1j*k1_l_i*u1_l_i-c12_i*1j*k2*u2_l_i
    rt = c11_i*1j*k1_t_i*u1_t_i-c12_i*1j*k2*u2_t_i
    ml = c66_i*(-1j*k2*u1_l_i+1j*k1_l_i*u2_l_i)
    mt = c66_i*(-1j*k2*u1_t_i+1j*k1_t_i*u2_t_i)   
    #%% Mat x1 = h 
    e12 = np.exp(-1j*k1_l_i*h_i)
    e14 = np.exp(-1j*k1_t_i*h_i)
    psi_h_i = np.array([[u1_l_i, u1_l_i*e12, u1_t_i, u1_t_i*e14],
                         [u2_l_i, -u2_l_i*e12, u2_t_i, -u2_t_i*e14],
                         [rl, -rl*e12, rt, -rt*e14],
                         [ml, ml*e12, mt, mt*e14]])        
    inv_psi_h_i_swap = np.linalg.inv(np.swapaxes(np.swapaxes(psi_h_i,0,2),1,3))
    inv_psi_h_i = np.swapaxes(np.swapaxes(inv_psi_h_i_swap,0,2),1,3)
    #%% Mat x1 = 0
    e11 = e12
    e13 = e14
    psi_0 = np.array([[u1_l_i*e11, u1_l_i, u1_t_i*e13, u1_t_i],
                         [u2_l_i*e11, -u2_l_i, u2_t_i*e13, -u2_t_i],
                         [rl*e11, -rl, rt*e13, -rt],
                         [ml*e11, ml, mt*e13, mt]])
    #%% Matrix B
    B = dot4444(psi_0, inv_psi_h_i)
    
    return B 

def Coefficients_xis_xia_af_ab(omega, k2, L,
                               psi_h1_m1_u_star,gamma_psi_h_star,inv_gamma_psi_h_star,psi_h1dh_m2_star,psi_H_m2_star,
                               sigma11ps0_m1, sigma12ps0_m1, sigma11psH_m2, sigma12psH_m2,
                               u1ph1_m1, u2ph1_m1, sigma11psh1_m1, sigma12psh1_m1,
                               u1ph1dh_m2, u2ph1dh_m2, sigma11psh1dh_m2, sigma12psh1dh_m2):
    
    #%% Resolution of the two 4x4 systems
    # Coupling matrix
    L_u = L[0:2,:,:,:]
    L_sigma = L[2::,:,:,:]
    #%% Stress: Interface 1 (x1 = h1) + Stress: Free surface (x1 = 0) 
    # gamma_Ah_star (4, 4, len(omega), len(k2))
    gamma_Ah_star = np.zeros((4,4, len(omega[:, 0]), len(k2[0, :])),complex)
    gamma_Ah_star[0:2,:,:,:] = dot2444(L_sigma, psi_h1dh_m2_star)
    # gamma_Ap_star (4, 1, len(omega), len(k2))
    gamma_Ap_star = np.zeros((4,1, len(omega[:, 0]), len(k2[0, :])),complex)
    usigmaph1dh_m2 = np.array([[u1ph1dh_m2],[u2ph1dh_m2],[sigma11psh1dh_m2],[sigma12psh1dh_m2]])  
    gamma_Ap_star[0:2,:,:,:] = dot2441(L_sigma, usigmaph1dh_m2) \
                                -np.array([[sigma11psh1_m1],[sigma12psh1_m1]])
    gamma_Ap_star[2::,:,:,:] = -np.array([[sigma11ps0_m1],[sigma12ps0_m1]])
    #%% Displacement: Interface 2 (x1=h1+dh) + Stress: Free surface (x1=H)
    # phi_A_star (4, 4, len(omega), len(k2))
    phi_Ah_star = np.zeros((4,4, len(omega[:, 0]), len(k2[0, :])),complex)
    mat1 = dot2444(psi_h1_m1_u_star,inv_gamma_psi_h_star)
    phi_Ah_star[0:2,:,:,:] = dot2444(mat1,gamma_Ah_star)-dot2444(L_u,psi_h1dh_m2_star)
    phi_Ah_star[2::,:,:,:] = psi_H_m2_star
    # phi_P_star (4, 1, len(omega), len(k2))
    phi_Ap_star = np.zeros((4,1, len(omega[:, 0]), len(k2[0, :])),complex)
    phi_Ap_star[0:2,:,:,:] = -dot2441(mat1,gamma_Ap_star)+dot2441(L_u,usigmaph1dh_m2)- \
                            np.array([[u1ph1_m1],[u2ph1_m1]])
    phi_Ap_star[2::,:,:,:] = -np.array([[sigma11psH_m2],[sigma12psH_m2]])                          
    #%% Solve ab_lm2_star, af_lm2_star, ab_tm2_star, af_tm2_star
    # Solve np.linalg.solve
    A1 = np.swapaxes(np.swapaxes(phi_Ah_star,0,2),1,3)
    B1 = np.swapaxes(np.swapaxes(phi_Ap_star,0,2),1,3)  
    sol1 = np.linalg.solve(A1, B1)
    ab_lm2_star, af_lm2_star, ab_tm2_star, af_tm2_star = np.swapaxes(np.swapaxes(sol1,0,2),1,3)[:,0,:,:]
    #%% Solve xis_lm1_star, xis_tm1_star, xia_lm1_star, xia_tm1_star
    mat2 = dot4441(gamma_Ah_star,np.array([[ab_lm2_star], [af_lm2_star], \
                                           [ab_tm2_star], [af_tm2_star]]))+gamma_Ap_star
    # Solve np.linalg.solve
    A2 = np.swapaxes(np.swapaxes(gamma_psi_h_star,0,2),1,3)
    B2 = np.swapaxes(np.swapaxes(mat2,0,2),1,3)  
    sol2 = np.linalg.solve(A2, B2)
    xis_lm1_star, xis_tm1_star, xia_lm1_star, xia_tm1_star = np.swapaxes(np.swapaxes(sol2,0,2),1,3)[:,0,:,:]

    return xis_lm1_star, xis_tm1_star, xia_lm1_star, \
            xia_tm1_star, ab_lm2_star, af_lm2_star, ab_tm2_star, af_tm2_star

def CS_m1(x1, h_1, k1_lm1, k1_tm1):
    
    cl_m1 = (np.exp(1j*k1_lm1*(x1-h_1))+np.exp(-1j*k1_lm1*x1))/2
    ct_m1 = (np.exp(1j*k1_tm1*(x1-h_1))+np.exp(-1j*k1_tm1*x1))/2
    sl_m1 = (np.exp(1j*k1_lm1*(x1-h_1))-np.exp(-1j*k1_lm1*x1))/(2*1j)
    st_m1 = (np.exp(1j*k1_tm1*(x1-h_1))-np.exp(-1j*k1_tm1*x1))/(2*1j)
    
    return cl_m1, ct_m1, sl_m1, st_m1

def E_m2(x1, h_1, H, dh, k1_lm2, k1_tm2):
    
    elH_m2 = np.exp(1j*k1_lm2*(x1-H))
    etH_m2 = np.exp(1j*k1_tm2*(x1-H))
    elh1_m2 = np.exp(-1j*k1_lm2*(x1-(h_1+dh)))
    eth1_m2 = np.exp(-1j*k1_tm2*(x1-(h_1+dh)))
    
    return elH_m2, etH_m2, elh1_m2, eth1_m2
           
def Simu_multilayer(L, time, x2, delta, x1, omega, k2, medium, dh, sh_m1, sh_m2, beta, t_gamma, 
                    upf_beta_1, upb_beta_1, upf_gamma_m1, upb_gamma_m1, 
                    upf_beta_2, upb_beta_2, upf_gamma_m2, upb_gamma_m2,
                    psi_h1_m1_u_star,gamma_psi_h_star,inv_gamma_psi_h_star,psi_h1dh_m2_star,psi_H_m2_star,
                    sigma11ps0_m1, sigma12ps0_m1, sigma11psH_m2, sigma12psH_m2,
                    u1ph1_m1, u2ph1_m1, sigma11psh1_m1, sigma12psh1_m1,
                    u1ph1dh_m2, u2ph1dh_m2, sigma11psh1dh_m2, sigma12psh1dh_m2,
                    theta_1, theta_2):  
    
    #%% beta, t_gamma, Tp, Th
    beta_1, beta_2 = beta
    gamma_m1, gamma_m2 = t_gamma 
    #%% Homogeneous solution medium 1
    u1h_lm1, u1h_tm1, u2h_lm1, u2h_tm1, k1_lm1, k1_tm1 = sh_m1
    #%% Homogeneous solution medium 2
    u1h_lm2, u1h_tm2, u2h_lm2, u2h_tm2, k1_lm2, k1_tm2 = sh_m2
    #%% Particular solution medium 1
    u1pf_beta_m1, u2pf_beta_m1 = upf_beta_1
    u1pb_beta_m1, u2pb_beta_m1 = upb_beta_1
    u1pf_gamma_m1, u2pf_gamma_m1 = upf_gamma_m1
    u1pb_gamma_m1, u2pb_gamma_m1 = upb_gamma_m1
    #%% Particular solution medium 2
    u1pf_beta_m2, u2pf_beta_m2 = upf_beta_2
    u1pb_beta_m2, u2pb_beta_m2 = upb_beta_2
    u1pf_gamma_m2, u2pf_gamma_m2 = upf_gamma_m2
    u1pb_gamma_m2, u2pb_gamma_m2 = upb_gamma_m2
    #%% Elastic coefficients, density and thickness of the medium
    c11_1, c12_1, c22_1, c66_1, rho_1, h_1 = medium[0]
    c11_2, c12_2, c22_1, c66_1, rho_2, h_2 = medium[1]
    c11_i1, c12_i1, c22_i1, c66_i1, rho_i1, h_i1 = medium[2]
    H = h_1+h_2+dh # Total thickness
    #%% Coefficients xis_lm1, xis_tm1, xia_lm1, xia_tm1, af_lm2_star, 
    #   af_tm2_star, ab_lm2_star, ab_tm2_star
    xis_lm1_star, xis_tm1_star, xia_lm1_star, \
    xia_tm1_star, ab_lm2_star, af_lm2_star, ab_tm2_star, af_tm2_star = \
    Coefficients_xis_xia_af_ab(omega, k2, L,
                               psi_h1_m1_u_star, gamma_psi_h_star, inv_gamma_psi_h_star, psi_h1dh_m2_star, psi_H_m2_star,
                               sigma11ps0_m1, sigma12ps0_m1, sigma11psH_m2, sigma12psH_m2,
                               u1ph1_m1, u2ph1_m1, sigma11psh1_m1, sigma12psh1_m1,
                               u1ph1dh_m2, u2ph1dh_m2, sigma11psh1dh_m2, sigma12psh1dh_m2)
    #%% Complete solution uc1, uc2
    # Medium 1
    if x1<=h_1:
        cl_m1, ct_m1, sl_m1, st_m1 = CS_m1(x1, h_1, k1_lm1, k1_tm1)
        # Homogeneous solutions
        u1_h = u1h_lm1*(xis_lm1_star*cl_m1+xia_lm1_star*sl_m1)+ \
                u1h_tm1*(xis_tm1_star*ct_m1+xia_tm1_star*st_m1)
#        u2_h = 1j*(u2h_lm1*(xis_lm1_star*sl_m1-xia_lm1_star*cl_m1)+ \
#                u2h_tm1*(xis_tm1_star*st_m1-xia_tm1_star*ct_m1))
        # Particular solutions 
        u1_p = u1pf_beta_m1*np.exp(-beta_1*x1)*np.exp(1j*k2*np.tan(theta_1)*x1)+\
                u1pb_beta_m1*np.exp(beta_1*(x1-h_1))*np.exp(-1j*k2*np.tan(theta_1)*x1)+ \
                u1pf_gamma_m1*np.exp(-gamma_m1*x1)+u1pb_gamma_m1*np.exp(gamma_m1*(x1-h_1))
#        u2_p = u2pf_beta_m1*np.exp(-beta_1*x1)*np.exp(1j*k2*np.tan(theta_1)*x1)+\
#                u2pb_beta_m1*np.exp(beta_1*(x1-h_1))*np.exp(-1j*k2*np.tan(theta_1)*x1)+ \
#                u2pf_gamma_m1*np.exp(-gamma_m1*x1)+u2pb_gamma_m1*np.exp(gamma_m1*(x1-h_1))
        # Complete solutions
        uc1 = u1_h+u1_p
#        uc2 = u2_h+u2_p
    # Sublayer(s)
    if x1>h_1 and x1<h_1+dh: 
        uc1 = np.zeros((np.shape(u1h_lm1)), complex)
#        uc2 = np.zeros((np.shape(u1h_lm1)), complex)
    # Medium 2
    if x1>h_1+dh:
        elH_m2, etH_m2, elh1_m2, eth1_m2 = E_m2(x1, h_1, H, dh, k1_lm2, k1_tm2)
        # Homogeneous solutions
        u1_h = u1h_lm2*(af_lm2_star*elh1_m2+ab_lm2_star*elH_m2)+ \
                u1h_tm2*(af_tm2_star*eth1_m2+ab_tm2_star*etH_m2)
#        u2_h = u2h_lm2*(-af_lm2_star*elh1_m2+ab_lm2_star*elH_m2)+ 
#                u2h_tm2*(-af_tm2_star*eth1_m2+ab_tm2_star*etH_m2)
        # Particular solutions
        u1_p = u1pf_beta_m2*np.exp(-beta_2*(x1-(h_1+dh)))*np.exp(1j*k2*np.tan(theta_2)*x1)+\
                u1pb_beta_m2*np.exp(beta_2*(x1-H))*np.exp(-1j*k2*np.tan(theta_2)*x1)+ \
                u1pf_gamma_m2*np.exp(-gamma_m2*(x1-(h_1+dh)))+u1pb_gamma_m2*np.exp(gamma_m2*(x1-H))
#        u2_p = u2pf_beta_m2*np.exp(-beta_2*(x1-(h_1+dh)))*np.exp(1j*k2*np.tan(theta_2)*x1)+\
#                u2pb_beta_m2*np.exp(beta_2*(x1-H))*np.exp(-1j*k2*np.tan(theta_2)*x1)+ \ 
#                u2pf_gamma_m2*np.exp(-gamma_m2*(x1-(h_1+dh)))+u2pb_gamma_m2*np.exp(gamma_m2*(x1-H))
        # Complete solutions
        uc1 = u1_h+u1_p
#        uc2 = u2_h+u2_p
    delta_t = (np.exp(delta*time)*np.ones((len(x2),1))).T
    uc1_temporal = 1./(2*np.pi)*np.real(delta_t*np.fft.irfft(np.fft.fftshift( 
                      np.fft.fft(uc1, axis=1, norm='ortho'),axes=1), axis=0, norm='ortho'))
    
    return uc1, uc1_temporal