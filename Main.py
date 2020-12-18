# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time as duration
# Electromagnetic problem
from Electromagnetic_problem import Coef_EM, Coupling_EM, Power_densities_Q
# Thermal_problem
from Thermal_problem import Coef_Temp, Temp_Complete_solution
# Elastodynamic problem
from Elastodynamic_problem import (Homogeneous_solution, 
                                   Particular_solution_beta, 
                                   Particular_solution_gamma, 
                                   Boundary_conditions_Homogeneous_solutions,
                                   Boundary_conditions_Particular_solutions_Source_Term,
                                   Orthotropic_layer, Interfacial_layer, Perfect_coupling, 
                                   Simu_multilayer)
# Transfer matrix
from Transfer_matrix import ElectroMag_coupling, Thermal_coupling, Mechanical_coupling
# Angular frequency (omega), wavenumber (k2)
from Omega_k2 import Omega_k2
# Filter
from Filter import Butter_bandpass_filter 
# Open results
from Open_results import Open_results

plt.rcParams.update({'font.size':24})
plt.rcParams.update({'figure.autolayout': True})
plt.close('all')
 
#%% General parameters
id_save = 'Test' # Filename
x1_vec = [0.] # x1-position (mm), x1 between 0 and H, with H=h1+h2+dh
# Time, x2
dt = 8e-3 # Step t (µs)
dx2 = 0.08 # Step x2 (mm)
time = np.arange(0., 16.384, dt) # Time vector (µs)
x2 = np.arange(-81.92, 81.92, dx2) # x2 vector (mm)
# Angular frequency (omega), wavenumber (k2)
omega, k2, delta = Omega_k2(time, x2) # omega (rad/µs), k2 (rad/mm), delta (shift in the imaginary plane)
#%% Mechanical Parameters
# Medium 1
h_1 = 1.5 # Thickness (mm)
vpl_1 = 6.380 # L-wave (mm/µs)
vpt_1 = 3.133 # T-wave (mm/µs)
rho_1 = 2.7 # Density (g/cm3)
c11_1 = vpl_1**2*rho_1 # Elastic coefiicient c_11 (GPa)
c66_1 = vpt_1**2*rho_1 # Elastic coefiicient c_66 (GPa)
c12_1 = (c11_1-2*c66_1) # Elastic coefiicient c_12 (GPa)
c22_1 = c11_1 # Elastic coefiicient c_22 (GPa)
eta11_1, eta12_1, eta22_1, eta66_1 = 0., 0., 0., 0. # Viscosity (Kelvin-Voigt model)
medium_1 = np.array([c11_1+1j*omega*eta11_1, c12_1+1j*omega*eta12_1,
                     c22_1+1j*omega*eta22_1, c66_1+1j*omega*eta66_1, 
                     rho_1, h_1])
# Medium 2
h_2 = 3. # Thickness (mm)
vpl_2 = 6.380 # L-wave (mm/µs)
vpt_2 = 3.133 # T-wave (mm/µs)
rho_2 = 2.7 # Density (g/cm3)
c11_2 = vpl_2**2*rho_2 # Elastic coefiicient c_11 (GPa)
c66_2 = vpt_2**2*rho_2 # Elastic coefiicient c_66 (GPa)
c12_2 = (c11_2-2*c66_2) # Elastic coefiicient c_12 (GPa)
c22_2 = c11_2 # Elastic coefiicient c_22 (GPa)
eta11_2, eta12_2, eta22_2, eta66_2 = 0., 0., 0., 0. # Viscosity (Kelvin-Voigt model)
medium_2 = np.array([c11_2+1j*omega*eta11_2, c12_2+1j*omega*eta12_2,
                     c22_2+1j*omega*eta22_2, c66_2+1j*omega*eta66_2, 
                     rho_2, h_2])
#%% Thermal parameters
# Medium 1
lambda11_m1 = 150*1e-9 # Thermal conductivity (MW.mm-1.K-1)
lambda22_m1 = lambda11_m1 # Thermal conductivity (MW.mm-1.K-1)
cp_1 = 900*1e-6 # Specific heat (J.mg-1.K-1)
alpha1_m1 = 25e-6 # Coefficient of thermal expansion (K-1) 
alpha2_m1 = alpha1_m1 # Coefficient of thermal expansion (K-1) 
C_alpha1_m1 = alpha1_m1*c11_1+alpha2_m1*c12_1 # (GPa.K-1)
C_alpha2_m1 = alpha1_m1*c12_1+alpha2_m1*c22_1 # (GPa.K-1)
th_prop_m1 = np.array([cp_1, lambda11_m1, lambda22_m1, C_alpha1_m1, C_alpha2_m1])
# Medium 2
lambda11_m2 = lambda11_m1 # Thermal conductivity (MW.mm-1.K-1)
lambda22_m2 = lambda22_m1 # Thermal conductivity (MW.mm-1.K-1)
cp_2 = cp_1 # Specific heat (J.mg-1.K-1)
alpha1_m2 = 25e-6 # Coefficient of thermal expansion (K-1)
alpha2_m2 = alpha1_m1 # Coefficient of thermal expansion (K-1) 
C_alpha1_m2 = alpha1_m2*c11_2+alpha2_m2*c12_2 # (GPa.K-1)
C_alpha2_m2 = alpha1_m2*c12_2+alpha2_m2*c22_2 # (GPa.K-1)
th_prop_m2 = np.array([cp_2, lambda11_m2, lambda22_m2, C_alpha1_m2, C_alpha2_m2])
# External conditions
h = 25*1e-12 # Heat transfer coefficient (MW.mm-2.K-1) 
T_ext = 20+273.15 # External temperature (K) 
#%% Electromagnetic parameters
c_0 = 299792.458 # Speed of light in vacuum (mm/µs) 
opt_wav = 532e-6 # Optical wavelength of the laser source (mm)  
# Medium 0
theta_0 = np.radians(0.) # Angle of incidence of the electromagnetic wave (rad) 
kopt_0 = 2*np.pi/opt_wav # Optical wavenumber (rad/mm) 
omega_0 = kopt_0*c_0 # Angular frequency of the electromagnetic wave (rad/µs) 
n0 = 1. # Refractive index
mu_0 = 1. # Magnetic permeability
# Medium 1
mu_1 = 1. # Magnetic permeability
n1p = 1.468 # Real part of the complex refractive index
n1pp = 8.949 # Imaginary part of the complex refractive index (extinction coefficient)
n1 = n1p+1j*n1pp # Complex refractive index
kopt_1 = kopt_0*n1 # Optical wavenumber (rad/mm)  
# Medium 2
mu_2 = 1. # Magnetic permeability
n2p = 1.468 # Real part of the complex refractive index
n2pp = 8.949 # Imaginary part of the complex refractive index (extinction coefficient)
n2 = n2p+1j*n2pp # Complex refractive index
kopt_2 = kopt_0*n2 # Optical wavenumber (rad/mm) 
#%% Mechanical boundary conditions between medium 1 and medium 2
# Interfacial stiffnesses 1
vec_pow_kn1 = [-3.] # 10**(vec_pow_kn1): Normal interfacial stiffness (kN/mm3) 
vec_pow_kt1 = [-3.] # 10**(vec_pow_kt1): Transverse interfacial stiffness (kN/mm3) 
# Medium i1
h_i1 = 0.01 # Thickness (mm)
vpl_i1 = 1.414 # L-wave (mm/µs)
vpt_i1 = 0.463 # T-wave (mm/µs)
rho_i1 = 2.100 # Density (g/cm3)
c11_i1 = vpl_i1**2*rho_i1 # Elastic coefiicient c_11 (GPa)
c66_i1 = vpt_i1**2*rho_i1 # Elastic coefiicient c_66 (GPa)
c12_i1 = (c11_i1-2*c66_i1) # Elastic coefiicient c_12 (GPa)
c22_i1 = c11_i1 # Elastic coefiicient c_22 (GPa)
eta11_i1, eta12_i1, eta22_i1, eta66_i1 = 0.0, 0.0, 0.0, 0.0 # Viscosity (Kelvin-Voigt model)
medium_i1 = np.array([c11_i1+1j*omega*eta11_i1, c12_i1+1j*omega*eta12_i1,
                      c22_i1+1j*omega*eta22_i1, c66_i1+1j*omega*eta66_i1,
                      rho_i1, h_i1]) # i for 'intermediate layer' and '1' because it is the first sublayer
dh = h_i1 # When there are more layers, add their thicknesses here (dh = h_i1+h_i2+...)
H = h_1+h_2+dh # Total thickness (mm)
# Interfacial stiffnesses 2
vec_pow_kn2 = [5.] # 10**(vec_pow_kn2): Normal interfacial stiffness (kN/mm3)
vec_pow_kt2 = [5.] # 10**(vec_pow_kt2): Transverse interfacial stiffness (kN/mm3)
medium = np.array([medium_1, medium_2, medium_i1])
#%% Thermal boundary conditions between medium 1 and medium 2
# Medium i1
lambda1_mi1 = 0.25*1e-9 # Thermal conductivity (MW.mm-1.K-1)  
Rc_i1 = h_i1/lambda1_mi1 # Surface thermal resistance of medium i1 (mm2.K.MW-1) 
# Surface thermal resistance between medium 1 and medium 2 
Rc = Thermal_coupling(Rc_i1) # (mm2.K.MW-1) 
#%% Electromagnetic boundary conditions between medium 1 and medium 2
# Medium i1
mu_i1 = 1. # Magnetic permeability
np_i1 = 1. # Real part of the complex refractive index
npp_i1 = 0. # Imaginary part of the complex refractive index (extinction coefficient)
n_i1 = np_i1+1j*npp_i1 # Complex refractive index
EMc_i1 = Coupling_EM(theta_0, kopt_0, n_i1, mu_i1, h_i1) 
# Electromagnetic transfer matrix
EMc = ElectroMag_coupling(EMc_i1)
#%% Pulse shape and intensity
# Laser source
t_imp = 0.008 # Pulse duration (µs)
I0 = 0.2 # Incident intensity of the laser beam (J.mm-1) 
a_s = 0.1 # Width at half maximum of the Gaussian generation spot (mm)
# Interferometer parameters
a_d = 0.6 # Width at half maximum of the Gaussian detection spot (mm)
hpf = 1. # Highpass filter (MHz)
lpf = 40. # Lowpass filter (MHz)
order = 2 # Order of the filter
#%% Loop x1_vec
count = 0 # Initialization of the counter
start_total = duration.time()
for id_x1 in range(len(x1_vec)):
    start_loop_x1 = duration.time()
    x1 = x1_vec[id_x1] 
    print('x1 = {:.2f} mm ({}/{})'.format(x1, id_x1+1, len(x1_vec)))
    #%% Resolution of the electromagnetic problem 
    beta_1, beta_2, theta_1, theta_2, \
    Rf_m1, Rb_m1, Rf_m2, Rb_m2 = Coef_EM(medium, dh, n0, mu_0, theta_0, kopt_0, n1,
                                         mu_1, kopt_1, n2, mu_2, kopt_2, EMc)
    Q = Power_densities_Q(I0, beta_1, beta_2, mu_1, mu_2, n1, n2, theta_1, theta_2,
                          Rf_m1, Rb_m1, Rf_m2, Rb_m2, time, omega, k2, t_imp, x1, x2, 
                          hpf, lpf, order, a_s, a_d, h_1, h_2, dh)
    #%% Resolution of the heat equation in the Fourier domain
    beta = np.array([beta_1, beta_2]) # Inverse of the optical penetration (mm-1)
    Tp, t_gamma, Th = Coef_Temp(omega, k2, medium, Rc, beta, Q, h, T_ext, 
                                th_prop_m1, th_prop_m2, np.real(theta_1), 
                                np.real(theta_2), dh)
    gamma_m1, gamma_m2 = t_gamma # Eigenvalues
    Tpf_beta_1, Tpb_beta_1, Tpf_beta_2, Tpb_beta_2 = Tp # Particular solutions
    Thf_m1, Thb_m1_star, Thf_m2_star, Thb_m2_star = Th # Homogeneous solutions
    Tc_temporal = Temp_Complete_solution(x1, k2, time, x2, delta, medium, dh, beta,
                                         Tp, t_gamma, Th, T_ext, 
                                         np.real(theta_1), np.real(theta_2))
    #%% Resolution of the mechanical problem in the Fourier domain
    # Homogeneous solutions 
    sh_m1 = Homogeneous_solution(omega, k2, medium[0])
    sh_m2 = Homogeneous_solution(omega, k2, medium[1])
    sh_mi1 = Homogeneous_solution(omega, k2, medium[2])
    # Particular solutions medium 1 [f: forward , b: backward]
    upf_beta_1 = Particular_solution_beta(omega, k2, medium[0], th_prop_m1, 
                                          beta_1, Tpf_beta_1, np.real(theta_1)) 
    upb_beta_1 = Particular_solution_beta(omega, k2, medium[0], th_prop_m1, 
                                          -beta_1, Tpb_beta_1, -np.real(theta_1)) 
    upf_gamma_m1 = Particular_solution_gamma(omega, k2, medium[0], th_prop_m1, 
                                             gamma_m1, Thf_m1)
    upb_gamma_m1 = Particular_solution_gamma(omega, k2, medium[0], th_prop_m1, 
                                             -gamma_m1, Thb_m1_star)
    # Particular solutions medium 2 [f: forward , b: backward]
    upf_beta_2 = Particular_solution_beta(omega, k2, medium[1], th_prop_m2, beta_2,
                                          Tpf_beta_2, np.real(theta_2))
    upb_beta_2 = Particular_solution_beta(omega, k2, medium[1], th_prop_m2, -beta_2,
                                          Tpb_beta_2, -np.real(theta_2))
    upf_gamma_m2 = Particular_solution_gamma(omega, k2, medium[1], th_prop_m2,
                                             gamma_m2, Thf_m2_star)
    upb_gamma_m2 = Particular_solution_gamma(omega, k2, medium[1], th_prop_m2,
                                             -gamma_m2, Thb_m2_star)
    # Boundary conditions Particular solutions
    sigma11ps0_m1, sigma12ps0_m1, sigma11psH_m2, sigma12psH_m2, \
    u1ph1_m1, u2ph1_m1, sigma11psh1_m1, sigma12psh1_m1, \
    u1ph1dh_m2, u2ph1dh_m2, sigma11psh1dh_m2, sigma12psh1dh_m2 = \
    Boundary_conditions_Particular_solutions_Source_Term(omega, k2, medium, beta, \
                                                         th_prop_m1, th_prop_m2, t_gamma, Tp, Th, \
                                                         upf_beta_1, upb_beta_1, upf_gamma_m1, upb_gamma_m1, \
                                                         upf_beta_2, upb_beta_2, upf_gamma_m2, upb_gamma_m2, \
                                                         np.real(theta_1), np.real(theta_2), dh)
    # Boundary conditions Homogeneous solutions
    psi_h1_m1_u_star,gamma_psi_h_star,inv_gamma_psi_h_star,psi_h1dh_m2_star,psi_H_m2_star = \
    Boundary_conditions_Homogeneous_solutions(omega, k2, medium, sh_m1, sh_m2)
    #%% Loop vec_pow_kn1, vec_pow_kt1
    pow_kn2 = vec_pow_kn2[0] # 10**(vec_pow_kn2[0]): Normal interfacial stiffness (kN/mm3)
    pow_kt2 = vec_pow_kt2[0] # 10**(vec_pow_kt2[0]): Transverse interfacial stiffness (kN/mm3)
    for i_n in range(len(vec_pow_kn1)):
        for i_t in range(len(vec_pow_kt1)):
            pow_kn1 = vec_pow_kn1[i_n] # 10**(vec_pow_kn1[i_n]): Normal interfacial stiffness (kN/mm3)
            pow_kt1 = vec_pow_kt1[i_t] # 10**(vec_pow_kt2[i_t]): Transverse interfacial stiffness (kN/mm3)
            if len(vec_pow_kn1)>1 or len(vec_pow_kt1)>1:
                print('kn1 = 10^({:.1f}) kN/mm^3, kt1 = 10^({:.1f}) kN/mm^3'.format(pow_kn1, pow_kt1))       
            #%% Transfer matrix L
            # Pc = Perfect_coupling(omega)
            B = Orthotropic_layer(omega, k2, medium_i1, sh_mi1)
            K1_mat = Interfacial_layer(omega, pow_kn1, pow_kt1)
            K2_mat = Interfacial_layer(omega, pow_kn2, pow_kt2)
            L = Mechanical_coupling(K1_mat, B, K2_mat) # Mechanical transfer matrix between medium 1 and medium 2
            # Normal displacement uc1 as a function of time
            uc1, uc1_temporal = Simu_multilayer(L, time, x2, delta, x1, omega, k2, medium, dh, sh_m1, sh_m2, beta, t_gamma, 
                                           upf_beta_1, upb_beta_1, upf_gamma_m1, upb_gamma_m1, 
                                           upf_beta_2, upb_beta_2, upf_gamma_m2, upb_gamma_m2,
                                           psi_h1_m1_u_star, gamma_psi_h_star, inv_gamma_psi_h_star, psi_h1dh_m2_star,psi_H_m2_star,
                                           sigma11ps0_m1, sigma12ps0_m1, sigma11psH_m2, sigma12psH_m2,
                                           u1ph1_m1, u2ph1_m1, sigma11psh1_m1, sigma12psh1_m1,
                                           u1ph1dh_m2, u2ph1dh_m2, sigma11psh1dh_m2, sigma12psh1dh_m2,
                                           np.real(theta_1), np.real(theta_2))
            # Limit time and x2 (before saving data)
            id_tmax = np.argmin(np.abs(time-np.floor(time[-1])))
            id_xmin = np.argmin(np.abs(x2+16.))
            id_xmax = np.argmin(np.abs(x2-16.))
            # Filter
            uc1_temporal_filt = np.zeros(uc1_temporal.shape)
            uc1_temporal_filt[0:id_tmax+1,:], b, a = Butter_bandpass_filter(uc1_temporal[0:id_tmax+1,:], hpf, lpf, 1/dt, order) # Bandpass  
            # Save
            uc1_temporal_save = uc1_temporal_filt[0:id_tmax+1, id_xmin:id_xmax+1]
            x2_save = x2[id_xmin:id_xmax+1]    
            time_save = time[0:id_tmax+1]             
            path_simu = os.getcwd()+'/'+id_save+'/data_simu//'  
            if not os.path.exists(path_simu):
                os.makedirs(path_simu)
            # Save uc1_temporal_save
            file_inp = id_save+'_kn'+str(pow_kn1)+'_kt'+str(pow_kt1)
            np.savez(path_simu+file_inp, time=time_save, position=x2_save, front=uc1_temporal_save, 
                     pow_kn_12=pow_kn1, pow_kt_12=pow_kt1, a_s=a_s, a_d=a_d, t_imp=t_imp)
            # Save uc1 (Fourier domain)
            file_inp_Fourier = id_save+'_Fourier_kn'+str(pow_kn1)+'_kt'+str(pow_kt1)
            np.savez(path_simu+file_inp_Fourier, omega=omega, k2=k2, front=uc1, 
                     pow_kn_12=pow_kn1, pow_kt_12=pow_kt1, a_s=a_s, a_d=a_d, t_imp=t_imp)
            
    end_loop_x1 = duration.time()
    print('Duration of the iteration = {:.1f} s'.format(end_loop_x1-start_loop_x1))
end_total = duration.time()
print('--- Total duration = {:.1f} s ---'.format(end_total-start_total))
#%% Open results
x1_plot = x1_vec[0] # Select x1-position 
Open_results(path_simu, id_save, x1_plot, pow_kn1, pow_kt1, dt, dx2, hpf, lpf, order)
