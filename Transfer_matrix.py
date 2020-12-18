# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2020

Laser ultrasonics: Semi-analytical model

@author: Romain HODE
"""

from Dot_prod import dot4444

def ElectroMag_coupling(*N_mat):
    
    if len(N_mat)==1:
        EMc = N_mat[0]
    else:
        # Initialization
        EMc = dot4444(N_mat[0], N_mat[1])
        # Loop
        for ii in range(len(N_mat)-2):
            EMc = dot4444(EMc, N_mat[ii+2])
    
    return EMc  

def Thermal_coupling(*N_mat):
    
    if len(N_mat)==1:
        Rc = N_mat[0]
    else:
        # Initialization
        Rc = N_mat[0]+N_mat[1]
        # Loop
        for ii in range(len(N_mat)-2):
            Rc = Rc+N_mat[ii+2]
    
    return Rc  

def Mechanical_coupling(*N_mat):
    
    if len(N_mat)==1:
        L = N_mat[0]
    else:
        # Initialization
        L = dot4444(N_mat[0], N_mat[1])
        # Loop
        for ii in range(len(N_mat)-2):
            L = dot4444(L, N_mat[ii+2])
    
    return L      
