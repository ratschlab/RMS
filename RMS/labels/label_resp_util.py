''' Label functions'''

import sys
import os
import os.path
import ipdb
import datetime
import timeit
import random
import gc
import psutil
import csv
import glob

import pandas as pd
import numpy as np
import scipy as sp

import matplotlib
matplotlib.use("pdf")

import matplotlib.pyplot as plt


IMPUTE_GRID_PERIOD_SECS=300.0


def future_worse_state_from_0_or_1( endpoint_status_arr,l_hours, r_hours,grid_step_secs):
    ''' Deterioration to level 2 from level 0 or 1'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]
        if np.isnan(e_val) or e_val>=2:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 0/1: Stability
        if e_val==0 or e_val==1:
            if (future_arr==2.0).any() or (future_arr==3.0).any():
                out_arr[idx]=1.0

    return out_arr




def future_worse_state_from_1(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Deterioration to level 2/3 from level 1'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if np.isnan(e_val) or e_val==0 or e_val>=2:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 1: Moderate stability
        if e_val==1:
            if (future_arr==2.0).any() or (future_arr==3.0).any():
                out_arr[idx]=1.0

    return out_arr


def future_worse_state_from_0(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Deterioration to level 1/2/3 from level 0'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if np.isnan(e_val) or e_val>=1:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 0: Stability
        if e_val==0:
            if (future_arr==1.0).any() or (future_arr==2.0).any() or (future_arr==3.0).any():
                out_arr[idx]=1.0

    return out_arr




def future_worse_state_from_2(endpoint_status_arr, l_hours, r_hours, grid_step_secs):
    ''' Deterioration to level 3 from level 2'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(endpoint_status_arr.size)
    sz=endpoint_status_arr.size
    
    for idx in range(endpoint_status_arr.size):
        e_val=endpoint_status_arr[idx]

        if np.isnan(e_val) or e_val<=1 or e_val>=3:
            out_arr[idx]=np.nan
            continue

        future_arr=endpoint_status_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no deterioration
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # State 2: Moderate respiratory failure
        if e_val==2:
            if (future_arr==3.0).any():
                out_arr[idx]=1.0

    return out_arr



def future_ventilation(vent_period_arr, l_hours, r_hours, grid_step_secs):
    ''' Ventilation onset from no ventilation, exclude from training/evaluation
        time-points which are in the 30 mins prior to ventilation onset to prevent
        future leaks conservatively.'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(vent_period_arr.size)
    sz=vent_period_arr.size
    
    for idx in range(vent_period_arr.size):
        e_val=vent_period_arr[idx]

        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue

        future_arr=vent_period_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no change
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # No ventilation
        if e_val==0:
            if (future_arr==1.0).any():
                first_idx=np.where(future_arr==1)[0].min()
                if first_idx>=6:
                    out_arr[idx]=1.0
                else:
                    out_arr[idx]=np.nan

    return out_arr


def future_ventilation_resource(vent_period_arr, l_hours, r_hours, grid_step_secs):
    ''' Resource planning ventilation task, predict ventilation both when the patient is already 
        ventilated as well as predict ventilation if the patient is not yet ventilated
        but will be in the near future'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(vent_period_arr.size)
    sz=vent_period_arr.size
    
    for idx in range(vent_period_arr.size):
        e_val=vent_period_arr[idx]

        future_arr=vent_period_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no change
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        if (future_arr==1.0).any():
            first_idx=np.where(future_arr==1)[0].min()
            if l_hours>0 or first_idx>=6:
                out_arr[idx]=1.0
            else:
                out_arr[idx]=np.nan

    return out_arr


def future_ventilation_mc(vent_period_arr,grid_step_secs):
    ''' Ventilation onset from no ventilation, exclude from training/evaluation
        time-points which are in the 30 mins prior to ventilation onset to prevent
        future leaks conservatively. Multiclass version with the 3 classes [0.5,8] hours = 1,
        [8,24] =2 hours, or not in 24 hours = 0'''
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(vent_period_arr.size)
    sz=vent_period_arr.size
    
    for idx in range(vent_period_arr.size):
        e_val=vent_period_arr[idx]

        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue

        future_arr=vent_period_arr[idx: min(sz, idx+int(gridstep_per_hours*24))]

        # No future to consider, => no change
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # No ventilation
        if e_val==0:
            if (future_arr==1.0).any():
                first_idx=np.where(future_arr==1)[0].min()
                if first_idx>=6 and first_idx<=8*12:
                    out_arr[idx]=1.0
                elif first_idx>8*12:
                    out_arr[idx]=2.0
                else:
                    out_arr[idx]=np.nan

    return out_arr


def future_ready_ext(ready_ext_arr, l_hours, r_hours, grid_step_secs):
    ''' Ready to be extubated from ventilation period'''
    assert(r_hours>=l_hours)
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(ready_ext_arr.size)
    sz=ready_ext_arr.size
    
    for idx in range(ready_ext_arr.size):
        e_val=ready_ext_arr[idx]

        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue

        future_arr=ready_ext_arr[min(sz, idx+int(gridstep_per_hours*l_hours)): min(sz, idx+int(gridstep_per_hours*r_hours))]

        # No future to consider, => no change
        if future_arr.size==0:
            continue

        # Future has only NANs for some event
        elif np.isnan(future_arr).all():
            out_arr[idx]=np.nan
            continue

        # Ventilation but not ready yet.
        if e_val==0:
            if (future_arr==1.0).any():
                out_arr[idx]=1.0

    return out_arr


def future_ready_ext_mc(ready_ext_arr, grid_step_secs):
    ''' Ready to be extubated from ventilation period with multi-task label 
        for the categories [0,8] hours := 1, [8,24] hours := 2, > 24 hours := 0'''
    gridstep_per_hours=int(3600/grid_step_secs)
    out_arr=np.zeros(ready_ext_arr.size)
    sz=ready_ext_arr.size
    
    for idx in range(ready_ext_arr.size):
        e_val=ready_ext_arr[idx]

        if np.isnan(e_val) or e_val==1:
            out_arr[idx]=np.nan
            continue

        future_arr_cat_1=ready_ext_arr[idx:min(sz, idx+int(gridstep_per_hours*8))]
        future_arr_cat_2=ready_ext_arr[min(sz, idx+int(gridstep_per_hours*8)):min(sz,idx+int(gridstep_per_hours*24))]

        if future_arr_cat_1.size==0:
            continue
        
        if np.isnan(future_arr_cat_1).all():
            out_arr[idx]=np.nan
            continue

        # Ventilation but not ready yet.
        if e_val==0:
            if (future_arr_cat_1==1.0).any():
                out_arr[idx]=1.0
            elif (future_arr_cat_2==1.0).any():
                out_arr[idx]=2.0

    return out_arr









