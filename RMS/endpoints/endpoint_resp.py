''' Functions to generate the respiratory failure endpoint'''

import argparse
import os
import os.path
import glob
import ipdb
import sys
import random
import math
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import skfda.preprocessing.smoothing.kernel_smoothers as skks
import skfda.representation.grid as skgrid
import scipy.stats as spstats
import sklearn.linear_model as sklm
import sklearn.metrics as skmetrics
import lightgbm as lgbm
import sklearn.preprocessing as skpproc

from tensorflow import keras
import tensorflow.keras.backend as K

SUPPOX_TO_FIO2={
    0: 21,
    1: 26,
    2: 34,
    3: 39,
    4: 45,
    5: 49,
    6: 54,
    7: 57,
    8: 58,
    9: 63,
    10: 66,
    11: 67,
    12: 69,
    13: 70,
    14: 73,
    15: 75 }

def mix_real_est_pao2(pao2_col, pao2_meas_cnt, pao2_est_arr, bandwidth=None):
    ''' Mix real PaO2 measurement and PaO2 estimates using a Gaussian kernel'''
    final_pao2_arr=np.copy(pao2_est_arr)
    sq_scale=57**2 # 1 hour has mass 1/3 approximately

    for idx in range(final_pao2_arr.size):
        meas_ref=pao2_meas_cnt[idx]
        real_val=None
        real_val_dist=None

        # Search forward and backward with priority giving to backward if equi-distant
        for sidx in range(48):
            if not idx-sidx<0 and pao2_meas_cnt[idx-sidx]<meas_ref:
                real_val=pao2_col[idx-sidx+1]
                real_val_dist=5*sidx
                break
            elif not idx+sidx>=final_pao2_arr.size and pao2_meas_cnt[idx+sidx]>meas_ref:
                real_val=pao2_col[idx+sidx]
                real_val_dist=5*sidx
                break

        if real_val is not None:
            alpha_mj=math.exp(-real_val_dist**2/sq_scale)
            alpha_ej=1-alpha_mj
            final_pao2_arr[idx]=alpha_mj*real_val+alpha_ej*pao2_est_arr[idx]
            
    return final_pao2_arr


def perf_regression_model(X_list, y_list, aux_list, configs=None):
    ''' Initial test of a regression model to estimate the current Pao2 based
        on 6 features of the past. Also pass FiO2 to calculate resulting mistakes in
        the P/F ratio'''

    print("Testing regression model for PaO2...")
    
    # Partition the data into 3 sets and run SGD regressor
    X_train=X_list[:int(0.6*len(X_list))]
    X_train=np.vstack(X_train)
    y_train=np.concatenate(y_list[:int(0.6*len(y_list))])
    X_val=X_list[int(0.6*len(X_list)):int(0.8*len(X_list))]
    X_val=np.vstack(X_val)
    y_val=np.concatenate(y_list[int(0.6*len(y_list)):int(0.8*len(y_list))])
    X_test=X_list[int(0.8*len(X_list)):]
    X_test=np.vstack(X_test)
    y_test=np.concatenate(y_list[int(0.8*len(y_list)):])

    fio2_test=np.concatenate(aux_list[int(0.8*len(aux_list)):])

    if configs["sur_model_type"]=="linear":
        scaler=skpproc.StandardScaler()
        X_train_std=scaler.fit_transform(X_train)
        X_val_std=scaler.transform(X_val)
        X_test_std=scaler.transform(X_test)

    if configs["sur_model_type"]=="linear":
        alpha_cands=[0.0001,0.001,0.01,0.1,1.0]
    elif configs["sur_model_type"]=="lgbm":
        alpha_cands=[32]

    best_alpha=None
    best_score=np.inf

    # Search for the best model on the validation set
    for alpha in alpha_cands:
        print("Testing alpha: {}".format(alpha))

        if configs["sur_model_type"]=="linear":
            lmodel_cand=sklm.SGDRegressor(alpha=alpha, random_state=2021)
        elif configs["sur_model_type"]=="lgbm":
            lmodel_cand=lgbm.LGBMRegressor(num_leaves=alpha, learning_rate=0.05, n_estimators=1000,
                                           random_state=2021)

        if configs["sur_model_type"]=="linear":
            lmodel_cand.fit(X_train_std,y_train)
        elif configs["sur_model_type"]=="lgbm":
            lmodel_cand.fit(X_train_std,y_train, eval_set=(X_val_std,y_val), early_stopping_rounds=20,
                            eval_metric="mae")
            
        pred_y_val=lmodel_cand.predict(X_val_std)
        mae_val=np.median(np.absolute(y_val-pred_y_val))
        if mae_val<best_score:
            best_score=mae_val
            best_alpha=alpha

    lmodel=sklm.SGDRegressor(alpha=best_alpha,random_state=2021)
    lmodel.fit(X_train_std,y_train)
    pred_y_test=lmodel.predict(X_test_std)

    pred_pf_ratio_test=pred_y_test/fio2_test
    true_pf_ratio_test=y_test/fio2_test

    mae_test=skmetrics.mean_absolute_error(y_test, pred_y_test)
    print("Mean absolute error in test set: {:.3f}".format(mae_test))

def percentile_smooth(signal_col, percentile, win_scope_mins):
    ''' Window percentile smoother, where percentile is in the interval [0,100]'''
    out_arr=np.zeros_like(signal_col)
    mins_per_window=5
    search_range=int(win_scope_mins/mins_per_window/2)
    for jdx in range(out_arr.size):
        search_arr=signal_col[max(0,jdx-search_range):min(out_arr.size,jdx+search_range)]
        out_arr[jdx]=np.percentile(search_arr,percentile)
    return out_arr


def subsample_blocked(val_arr, meas_arr=None, ss_ratio=None, block_length=None, normal_value=None):
    ''' Subsample blocked with ratio and block length'''
    val_arr_out=np.copy(val_arr)
    meas_arr_out=np.copy(meas_arr)
    meas_idxs=[]
    n_meas=0
    
    for idx in range(meas_arr.size):
        if meas_arr[idx]>n_meas:
            meas_idxs.append(idx)
            n_meas+=1

    if len(meas_idxs)==0:
        return (val_arr_out, meas_arr_out)
            
    meas_select=int((1-ss_ratio)*len(meas_idxs))
    begin_select=meas_select//block_length
    feas_begins=[meas_idxs[idx] for idx in np.arange(0,len(meas_idxs),block_length)]
    sel_meas_begins=sorted(random.sample(feas_begins, begin_select))
    sel_meas_delete=[]
    for begin in sel_meas_begins:
        for add_idx in range(block_length):
            sel_meas_delete.append(begin+add_idx)

    # Rewrite the measuremnent array with deleted indices
    for midx,meas_idx in enumerate(meas_idxs):
        prev_cnt=0 if meas_idx==0 else meas_arr_out[meas_idx-1]
        revised_cnt=prev_cnt if meas_idx in sel_meas_delete else prev_cnt+1
        if midx<len(meas_idxs)-1:
            for rewrite_idx in range(meas_idx,meas_idxs[midx+1]):
                meas_arr_out[rewrite_idx]=revised_cnt
        else:
            for rewrite_idx in range(meas_idx,len(meas_arr_out)):
                meas_arr_out[rewrite_idx]=revised_cnt

    # Rewrite the value array with deleted indices, with assuming forward filling
    for midx,meas_idx in enumerate(meas_idxs):
        prev_val=normal_value if meas_idx==0 else val_arr_out[meas_idx-1]
        cur_val=val_arr_out[meas_idx]
        revised_value=prev_val if meas_idx in sel_meas_delete else cur_val
        if midx<len(meas_idxs)-1:
            for rewrite_idx in range(meas_idx,meas_idxs[midx+1]):
                val_arr_out[rewrite_idx]=revised_value
        else:
            for rewrite_idx in range(meas_idx,len(meas_arr_out)):
                val_arr_out[rewrite_idx]=revised_value

    return (val_arr_out, meas_arr_out)    

def subsample_individual(val_arr, meas_arr=None, ss_ratio=None, normal_value=None):
    ''' Subsample individual measurements completely randomly with random choice'''
    val_arr_out=np.copy(val_arr)
    meas_arr_out=np.copy(meas_arr)
    meas_idxs=[]
    n_meas=0
    
    for idx in range(meas_arr.size):
        if meas_arr[idx]>n_meas:
            meas_idxs.append(idx)
            n_meas+=1

    if len(meas_idxs)==0:
        return (val_arr_out, meas_arr_out)
            
    meas_select=int((1-ss_ratio)*len(meas_idxs))
    sel_meas_delete=sorted(random.sample(meas_idxs, meas_select))

    # Rewrite the measuremnent array with deleted indices
    for midx,meas_idx in enumerate(meas_idxs):
        prev_cnt=0 if meas_idx==0 else meas_arr_out[meas_idx-1]
        revised_cnt=prev_cnt if meas_idx in sel_meas_delete else prev_cnt+1
        if midx<len(meas_idxs)-1:
            for rewrite_idx in range(meas_idx,meas_idxs[midx+1]):
                meas_arr_out[rewrite_idx]=revised_cnt
        else:
            for rewrite_idx in range(meas_idx,len(meas_arr_out)):
                meas_arr_out[rewrite_idx]=revised_cnt

    # Rewrite the value array with deleted indices, with assuming forward filling
    for midx,meas_idx in enumerate(meas_idxs):
        prev_val=normal_value if meas_idx==0 else val_arr_out[meas_idx-1]
        cur_val=val_arr_out[meas_idx]
        revised_value=prev_val if meas_idx in sel_meas_delete else cur_val
        if midx<len(meas_idxs)-1:
            for rewrite_idx in range(meas_idx,meas_idxs[midx+1]):
                val_arr_out[rewrite_idx]=revised_value
        else:
            for rewrite_idx in range(meas_idx,len(meas_arr_out)):
                val_arr_out[rewrite_idx]=revised_value

    return (val_arr_out, meas_arr_out)
    
def lookup_admission_time(pid, df_patient_full):
    ''' Looks up a proxy to admission time for a PID'''
    df_patient=df_patient_full[df_patient_full["PatientID"]==pid]

    if not df_patient.shape[0]==1:
        return None
        
    adm_time=np.array(df_patient["AdmissionTime"])[0]
    return adm_time


def merge_short_vent_gaps(vent_status_arr, short_gap_hours):
    ''' Merge short gaps in the ventilation status array'''
    in_gap=False
    gap_length=0
    before_gap_status=np.nan
    
    for idx in range(len(vent_status_arr)):
        cur_state=vent_status_arr[idx]
        if in_gap and (cur_state==0.0 or np.isnan(cur_state)):
            gap_length+=5
        elif not in_gap and (cur_state==0.0 or np.isnan(cur_state)):
            if idx>0:
                before_gap_status=vent_status_arr[idx-1]
            in_gap=True
            in_gap_idx=idx
            gap_length=5
        elif in_gap and cur_state==1.0:
            in_gap=False
            after_gap_status=cur_state
            if gap_length/60.<=short_gap_hours:
                vent_status_arr[in_gap_idx:idx]=1.0
    
    return vent_status_arr


def kernel_smooth_arr(input_arr, bandwidth=None):
    ''' Kernel smooth an input array with a Nadaraya-Watson kernel smoother'''
    output_arr=np.copy(input_arr)
    fin_arr=output_arr[np.isfinite(output_arr)]
    time_axis=5*np.arange(len(output_arr))
    fin_time=time_axis[np.isfinite(output_arr)]

    # Return the unsmoothed array if fewer than 2 observations
    if fin_arr.size<2:
        return output_arr
        
    smoother=skks.NadarayaWatsonSmoother(smoothing_parameter=bandwidth)
    fgrid=skgrid.FDataGrid([fin_arr], fin_time)
    fd_smoothed=smoother.fit_transform(fgrid)
    output_smoothed=fd_smoothed.data_matrix.flatten()
    output_arr[np.isfinite(output_arr)]=output_smoothed
    return output_arr

def delete_short_vent_events(vent_status_arr, short_event_hours):
    ''' Delete short events in the ventilation status array'''
    in_event=False
    event_length=0
    for idx in range(len(vent_status_arr)):
        cur_state=vent_status_arr[idx]
        if in_event and cur_state==1.0:
            event_length+=5
        if not in_event and cur_state==1.0:
            in_event=True
            event_length=5
            event_start_idx=idx
        if in_event and (cur_state==0.0 or np.isnan(cur_state)):
            in_event=False
            if event_length/60.<short_event_hours:
                vent_status_arr[event_start_idx:idx]=0.0
    return vent_status_arr



def correct_left_edge_vent(vent_status_arr, etco2_meas_cnt, etco2_col):
    ''' Corrects the left edge of the ventilation status array, to pin-point the exact conditions'''
    on_left_edge=False
    in_event=False

    # Correct left ventilation edges of the ventilation zone
    for idx in range(len(vent_status_arr)):
        if vent_status_arr[idx]==1.0 and not in_event:
            in_event=True
            on_left_edge=True
        if on_left_edge and in_event:
            if vent_status_arr[idx]==0.0:
                in_event=False
                on_left_edge=False
            elif (idx==0 and etco2_meas_cnt[idx]>0 or etco2_meas_cnt[idx]-etco2_meas_cnt[idx-1]>=1) and etco2_col[idx]>0.5:
                on_left_edge=False
            else:
                vent_status_arr[idx]=0.0

    return vent_status_arr


def correct_right_edge_vent(vent_status_arr, etco2_meas_cnt, etco2_col):
    ''' Corrects the right edge of the ventilation status array to pin-point the exact conditions'''
    on_right_edge=False
    in_event=False

    # Correct right edges of the ventilation zone
    for idx in range(len(event_status_arr)):
        if vent_status_arr[idx]==1.0 and not in_event:
            in_event=True
        elif in_event and vent_status_arr[idx]==0.0:
            in_event=False
            on_right_edge=True
        if on_right_edge:
            if etco2_col[idx]>0.5 and (idx==0 and etco2_meas_cnt[idx]>0 or etco2_meas_cnt[idx]-etco2_meas_cnt[idx-1]>=1):
                vent_status_arr[idx]=1.0
            else:
                on_right_edge=False

    return vent_status_arr


def delete_small_continuous_blocks(event_arr,block_threshold=None):
    ''' Given an event array, deletes small contiguous blocks that are sandwiched between two other blocks, one of which
        is longer, they both have the same label. For the moment we delete blocks smaller than 30 minutes. Note this
        requires only a linear pass over the array'''
    block_list=[]
    active_block=None

    # Build a block list
    for jdx in range(event_arr.size):
        new_block=event_arr[jdx]

        # Start a new block at the beginning
        if active_block is None:
            active_block=new_block
            left_block_idx=jdx

        # Change to a new block
        elif not active_block==new_block:
            block_list.append((active_block,left_block_idx,jdx-1))
            left_block_idx=jdx
            active_block=new_block

        # Same last block unconditionally
        if jdx==event_arr.size-1:
            block_list.append((new_block,left_block_idx,jdx))

    # Merge blocks

    while True:

        sorted_block_list=sorted(block_list, key=lambda block: block[2]-block[1])
        all_clean=True
        
        for block in sorted_block_list:
            bidx=block_list.index(block)
            block_label,lidx,ridx=block
            block_len=ridx-lidx+1

            # Candidate for merging
            if block_len<=block_threshold:

                if len(block_list)==1:
                    all_clean=True
                    break
                
                # Only right block
                elif bidx==0:
                    next_block=block_list[bidx+1]
                    nb_label,nb_lidx,nb_ridx=next_block
                    nb_len=nb_ridx-nb_lidx+1

                    # Merge blocks
                    if nb_len>block_len and nb_len>block_threshold:
                        block_list[bidx]=(nb_label,lidx,nb_ridx)
                        block_list.remove(next_block)
                        all_clean=False
                        break

                # Only left block
                elif bidx==len(block_list)-1:
                    prev_block=block_list[bidx-1]
                    pb_label,pb_lidx,pb_ridx=prev_block
                    pb_len=pb_ridx-pb_lidx+1

                    if pb_len>block_len and pb_len>block_threshold:
                        block_list[bidx]=(pb_label,pb_lidx,ridx)
                        block_list.remove(prev_block)
                        all_clean=False
                        break

                # Interior block
                else:
                    prev_block=block_list[bidx-1]
                    next_block=block_list[bidx+1]
                    pb_label,pb_lidx,pb_ridx=prev_block
                    nb_label,nb_lidx,nb_ridx=next_block
                    pb_len=pb_ridx-pb_lidx+1
                    nb_len=nb_ridx-nb_lidx+1

                    if pb_label==nb_label and (pb_len>block_threshold or nb_len>block_threshold):
                        block_list[bidx]=(pb_label,pb_lidx,nb_ridx)
                        block_list.remove(prev_block)
                        block_list.remove(next_block)
                        all_clean=False
                        break
                    
        # Traversed block list with no actions required
        if all_clean:
            break

    # Now back-translate the block list to the list
    out_arr=np.copy(event_arr)

    for blabel,lidx,ridx in block_list:
        out_arr[lidx:ridx+1]=blabel

    # Additionally build an array where the two arrays are different
    diff_arr=(out_arr!=event_arr).astype(np.bool)

    return (out_arr,diff_arr)


def collect_regression_data(spo2_col, spo2_meas_cnt, pao2_col, pao2_meas_cnt, fio2_est_arr,
                            sao2_col, sao2_meas_cnt, ph_col, ph_meas_cnt ):
    ''' Collect regression data at time-stamps where we have a real PaO2 measurement, return
        partial training X,y pairs for this patient'''
    X_arr_collect=[]
    y_arr_collect=[]
    aux_collect=[]
    cur_pao2_cnt=0
    cur_spo2_cnt=0
    cur_sao2_cnt=0
    cur_ph_cnt=0
    pao2_real_meas=[]
    spo2_real_meas=[]
    sao2_real_meas=[]
    ph_real_meas=[]

    for jdx in range(spo2_col.size):
        
        if spo2_meas_cnt[jdx]>cur_spo2_cnt:
            spo2_real_meas.append(jdx)
            cur_spo2_cnt=spo2_meas_cnt[jdx]
        if sao2_meas_cnt[jdx]>cur_sao2_cnt:
            sao2_real_meas.append(jdx)
            cur_sao2_cnt=sao2_meas_cnt[jdx]
        if ph_meas_cnt[jdx]>cur_ph_cnt:
            ph_real_meas.append(jdx)
            cur_ph_cnt=ph_meas_cnt[jdx]
            
        if pao2_meas_cnt[jdx]>cur_pao2_cnt:
            pao2_real_meas.append(jdx)
            cur_pao2_cnt=pao2_meas_cnt[jdx]

            # Only start fitting the model from the 2nd measurement onwards
            if len(pao2_real_meas)>=2 and len(spo2_real_meas)>=2 and len(sao2_real_meas)>=2 and len(ph_real_meas)>=2:

                # Dimensions of features
                # 0: Last real SpO2 measurement
                # 1: Last real PaO2 measurement
                # 2: Last real SaO2 measurement
                # 3: Last real pH measurement
                # 4: Time to last real SpO2 measurement
                # 5: Time to last real PaO2 measurement
                # 6: Closest SpO2 to last real PaO2 measurement
                x_vect=np.array([spo2_col[jdx-1], pao2_col[jdx-1], sao2_col[jdx-1], ph_col[jdx-1],
                                 jdx-spo2_real_meas[-2], jdx-pao2_real_meas[-2],spo2_col[pao2_real_meas[-2]]])
                y_val=pao2_col[jdx]
                aux_val=fio2_est_arr[jdx]

                if np.isnan(x_vect).sum()==0 and np.isfinite(y_val) and np.isfinite(aux_val):
                    X_arr_collect.append(x_vect)
                    y_arr_collect.append(y_val)
                    aux_collect.append(aux_val)

    if len(X_arr_collect)>0:
        X_arr=np.vstack(X_arr_collect)
        y_arr=np.array(y_arr_collect)
        aux_arr=np.array(aux_collect)
        assert(np.isnan(X_arr).sum()==0 and np.isnan(y_arr).sum()==0)
        return (X_arr,y_arr,aux_arr)
    else:
        return (None,None,None)


def apply_regression_model(spo2_col, spo2_meas_cnt, pao2_col, pao2_meas_cnt,
                           sao2_col, sao2_meas_cnt, ph_col, ph_meas_cnt,
                           sur_base_model, sur_meta_model, sur_base_scaler, sur_meta_scaler):
    ''' Apply regression model at all times.'''
    cur_pao2_cnt=0
    cur_spo2_cnt=0
    cur_sao2_cnt=0
    cur_ph_cnt=0
    pao2_real_meas=[]
    spo2_real_meas=[]
    sao2_real_meas=[]
    ph_real_meas=[]
    mistakes=[0,0,0,0,0,0,0,0,0,0]
    poly_base=skpproc.PolynomialFeatures(degree=3)

    est_output=np.zeros_like(spo2_col)
    est_base_output=np.zeros_like(spo2_col)

    for jdx in range(spo2_col.size):

        # Keep track of SpO2/PaO2/SaO2/PH measurements
        if spo2_meas_cnt[jdx]>cur_spo2_cnt:
            spo2_real_meas.append(jdx)
            cur_spo2_cnt=spo2_meas_cnt[jdx]
        if sao2_meas_cnt[jdx]>cur_sao2_cnt:
            sao2_real_meas.append(jdx)
            cur_sao2_cnt=sao2_meas_cnt[jdx]
        if ph_meas_cnt[jdx]>cur_ph_cnt:
            ph_real_meas.append(jdx)
            cur_ph_cnt=ph_meas_cnt[jdx]
        if pao2_meas_cnt[jdx]>cur_pao2_cnt:
            pao2_real_meas.append(jdx)
            cur_pao2_cnt=pao2_meas_cnt[jdx]

            if len(pao2_real_meas)>=2 and len(spo2_real_meas)>=2 and len(sao2_real_meas)>=2 and len(ph_real_meas)>=2:
                x_vect=np.array([spo2_col[jdx-1], pao2_col[jdx-1], sao2_col[jdx-1], ph_col[jdx-1],
                                 jdx-spo2_real_meas[-2], jdx-pao2_real_meas[-2],spo2_col[pao2_real_meas[-2]]])
                x_vect=x_vect.reshape((1,x_vect.size))
                x_vect=poly_base.fit_transform(x_vect)
                x_vect=sur_base_scaler.transform(x_vect)

                if np.sum(np.isnan(x_vect))==0:
                    tent_pred=float(sur_base_model.predict(x_vect))
                    mistakes.append(tent_pred-pao2_col[jdx])

        # Only start doing real predictions from the first Pao2 onwards
        if len(pao2_real_meas)>=1 and len(spo2_real_meas)>=1 and len(sao2_real_meas)>=1 and len(ph_real_meas)>=1:

            # Dimensions of features
            # 0: Last real SpO2 measurement
            # 1: Last real PaO2 measurement
            # 2: Last real SaO2 measurement
            # 3: Last real pH measurement
            # 4: Time to last real SpO2 measurement
            # 5: Time to last real PaO2 measurement
            # 6: Closest SpO2 to last real PaO2 measurement            
            x_vect=np.array([spo2_col[jdx], pao2_col[jdx], sao2_col[jdx], ph_col[jdx],
                             jdx-spo2_real_meas[-1], jdx-pao2_real_meas[-1],spo2_col[pao2_real_meas[-1]]])

            if np.isnan(x_vect).sum()==0:
                x_vect=x_vect.reshape((1,x_vect.size))
                x_vect=poly_base.fit_transform(x_vect)
                x_vect_untf=np.copy(x_vect).flatten()
                x_vect=sur_base_scaler.transform(x_vect)
                base_pred=float(sur_base_model.predict(x_vect))
                x_meta_vect=np.concatenate([x_vect_untf, np.array([base_pred]+mistakes[-10:])])
                x_meta_vect=x_meta_vect.reshape((1,x_meta_vect.size))
                x_meta_vect=sur_meta_scaler.transform(x_meta_vect)
                final_pred=float(sur_meta_model.predict(x_meta_vect))
                est_output[jdx]=final_pred
                est_base_output[jdx]=base_pred

            # Input error
            else:
                est_output[jdx]=pao2_col[jdx]
                est_base_output[jdx]=pao2_col[jdx]

        # Just use the current estimate as prediction
        else:
            est_output[jdx]=pao2_col[jdx]
            est_base_output[jdx]=pao2_col[jdx]
            
    return (est_output,est_base_output)

    
def delete_low_density_hr_gap(vent_status_arr, hr_status_arr, configs=None):
    ''' Deletes gaps in ventilation which are caused by likely sensor dis-connections'''
    in_event=False
    in_gap=False
    gap_idx=-1
    for idx in range(len(vent_status_arr)):

        # Beginning of new event, not from inside gap
        if not in_event and not in_gap and vent_status_arr[idx]==1.0:
            in_event=True

        # Beginning of potential gap that needs to be closed
        elif in_event and vent_status_arr[idx]==0.0:
            in_gap=True
            gap_idx=idx
            in_event=False

        # The gap is over, re-assign the status of ventilation to merge the gap, enter new event
        if in_gap and vent_status_arr[idx]==1.0:
            
            hr_sub_arr=hr_status_arr[gap_idx:idx]

            # Close the gap if the density of HR is too low in between
            if np.sum(hr_sub_arr)/hr_sub_arr.size<=configs["vent_hr_density_threshold"]:
                vent_status_arr[gap_idx:idx]=1.0
                
            in_gap=False
            in_event=True

    return vent_status_arr



def suppox_to_fio2(suppox_val):
    ''' Conversion of supplemental oxygen to FiO2 estimated value'''
    if suppox_val>15:
        return 75
    else:
        return SUPPOX_TO_FIO2[suppox_val]

def conservative_state(state1,state2):
    ''' Given two states, return the lower one '''
    if state1==state2:
        return state1
    for skey in ["event_0","event_1","event_2"]:
        if state1==skey or state2==skey:
            return skey
    return "event_3"

def endpoint_gen_resp(configs):
    var_map=configs["VAR_IDS"]
    raw_var_map=configs["RAW_VAR_IDS"]
    sz_window=configs["length_fw_window"]
    abga_window=configs["length_ABGA_window"]

    # Load normalization values

    # Threshold statistics
    stat_counts_ready_and_failure=0
    stat_counts_ready_and_success=0
    stat_counts_nready_and_failure=0
    stat_counts_nready_and_success=0
    stat_counts_ready_nextube=0
    stat_counts_nready_nextube=0

    # Load surrogate PaO2 regression model
    with open(configs["sur_reg_model_path"],'rb') as fp:
        model_dict=pickle.load(fp)
        sur_reg_base_model=model_dict["reg_base_model"]
        sur_meta_model=model_dict["meta_model"]
        sur_base_scaler=model_dict["base_scaler"]
        sur_meta_scaler=model_dict["meta_scaler"]

    imputed_f=os.path.join(configs["imputed_path"],configs["split"])
    merged_f=os.path.join(configs["merged_h5"])

    if configs["reliability_analysis"]:
        out_folder=os.path.join(configs["endpoint_path"],configs["split"], "seed_{}".format(configs["random_state"]))
    else:
        out_folder=os.path.join(configs["endpoint_path"],configs["split"])

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    batch_id=configs["batch_idx"]
    
    print("Generating endpoints for batch {}".format(batch_id),flush=True)
    batch_fpath=os.path.join(imputed_f,"batch_{}.h5".format(batch_id))

    if not os.path.exists(batch_fpath):
        print("WARNING: Input file does not exist, exiting...")
        sys.exit(0)

        
    df_batch=pd.read_hdf(os.path.join(imputed_f,"batch_{}.h5".format(batch_id)),mode='r') 

    hr_gap_detected=0
    
    print("Loaded imputed data done...")

    if configs["endpoint"]=="resp_extval":
        cand_raw_batch=[sorted(glob.glob(os.path.join(merged_f,"merged_*.parquet")),key=lambda fpath: int(fpath.split("/")[-1].split("_")[1]))[batch_id]]
    else:
        cand_raw_batch=glob.glob(os.path.join(merged_f,"reduced_rm_drugoor","reduced_fmat_{}_*.h5".format(batch_id)))
        
    assert(len(cand_raw_batch)==1)
    pids=list(df_batch.PatientID.unique())
    
    print("Number of patients in batch: {}".format(len(df_batch.PatientID.unique())),flush=True)
    first_write=True
    out_fp=os.path.join(out_folder,"batch_{}.h5".format(batch_id))

    event_count={"FIO2_AVAILABLE": 0, "SUPPOX_NO_MEAS_12_HOURS_LIMIT": 0, "SUPPOX_MAIN_VAR": 0, "SUPPOX_HIGH_FLOW": 0,
                 "SUPPOX_NO_FILL_STOP": 0}
    readiness_ext_count=0
    not_ready_ext_count=0
    readiness_and_extubated_cnt=0
    extubated_cnt=0

    if configs["endpoint"]=="resp_extval":
        df_static=pd.read_parquet(configs["general_data_table_path"])
    else:
        df_static=pd.read_hdf(configs["general_data_table_path"],mode='r')

    X_reg_collect=[]
    y_reg_collect=[]
    aux_reg_collect=[]
    
    for pidx,pid in enumerate(pids):

        if (pidx+1)%10==0:
            print("Processing PID {}/{}".format(pidx+1,len(pids)),flush=True)

        if not configs["endpoint"]=="resp_extval":
            adm_time=lookup_admission_time(pid, df_static)

        if configs["endpoint"]=="resp_extval":
            death_str=df_static[df_static["admissionid"]==pid].destination.values[0]
            mort_status=4.0 if death_str=="Overleden" else 2.0
        else:
            try:
                mort_status=int(df_static[df_static["PatientID"]==pid].Discharge)
            except ValueError:
                mort_status=2.0
                
        df_pid=pd.read_hdf(os.path.join(imputed_f,"batch_{}.h5".format(batch_id)),mode='r', where="PatientID={}".format(pid))

        if df_pid.shape[0]==0:
            print("WARNING: No input data for PID: {}".format(pid))
            continue

        if configs["endpoint"]=="resp_extval":
            df_merged_pid=pd.read_parquet(cand_raw_batch[0], filters=[("PatientID","=",pid)])
        else:
            df_merged_pid=pd.read_hdf(cand_raw_batch[0],mode='r', where="PatientID={}".format(pid))
            
        df_merged_pid.sort_values(by="Datetime", inplace=True)

        suppox_val={}
        suppox_ts={}

        # Main route of SuppOx

        # External validation data-set has no high-flow oxygen records
        if configs["endpoint"]=="resp_extval":
            df_suppox_red_async=df_merged_pid[[var_map["SuppFiO2_1"], "AbsDatetime"]]
            df_suppox_red_async.dropna(inplace=True,how="any")
            suppox_async_red_ts=np.array(df_suppox_red_async["AbsDatetime"])
            suppox_val["SUPPFIO2_1"]=np.array(df_suppox_red_async[var_map["SuppFiO2_1"]])
        else:
            df_suppox_red_async=df_merged_pid[[var_map["SuppFiO2_1"], var_map["SuppFiO2_2"], "Datetime"]]
            df_suppox_red_async.dropna(inplace=True,how="all",thresh=2)
            suppox_async_red_ts=np.array(df_suppox_red_async["Datetime"])
            suppox_val["SUPPFIO2_1"]=np.array(df_suppox_red_async[var_map["SuppFiO2_1"]])
            suppox_val["SUPPFIO2_2"]=np.array(df_suppox_red_async[var_map["SuppFiO2_2"]])

        # Strategy is to create an imputed SuppOx column based on the spec using
        # forward filling heuristics

        # Relevant meta-variables
        
        fio2_col=np.array(df_pid[var_map["FiO2"]])
        pao2_col=np.array(df_pid[var_map["PaO2"]])
        etco2_col=np.array(df_pid[var_map["etCO2"]])
        paco2_col=np.array(df_pid[var_map["PaCO2"]])

        gcs_a_col=np.array(df_pid[var_map["GCS_Antwort"]])
        gcs_m_col=np.array(df_pid[var_map["GCS_Motorik"]])
        gcs_aug_col=np.array(df_pid[var_map["GCS_Augen"]])

        # For resp ext-val get the weight from the static frame, and set it to
        # the center of the weight bin

        if configs["endpoint"]=="resp_extval":
            df_static_pid=df_static[df_static["admissionid"]==pid]
            weight_str=df_static_pid.iloc[0].weightgroup
            if weight_str is None or weight_str=="None":
                bin_center=np.nan
            elif "+" in weight_str:
                bounds=weight_str.split("+")
                bin_center=int(bounds[0])
            elif "-" in weight_str:
                bounds=weight_str.split("-")
                if bounds[-1]=='':
                    bin_center=int(bounds[0])
                else:
                    bin_center=(int(bounds[-1])+int(bounds[0]))/2
            else:
                assert(False)
                    
            weight_col=np.zeros_like(fio2_col)
            weight_col[:]=bin_center
        else:
            weight_col=np.array(df_pid[var_map["Weight"][0]])
            
        noreph_col=np.array(df_pid[var_map["Norephenephrine"][0]])
        epineph_col=np.array(df_pid[var_map["Epinephrine"][0]])
        
        # Milrinone (pm42) is currently not available in the external validation data-set
        if not configs["endpoint"]=="resp_extval":
            milri_col=np.array(df_pid[var_map["Milrinone"][0]])
            levosi_col=np.array(df_pid[var_map["Levosimendan"][0]])
            theophy_col=np.array(df_pid[var_map["Theophyllin"][0]])
            vasopre_col=np.array(df_pid[var_map["Vasopressin"][0]])            
            
        dobut_col=np.array(df_pid[var_map["Dobutamine"][0]])
        lactate_col=np.array(df_pid[var_map["Lactate"][0]])
        
        peep_col=np.array(df_pid[var_map["PEEP"]])
        pressure_support_col=np.array(df_pid[var_map["PressSupport"]])
        min_volume_col=np.array(df_pid[var_map["MinuteVolume"]])

        # Heartrate
        hr_col=np.array(df_pid[var_map["HR"]])
        hr_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])

        rrate_col=np.array(df_pid[var_map["RRate"]])

        tv_col=np.array(df_pid[var_map["TV"]])
        map_col=np.array(df_pid[var_map["MAP"][0]])

        # Airway (vm66) is currently not available in the external validation data-set
        if not configs["endpoint"]=="resp_extval":
            airway_col=np.array(df_pid[var_map["Airway"]])
            
        intube_col=np.array(df_pid[var_map["int_state"]])
        trach_col=np.array(df_pid[var_map["trach"]])

        # Ventilator mode group columns (not yet complete)
        vent_mode_col=np.array(df_pid[var_map["vent_mode"]])

        spo2_col=np.array(df_pid[var_map["SpO2"]])

        if configs["presmooth_spo2"]:
            spo2_col=percentile_smooth(spo2_col,configs["spo2_smooth_percentile"],configs["spo2_smooth_window_size_mins"])
        
        sao2_col=np.array(df_pid[var_map["SaO2"]])
        ph_col=np.array(df_pid[var_map["pH"]])

        fio2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["FiO2"])])
        pao2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PaO2"])])
        etco2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["etCO2"])])
        peep_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PEEP"])])
        hr_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["HR"])])
        spo2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SpO2"])])
        sao2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SaO2"])])
        ph_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["pH"])])

        # Subsample ABGAs values
        if configs["subsample_abga"]:
            pao2_col, pao2_meas_cnt=subsample_individual(pao2_col, pao2_meas_cnt, ss_ratio=configs["abga_ss_ratio"],
                                                         normal_value=87)
            sao2_col, sao2_meas_cnt=subsample_individual(sao2_col, sao2_meas_cnt, ss_ratio=configs["abga_ss_ratio"],
                                                         normal_value=96)

        # Subsample non-invasive SpO2
        if configs["subsample_spo2"]:
            spo2_col, spo2_meas_cnt=subsample_blocked(spo2_col, spo2_meas_cnt, ss_ratio=configs["spo2_ss_ratio"],
                                                      block_length=configs["spo2_ss_block_length"],normal_value=98)

        abs_dtime_arr=np.array(df_pid["AbsDatetime"])
        
        event_status_arr=np.zeros(shape=(fio2_col.size),dtype="<S10")

        # Status arrays
        pao2_avail_arr=np.zeros(shape=(fio2_col.size))
        fio2_avail_arr=np.zeros(shape=(fio2_col.size))
        fio2_suppox_arr=np.zeros(shape=(fio2_col.size))
        fio2_ambient_arr=np.zeros(shape=(fio2_col.size))
        pao2_sao2_model_arr=np.zeros(shape=(fio2_col.size))
        pao2_full_model_arr=np.zeros(shape=(fio2_col.size))
        
        ratio_arr=np.zeros(shape=(fio2_col.size))
        sur_ratio_arr=np.zeros(shape=(fio2_col.size))
        
        pao2_est_arr=np.zeros(shape=(fio2_col.size))
        fio2_est_arr=np.zeros(shape=(fio2_col.size))
        vent_status_arr=np.zeros(shape=(fio2_col.size))
        readiness_ext_arr=np.zeros(shape=(fio2_col.size))
        readiness_ext_arr[:]=np.nan

        # Votes arrays
        vent_votes_arr=np.zeros(shape=(fio2_col.size))
        vent_votes_etco2_arr=np.zeros(shape=(fio2_col.size))
        vent_votes_ventgroup_arr=np.zeros(shape=(fio2_col.size))
        vent_votes_tv_arr=np.zeros(shape=(fio2_col.size))
        vent_votes_airway_arr=np.zeros(shape=(fio2_col.size))
        vent_votes_adm_arr=np.zeros(shape=(fio2_col.size))
        
        peep_status_arr=np.zeros(shape=(fio2_col.size))
        peep_threshold_arr=np.zeros(shape=(fio2_col.size))
        hr_status_arr=np.zeros(shape=(fio2_col.size))
        etco2_status_arr=np.zeros(shape=(fio2_col.size))
        event_status_arr.fill("UNKNOWN")

        # Array pointers tracking the current active value of each type
        suppox_async_red_ptr=-1

        # ======================== VENTILATION ================================================================================================

        # Label each point in the 30 minute window with ventilation 
        in_vent_event=False

        if configs["endpoint"]=="resp_extval":
            vent_vote_base_score=0
        else:
            vent_vote_base_score=1 if adm_time<np.datetime64("2009-12-06T11:00:00.000000000") else 0
        
        for jdx in range(0,len(ratio_arr)):
            win_etco2=etco2_col[max(0,jdx-configs["etco2_vent_search_bw"]):min(len(ratio_arr),jdx+configs["etco2_vent_search_fw"])]
            win_etco2_meas=etco2_meas_cnt[max(0,jdx-configs["etco2_vent_search_bw"]):min(len(ratio_arr),jdx+configs["etco2_vent_search_fw"])]
            win_peep=peep_col[max(0,jdx-configs["peep_search_bw"]):min(len(ratio_arr),jdx+configs["peep_search_fw"])]
            win_peep_meas=peep_meas_cnt[max(0,jdx-configs["peep_search_bw"]):min(len(ratio_arr),jdx+configs["peep_search_fw"])]
            win_hr_meas=hr_meas_cnt[max(0,jdx-configs["hr_vent_search_bw"]):min(len(ratio_arr),jdx+configs["hr_vent_search_fw"])]
            etco2_meas_win=win_etco2_meas[-1]-win_etco2_meas[0]>0
            peep_meas_win=win_peep_meas[-1]-win_peep_meas[0]>0
            hr_meas_win=win_hr_meas[-1]-win_hr_meas[0]>0
            current_vent_group=vent_mode_col[jdx]
            current_tv=tv_col[jdx]
            current_trach=trach_col[jdx]

            if not configs["endpoint"]=="resp_extval":
                current_airway=airway_col[jdx]
                
            current_intube=intube_col[jdx]

            vote_score=0
            
            # EtCO2 requirement
            if etco2_meas_win and (win_etco2>0.5).any():
                vote_score+=2
                vent_votes_etco2_arr[jdx]=2

            # Ventilation group requirement
            if current_vent_group in [2.0,3.0]:
                vote_score+=1
                vent_votes_ventgroup_arr[jdx]+=1
            elif current_vent_group in [1.0]:
                vote_score-=1
                vent_votes_ventgroup_arr[jdx]-=1
            elif current_vent_group in [4.0,5.0,6.0]:
                vote_score-=2
                vent_votes_ventgroup_arr[jdx]-=2

            # TV presence requirement
            if current_tv>0:
                vote_score+=1
                vent_votes_tv_arr[jdx]=1

            # Airway requirement

            if configs["endpoint"]=="resp_extval":
                airway_cond=current_trach>0 or current_intube>0
            else:
                airway_cond=current_trach>0 or current_intube>0 or current_airway in [1,2]
            
            if airway_cond:
                vote_score+=2
                vent_votes_airway_arr[jdx]=2

            # No airway
            if not configs["endpoint"]=="resp_extval" and current_airway in [3,4,5,6]:
                vote_score-=1
                vent_votes_airway_arr[jdx]=-1

            vote_score+=vent_vote_base_score
            vent_votes_adm_arr[jdx]=vent_vote_base_score
                
            vent_votes_arr[jdx]=vote_score
                
            if vote_score>=configs["vent_vote_threshold"]:
                in_vent_event=True
                vent_status_arr[jdx]=1
            else:
                in_vent_event=False
                
            if peep_meas_win:
                peep_status_arr[jdx]=1
            if (win_peep>=configs["peep_threshold"]).any():
                peep_threshold_arr[jdx]=1
            if etco2_meas_win:
                etco2_status_arr[jdx]=1
            if hr_meas_win:
                hr_status_arr[jdx]=1

        if configs["detect_hr_gaps"]:
            vent_status_arr=delete_low_density_hr_gap(vent_status_arr, hr_status_arr, configs=configs)

        if configs["merge_short_vent_gaps"]:
            vent_status_arr=merge_short_vent_gaps(vent_status_arr, configs["short_gap_hours"])

        if configs["delete_short_vent_events"]:
            vent_status_arr=delete_short_vent_events(vent_status_arr, configs["short_event_hours"])

        # Ventilation period array, merging gaps of 24 hours if on both sides tracheotomy is active
        vent_period_arr=np.copy(vent_status_arr)
        in_gap=False
        gap_length=0
        before_gap_trach=0

        # Compute ventilation period array
        for idx in range(len(event_status_arr)):
            cur_state=vent_status_arr[idx]
            if in_gap and cur_state==0:
                gap_length+=5
            elif not in_gap and cur_state==0:
                if idx>0:
                    before_gap_trach=trach_col[idx-1]
                in_gap=True
                in_gap_idx=idx
                gap_length=5
            elif in_gap and cur_state==1:
                in_gap=False
                after_gap_trach=trach_col[idx]
                if gap_length/60.<=configs["trach_gap_hours"] and before_gap_trach==1 and after_gap_trach==1:
                    vent_period_arr[in_gap_idx:idx]=1

        # Delete short ventilation periods if no HR gap before
        in_event=False
        event_length=0
        for idx in range(len(vent_period_arr)):
            cur_state=vent_period_arr[idx]
            if in_event and cur_state==1.0:
                event_length+=5
            if not in_event and cur_state==1.0:
                in_event=True
                event_length=5
                event_start_idx=idx
            if in_event and (np.isnan(cur_state) or cur_state==0.0):
                in_event=False

                # Short event at beginning of stay shall never be deleted...
                if event_start_idx==0:
                    delete_event=False
                else:
                    search_hr_idx=event_start_idx-1
                    while search_hr_idx>=0:
                        if hr_status_arr[search_hr_idx]==1.0:
                            hr_gap_length=5*(event_start_idx-search_hr_idx)
                            delete_event=True
                            break
                        search_hr_idx-=1

                    # Found no HR before event, do not delete event...
                    if search_hr_idx==-1:
                        delete_event=False

                # Delete event in principle, then check if short enough...
                if delete_event:
                    event_length+=hr_gap_length
                    if event_length/60.<=configs["short_event_hours_vent_period"]:
                        vent_period_arr[event_start_idx:idx]=0.0

        # ============================== OXYGENATION ENDPOINTS ==================================================================

        # Label each point in the 30 minute window (except ventilation)
        for jdx in range(0,len(ratio_arr)):

            # Advance to the last SuppOx infos before grid point
            cur_time=abs_dtime_arr[jdx]
            while True:
                suppox_async_red_ptr=suppox_async_red_ptr+1
                if suppox_async_red_ptr>=len(suppox_async_red_ts) or suppox_async_red_ts[suppox_async_red_ptr]>cur_time:
                    suppox_async_red_ptr=suppox_async_red_ptr-1
                    break 

            # Estimate the current FiO2 value
            bw_fio2=fio2_col[max(0,jdx-configs["sz_fio2_window"]):jdx+1]
            bw_fio2_meas=fio2_meas_cnt[max(0,jdx-configs["sz_fio2_window"]):jdx+1]
            bw_etco2_meas=etco2_meas_cnt[max(0,jdx-configs["sz_etco2_window"]):jdx+1]
            fio2_meas=bw_fio2_meas[-1]-bw_fio2_meas[0]>0
            etco2_meas=bw_etco2_meas[-1]-bw_etco2_meas[0]>0
            mode_group_est=vent_mode_col[jdx]

            # FiO2 is measured since beginning of stay and EtCO2 was measured, we use FiO2 (indefinite forward filling)
            # if ventilation is active or the current estimate of ventilation mode group is NIV.
            if fio2_meas and (vent_status_arr[jdx]==1.0 or mode_group_est==4.0):
                event_count["FIO2_AVAILABLE"]+=1
                fio2_val=bw_fio2[-1]/100
                fio2_avail_arr[jdx]=1

            # Use supplemental oxygen or ambient air oxygen
            else:
                # No real measurements up to now, or the last real measurement 
                # was more than 8 hours away.
                
                if suppox_async_red_ptr==-1 or (cur_time-suppox_async_red_ts[suppox_async_red_ptr])>np.timedelta64(configs["suppox_max_ffill"],'h'):
                    event_count["SUPPOX_NO_MEAS_12_HOURS_LIMIT"]+=1
                    fio2_val=configs["ambient_fio2"]
                    fio2_ambient_arr[jdx]=1

                # Find the most recent source variable of SuppOx
                else:
                    suppox_fio2_1=suppox_val["SUPPFIO2_1"][suppox_async_red_ptr]
                    if not configs["endpoint"]=="resp_extval":
                        suppox_fio2_2=suppox_val["SUPPFIO2_2"][suppox_async_red_ptr]
                    
                    # SuppOx information from main source
                    if np.isfinite(suppox_fio2_1):
                        event_count["SUPPOX_MAIN_VAR"]+=1
                        fio2_val=int(suppox_fio2_1)/100
                        fio2_suppox_arr[jdx]=1

                    # SuppOx information from high-flow oxygen
                    elif not configs["endpoint"]=="resp_extval" and np.isfinite(suppox_fio2_2):
                        event_count["SUPPOX_HIGH_FLOW"]+=1
                        fio2_val=int(suppox_fio2_2)/100
                        fio2_suppox_arr[jdx]=1

                    else:
                        assert(False)

            fio2_est_arr[jdx]=fio2_val

        if configs["write_reg_data"]:
            X_reg,y_reg,aux_reg = \
                collect_regression_data(spo2_col, spo2_meas_cnt, pao2_col, pao2_meas_cnt, fio2_est_arr,
                                        sao2_col, sao2_meas_cnt, ph_col, ph_meas_cnt)
            if X_reg is not None:
                X_reg_collect.append(X_reg)
                y_reg_collect.append(y_reg)
                aux_reg_collect.append(aux_reg)

        if configs["kernel_smooth_estimate_fio2"]:
            fio2_est_arr=kernel_smooth_arr(fio2_est_arr, bandwidth=configs["smoothing_bandwidth"])

        # Apply surrogate regression model
        pao2_sur_est, pao2_pop_est=apply_regression_model(spo2_col, spo2_meas_cnt, pao2_col, pao2_meas_cnt,
                                                          sao2_col, sao2_meas_cnt, ph_col, ph_meas_cnt,
                                                          sur_reg_base_model, sur_meta_model, sur_base_scaler, sur_meta_scaler)

        if configs["kernel_smooth_estimate_pao2"]:
            pao2_sur_est=kernel_smooth_arr(pao2_sur_est, bandwidth=configs["smoothing_bandwidth"])        

        # Convex combination of the estimate
        if configs["mix_real_estimated_pao2"]:
            pao2_sur_est=mix_real_est_pao2(pao2_col, pao2_meas_cnt, pao2_sur_est, bandwidth=configs["smoothing_bandwidth"])

        # Compute Horowitz indices (Kernel pipeline / Surrogate model pipeline)
        for jdx in range(len(ratio_arr)):
            sur_ratio_arr[jdx]=pao2_sur_est[jdx]/fio2_est_arr[jdx]

        # Post-smooth Horowitz index
        if configs["post_smooth_pf_ratio"]:
            sur_ratio_arr=kernel_smooth_arr(sur_ratio_arr, bandwidth=configs["post_smoothing_bandwidth"])

        pf_event_est_arr=np.copy(sur_ratio_arr)
            
        # Now label based on the array of estimated Horowitz indices
        for idx in range(0,len(event_status_arr)-configs["offset_back_windows"]):
            est_idx=pf_event_est_arr[idx:min(len(ratio_arr),idx+sz_window)]
            est_vent=vent_status_arr[idx:min(len(ratio_arr),idx+sz_window)]
            est_peep_dense=peep_status_arr[idx:min(len(ratio_arr),idx+sz_window)]
            est_peep_threshold=peep_threshold_arr[idx:min(len(ratio_arr),idx+sz_window)]

            if np.sum((est_idx<=100) & ((est_vent==0.0) | (est_vent==1.0) & (est_peep_dense==0.0) | (est_vent==1.0) & (est_peep_dense==1.0) & (est_peep_threshold==1.0))  )>=2/3*len(est_idx):
                event_status_arr[idx]="event_3"
            elif np.sum((est_idx<=200) & ((est_vent==0.0) | (est_vent==1.0) & (est_peep_dense==0.0) | (est_vent==1.0) & (est_peep_dense==1.0) & (est_peep_threshold==1.0))  )>=2/3*len(est_idx):
                event_status_arr[idx]="event_2"
            elif np.sum((est_idx<=300) & ((est_vent==0.0) | (est_vent==1.0) & (est_peep_dense==0.0) | (est_vent==1.0) & (est_peep_dense==1.0) & (est_peep_threshold==1.0))  )>=2/3*len(est_idx):
                event_status_arr[idx]="event_1"
            elif np.sum(np.isnan(est_idx))<2/3*len(est_idx):
                event_status_arr[idx]="event_0"
                
        # Re-traverse the array and correct the right edges of events

        # Correct right edges of event 0 (correct level to level 0)
        on_right_edge=False
        in_event=False
        for idx in range(0,len(event_status_arr)-configs["offset_back_windows"]):
            cur_state=event_status_arr[idx].decode()
            if cur_state in ["event_0"] and not in_event:
                in_event=True
            elif in_event and cur_state not in ["event_0"]:
                in_event=False
                on_right_edge=True
            if on_right_edge:
                if pf_event_est_arr[idx]<300:
                    on_right_edge=False
                else:
                    event_status_arr[idx]="event_0"

        # Correct right edges of event 1 (correct to level 1)                    
        on_right_edge=False
        in_event=False
        for idx in range(0,len(event_status_arr)-configs["offset_back_windows"]):
            cur_state=event_status_arr[idx].decode()
            if cur_state in ["event_1"] and not in_event:
                in_event=True
            elif in_event and cur_state not in ["event_1"]:
                in_event=False
                on_right_edge=True
            if on_right_edge:
                if pf_event_est_arr[idx]<200 or pf_event_est_arr[idx]>=300:
                    on_right_edge=False
                else:
                    event_status_arr[idx]="event_1"

        # Correct right edges of event 2 (correct to level 2)                    
        on_right_edge=False
        in_event=False
        for idx in range(0,len(event_status_arr)-configs["offset_back_windows"]):
            cur_state=event_status_arr[idx].decode()
            if cur_state in ["event_2"] and not in_event:
                in_event=True
            elif in_event and cur_state not in ["event_2"]:
                in_event=False
                on_right_edge=True
            if on_right_edge:
                if pf_event_est_arr[idx]<100 or pf_event_est_arr[idx]>=200:
                    on_right_edge=False
                else:
                    event_status_arr[idx]="event_2"

        # Correct right edges of event 3 (correct to level 3)
        on_right_edge=False
        in_event=False
        for idx in range(0,len(event_status_arr)-configs["offset_back_windows"]):
            cur_state=event_status_arr[idx].decode()
            if cur_state in ["event_3"] and not in_event:
                in_event=True
            elif in_event and cur_state not in ["event_3"]:
                in_event=False
                on_right_edge=True
            if on_right_edge:
                if pf_event_est_arr[idx]>=100:
                    on_right_edge=False
                else:
                    event_status_arr[idx]="event_3"

        # Traverse the array and delete short gap
        event_status_arr, relabel_arr=delete_small_continuous_blocks(event_status_arr,
                                                                     block_threshold=configs["pf_event_merge_threshold"])

        # ------------ READINESS FOR EXTUBATION ------------------------------------------------------------------------

        ext_ready_violations=np.copy(readiness_ext_arr)
        ext_not_ready_vent_mode=np.copy(readiness_ext_arr)
        ext_not_ready_peep=np.copy(readiness_ext_arr)
        ext_not_ready_psupport=np.copy(readiness_ext_arr)
        ext_not_ready_fio2=np.copy(readiness_ext_arr)
        ext_not_ready_sbidx=np.copy(readiness_ext_arr)
        ext_not_ready_rr=np.copy(readiness_ext_arr)
        ext_not_ready_minvol=np.copy(readiness_ext_arr)
        ext_not_ready_pfratio=np.copy(readiness_ext_arr)
        ext_not_ready_paco2=np.copy(readiness_ext_arr)
        ext_not_ready_gcs=np.copy(readiness_ext_arr)
        ext_not_ready_map=np.copy(readiness_ext_arr)
        ext_not_ready_drugs=np.copy(readiness_ext_arr)
        ext_not_ready_lactate=np.copy(readiness_ext_arr)
        
        for jdx in range(len(event_status_arr)-1):
            
            # Can only be extubated while intubated
            if vent_period_arr[jdx]==0.0:
                continue

            violation_score=0

            current_mode=vent_mode_col[jdx]
            if not current_mode==3.0:
                ext_not_ready_vent_mode[jdx]=1.0
                violation_score+=configs["ext_ready_violation_threshold"]
            else:
                ext_not_ready_vent_mode[jdx]=0.0

            # Before 2010, subtract violation again
            if vent_vote_base_score==1:
                violation_score-=configs["ext_ready_violation_threshold"]
                
            current_peep=peep_col[jdx]
            if current_peep>7:
                ext_not_ready_peep[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_peep[jdx]=0.0

            current_psupport=pressure_support_col[jdx]
            if current_psupport>10:
                ext_not_ready_psupport[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_psupport[jdx]=0.0

            current_fio2=fio2_est_arr[jdx]
            if current_fio2>0.4:
                ext_not_ready_fio2[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_fio2[jdx]=0.0

            current_rr=rrate_col[jdx]
            current_tv=tv_col[jdx]
            rapid_shallow_breathing_idx=current_rr/current_tv*1000
            if current_tv>0 and rapid_shallow_breathing_idx>=105:
                ext_not_ready_sbidx[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_sbidx[jdx]=0.0

            if current_rr>=35:
                ext_not_ready_rr[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_rr[jdx]=0.0

            current_min_volume=min_volume_col[jdx]
            if current_min_volume>=10:
                ext_not_ready_minvol[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_minvol[jdx]=0.0

            current_pf_ratio=sur_ratio_arr[jdx]
            if current_pf_ratio<=150:
                ext_not_ready_pfratio[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_pfratio[jdx]=0.0

            current_paco2=paco2_col[jdx]
            if current_paco2>=50:
                ext_not_ready_paco2[jdx]=1.0
                violation_score+=3
            else:
                ext_not_ready_paco2[jdx]=0.0

            current_gcs=gcs_a_col[jdx]+gcs_m_col[jdx]+gcs_aug_col[jdx]
            if current_gcs<=8:
                ext_not_ready_gcs[jdx]=1.0
                violation_score+=1
            else:
                ext_not_ready_gcs[jdx]=0.0

            current_map=map_col[jdx]
            if current_map<=60:
                ext_not_ready_map[jdx]=1.0
                violation_score+=1
            else:
                ext_not_ready_map[jdx]=0.0

            current_stand_noreph=noreph_col[jdx]/weight_col[jdx]
            current_epineph=epineph_col[jdx]

            if not configs["endpoint"]=="resp_extval":
                current_milri=milri_col[jdx]
                current_levosi=levosi_col[jdx]
                current_theophy=theophy_col[jdx]
                current_vasopre=vasopre_col[jdx]
            
            current_dobut=dobut_col[jdx]

            # Restricted drug condition depending on how many vasopressors we have available
            if configs["endpoint"]=="resp_extval":
                drug_cond=current_stand_noreph>0.05 or current_epineph>0 or current_dobut>0
            else:
                drug_cond=current_stand_noreph>0.05 or current_epineph>0 or current_dobut>0 or \
                    current_milri>0 or current_levosi>0 or current_theophy>0 or current_vasopre>0
            
            if drug_cond:
                ext_not_ready_drugs[jdx]=1.0
                violation_score+=1
            else:
                ext_not_ready_drugs[jdx]=0.0

            current_lactate=lactate_col[jdx]
            if current_lactate>=2.5:
                ext_not_ready_lactate[jdx]=1.0
                violation_score+=1
            else:
                ext_not_ready_lactate[jdx]=0.0

            ext_ready_violations[jdx]=violation_score
            readiness_ext_arr[jdx]=1.0 if violation_score<configs["ext_ready_violation_threshold"] else 0.0

        # Now post process the ext readiness array with a 60-minute backwards window
        readiness_tmp=np.copy(readiness_ext_arr)

        for jdx in range(len(readiness_tmp)):
            if np.isfinite(readiness_tmp[jdx]):
                subarr=readiness_tmp[max(0,jdx-11):min(readiness_tmp.size,jdx+1)]
                subarr=subarr[np.isfinite(subarr)]
                if np.sum(subarr==1.0)/subarr.size>=configs["readiness_vote_window_threshold"]:
                    readiness_ext_arr[jdx]=1.0
                    readiness_ext_count+=1
                else:
                    readiness_ext_arr[jdx]=0.0
                    not_ready_ext_count+=1

                    
        time_col=np.array(df_pid["AbsDatetime"])
        rel_time_col=np.array(df_pid["RelDatetime"])
        pid_col=np.array(df_pid["PatientID"])

        df_out_dict={}

        df_out_dict["AbsDatetime"]=time_col
        df_out_dict["RelDatetime"]=rel_time_col
        df_out_dict["PatientID"]=pid_col
        status_list=list(map(lambda raw_str: raw_str.decode("unicode_escape"), event_status_arr.tolist()))
        df_out_dict["endpoint_status"]=status_list
        df_out_dict["ep_status_relabel"]=relabel_arr

        # Status columns
        df_out_dict["fio2_available"]=fio2_avail_arr
        df_out_dict["fio2_suppox"]=fio2_suppox_arr
        df_out_dict["fio2_ambient"]=fio2_ambient_arr
        df_out_dict["fio2_estimated"]=fio2_est_arr
        df_out_dict["pao2_estimated"]=pao2_sur_est
        df_out_dict["pao2_estimated_pop"]=pao2_pop_est
        df_out_dict["estimated_ratio"]=sur_ratio_arr
        df_out_dict["vent_state"]=vent_status_arr
        df_out_dict["vent_period"]=vent_period_arr

        # Ventilation voting base columns
        df_out_dict["vent_votes"]=vent_votes_arr
        df_out_dict["vent_votes_etco2"]=vent_votes_etco2_arr
        df_out_dict["vent_votes_ventgroup"]=vent_votes_ventgroup_arr
        df_out_dict["vent_votes_tv"]=vent_votes_tv_arr
        df_out_dict["vent_votes_airway"]=vent_votes_airway_arr
        df_out_dict["vent_votes_adm"]=vent_votes_adm_arr
        
        ext_failure_arr=np.zeros_like(vent_period_arr)
        ext_failure_arr[:]=np.nan
        ext_failure_simple_arr=np.zeros_like(vent_period_arr)
        ext_failure_simple_arr[:]=np.nan
        
        ext_failure_reason_det=np.zeros_like(vent_period_arr)
        ext_failure_reason_det[:]=np.nan
        ext_failure_reason_reintube=np.zeros_like(vent_period_arr)
        ext_failure_reason_reintube[:]=np.nan
        
        for idx in range(vent_period_arr.size-1):

            if vent_period_arr.size-idx<=576 and mort_status==4.0:
                patient_died=True
            else:
                patient_died=False

            # Consider only extubations that are not from tracheotomy
            if vent_period_arr[idx]==1.0 and vent_period_arr[idx+1]==0.0 and trach_col[idx]==0.0:
                future_event_status=status_list[idx+1:min(len(status_list),idx+configs["ext_failure_window"])]
                current_event_status=status_list[idx]
                future_vent_per=vent_period_arr[idx+1:min(len(vent_period_arr),idx+configs["ext_failure_window"])]
                e2_count=future_event_status.count("event_2")
                e3_count=future_event_status.count("event_3")

                ext_failure_cond_reint=(future_vent_per==1.0).any()
                # Check for the special case of a gap before the re-intubation
                if ext_failure_cond_reint:
                    hr_status_future=hr_status_arr[idx+1:min(len(vent_period_arr),idx+configs["ext_failure_window"])]
                    for jdx in range(future_vent_per.size):
                        if future_vent_per[jdx]==1.0:
                            hr_search_arr=hr_status_future[max(0,jdx-9):jdx+1]
                            # Detected HR gap, not a valid re-intubation
                            if np.sum(hr_search_arr==0)>=2/3*hr_search_arr.size:
                                ext_failure_cond_reint=False
                            break

                ext_failure_cond_oxy=current_event_status in ["event_0","event_1"] \
                    and (e2_count+e3_count)/len(future_event_status)>=configs["ext_failure_rf_ratio"]

                # Patient died and was not reintubated, model not applicable
                if patient_died and not ext_failure_cond_reint:
                    ext_failure_arr[idx]=np.nan
                    ext_failure_simple_arr[idx]=np.nan

                # Reintubation -> Extubation failure
                else:

                    if ext_failure_cond_reint or ext_failure_cond_oxy:
                        ext_failure_arr[idx]=1.0
                    else:
                        ext_failure_arr[idx]=0.0
                        
                    if ext_failure_cond_reint:
                        ext_failure_simple_arr[idx]=1.0
                    else:
                        ext_failure_simple_arr[idx]=0.0

                    # Store the reason for the extubation failure
                    if ext_failure_cond_oxy:
                        ext_failure_reason_det[idx]=1.0
                    else:
                        ext_failure_reason_det[idx]=0.0
                    if ext_failure_cond_reint:
                        ext_failure_reason_reintube[idx]=1.0
                    else:
                        ext_failure_reason_reintube[idx]=0.0


        # Complex version of extubation failure, taking into account reintubations/oxygenation failure
        df_out_dict["ext_failure"]=ext_failure_arr

        # Simple version of extubation failure, taking into account only reintubations
        df_out_dict["ext_failure_simple"]=ext_failure_simple_arr

        # Store the reasons for the extubation failures
        df_out_dict["ext_failure_reason_det"]=ext_failure_reason_det
        df_out_dict["ext_failure_reason_reintube"]=ext_failure_reason_reintube

        stat_counts_ready_and_failure+=np.sum((ext_failure_arr==1.0) & (readiness_ext_arr==1.0))
        stat_counts_ready_and_success+=np.sum((ext_failure_arr==0.0) & (readiness_ext_arr==1.0))
        stat_counts_nready_and_failure+=np.sum((ext_failure_arr==1.0) & (readiness_ext_arr==0.0))
        stat_counts_nready_and_success+=np.sum((ext_failure_arr==0.0) & (readiness_ext_arr==0.0))
        stat_counts_ready_nextube+=np.sum(np.isnan(ext_failure_arr) & (readiness_ext_arr==1.0))
        stat_counts_nready_nextube+=np.sum(np.isnan(ext_failure_arr) & (readiness_ext_arr==0.0))

        # Readiness to extubate status
        df_out_dict["readiness_ext"]=readiness_ext_arr

        # Violation score for extubation readiness
        df_out_dict["ext_ready_violation_score"]=ext_ready_violations

        readiness_and_extubated_cnt+=np.sum(np.isfinite(ext_failure_arr) & (readiness_ext_arr==1.0))
        extubated_cnt+=np.sum(readiness_ext_arr==1.0)

        # Reasons for not being ready to extubate
        df_out_dict["ext_not_ready_vent_mode"]=ext_not_ready_vent_mode
        df_out_dict["ext_not_ready_peep"]=ext_not_ready_peep
        df_out_dict["ext_not_ready_psupport"]=ext_not_ready_psupport
        df_out_dict["ext_not_ready_fio2"]=ext_not_ready_fio2
        df_out_dict["ext_not_ready_sbidx"]=ext_not_ready_sbidx
        df_out_dict["ext_not_ready_rr"]=ext_not_ready_rr
        df_out_dict["ext_not_ready_minvol"]=ext_not_ready_minvol
        df_out_dict["ext_not_ready_pfratio"]=ext_not_ready_pfratio
        df_out_dict["ext_not_ready_paco2"]=ext_not_ready_paco2
        df_out_dict["ext_not_ready_gcs"]=ext_not_ready_gcs
        df_out_dict["ext_not_ready_map"]=ext_not_ready_map
        df_out_dict["ext_not_ready_drugs"]=ext_not_ready_drugs
        df_out_dict["ext_not_ready_lactate"]=ext_not_ready_lactate

        df_out=pd.DataFrame(df_out_dict)

        if first_write:
            append_mode=False
            open_mode='w'
        else:
            append_mode=True
            open_mode='a'

        if not configs["debug_mode"] and configs["write_endpoint_data"]:
            df_out.to_hdf(out_fp,'data',complevel=configs["hdf_comp_level"],complib=configs["hdf_comp_alg"],
                          format="table", append=append_mode, mode=open_mode, data_columns=["PatientID"])

        first_write=False

    if configs["write_reg_data"]:
        train_reg_dict={"X_reg": X_reg_collect, "y_reg": y_reg_collect, "fio2_reg": aux_reg_collect}
        with open(os.path.join(out_folder,"reg_data_batch_{}.pickle".format(batch_id)),'wb') as reg_fp:
            pickle.dump(train_reg_dict,reg_fp)
    

