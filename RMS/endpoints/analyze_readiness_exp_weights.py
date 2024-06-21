''' Analyze the weights for readiness to extubate, for extracting
    a continuous REXP-score to use as a baseline for extubation
    failure prediction'''

import argparse
import glob
import os
import os.path
import ipdb
import random

import pandas as pd
import numpy as np
import sklearn.linear_model as sklm
import sklearn.metrics as skmetrics

def execute(configs):
    random.seed(configs["random_state"])
    all_eps=glob.glob(os.path.join(configs["endpoint_path"], "point_est","batch_*.h5"))
    random.shuffle(all_eps)

    if configs["small_sample"]:
        all_eps=all_eps[:1]

    y_arr=[]
    X_arr=[]
        
    for epix,ep_file in enumerate(all_eps):
        print("Batch file: {}/{}".format(epix+1,len(all_eps)))
        df_batch=pd.read_hdf(ep_file,mode='r')
        
        for pid in df_batch.PatientID.unique():
            df_pid=df_batch[df_batch.PatientID==pid]
            trace_arr=np.array(df_pid["ext_not_ready_vent_mode"])
            vent_mode_arr=np.array(df_pid["ext_not_ready_vent_mode"])
            peep_arr=np.array(df_pid["ext_not_ready_peep"])
            psupport_arr=np.array(df_pid["ext_not_ready_psupport"])
            fio2_arr=np.array(df_pid["ext_not_ready_fio2"])
            sbidx_arr=np.array(df_pid["ext_not_ready_sbidx"])
            rr_arr=np.array(df_pid["ext_not_ready_rr"])
            minvol_arr=np.array(df_pid["ext_not_ready_minvol"])
            pfratio_arr=np.array(df_pid["ext_not_ready_pfratio"])
            paco2_arr=np.array(df_pid["ext_not_ready_paco2"])
            gcs_arr=np.array(df_pid["ext_not_ready_gcs"])
            map_arr=np.array(df_pid["ext_not_ready_map"])
            drugs_arr=np.array(df_pid["ext_not_ready_drugs"])
            lactate_arr=np.array(df_pid["ext_not_ready_lactate"])

            ext_failure_arr=np.array(df_pid["ext_failure"])
            vent_period_arr=np.array(df_pid["vent_period"])

            for jdx in range(trace_arr.size-7):

                # Not inside ventilation period
                if not vent_period_arr[jdx]==1:
                    continue
                
                search_arr=ext_failure_arr[min(jdx+6,ext_failure_arr.size):min(jdx+18,ext_failure_arr.size)]
                X_sample=np.array([vent_mode_arr[jdx],peep_arr[jdx],psupport_arr[jdx],fio2_arr[jdx],sbidx_arr[jdx],
                          rr_arr[jdx],minvol_arr[jdx],pfratio_arr[jdx],paco2_arr[jdx],gcs_arr[jdx],
                          map_arr[jdx],drugs_arr[jdx],lactate_arr[jdx]])
                if np.isfinite(search_arr).any():
                    y_sample=1
                else:
                    y_sample=0
                y_arr.append(y_sample)
                X_arr.append(X_sample)

    # Collected data
    y_arr=np.array(y_arr)
    X_arr=np.vstack(X_arr)

    X_arr[np.isnan(X_arr)]=0

    X_train=X_arr[:int(0.7*X_arr.shape[0]),:]
    y_train=y_arr[:int(0.7*y_arr.size)]

    print("Training matrix dimension: {}".format(X_train.shape))

    X_val=X_arr[int(0.7*X_arr.shape[0]):int(0.85*X_arr.shape[0]),:]
    y_val=y_arr[int(0.7*y_arr.size):int(0.85*y_arr.size)]

    X_test=X_arr[int(0.9*X_arr.shape[0]):,:]
    y_test=y_arr[int(0.9*y_arr.size):]

    alpha_grid=[1.0,0.1,0.01,0.001,0.0001]
    best_score=-np.inf
    best_alpha=None

    for alpha in alpha_grid:
        cand_model=sklm.SGDClassifier(loss="log",penalty="l2",class_weight="balanced",alpha=alpha)
        cand_model.fit(X_train,y_train)
        pred_val=cand_model.predict_proba(X_val)[:,1]
        auroc_score=skmetrics.roc_auc_score(y_val,pred_val)
        if auroc_score>best_score:
            best_alpha=alpha

    best_model=sklm.SGDClassifier(loss="log",penalty="l2",class_weight="balanced",alpha=best_alpha)
    best_model.fit(X_train,y_train)

    pred_test=best_model.predict_proba(X_test)[:,1]

    print("Test AUROC: {:.3f}".format(skmetrics.roc_auc_score(y_test,pred_test)))
    print("Test AUPRC: {:.3f}".format(skmetrics.average_precision_score(y_test,pred_test)))    

    precs,recs,thresholds=skmetrics.precision_recall_curve(y_test,pred_test)
    for jdx in range(thresholds.size):
        if recs[jdx]>=0.9 and recs[jdx+1]<0.9:
            print("Threshold 90 % recall: {:.3f}".format(thresholds[jdx]))
    for jdx in range(thresholds.size):
        if recs[jdx]>=0.8 and recs[jdx+1]<0.8:
            print("Threshold 80 % recall: {:.3f}".format(thresholds[jdx]))

    labels=["VentMode","PEEP","Pressure support","FiO2","SbIDX","RR",
            "MinVol","PF ratio","PaCO2","GCS","MAP","Drugs","Lactate"]

    model_coeffs=best_model.coef_

    for lidx,label in enumerate(labels):
        print("Label: {}, Abs weight: {:.3f}".format(label,model_coeffs[0,lidx]))

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--endpoint_path", default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112")

    # Output paths

    # Arguments
    parser.add_argument("--small_sample", default=True, action="store_true")
    parser.add_argument("--random_state", type=int, default=2021)

    configs=vars(parser.parse_args())

    execute(configs)
