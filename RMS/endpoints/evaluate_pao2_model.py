''' Evaluate the PaO2 model and produce evaluation plots for
    the extended data figures'''

import os
import os.path
import argparse
import ipdb
import pickle
import math
import random
import sys

import pandas as pd
import numpy as np
import sklearn.preprocessing as skpproc

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt


def ellis(x):
    x=x/100
    x[x==1]=0.999
    exp_base = (11700/((1/x)-1))
    exp_sqrbracket = np.sqrt(pow(50,3)+(exp_base**2))
    exp_first = np.cbrt(exp_base + exp_sqrbracket)
    exp_second = np.cbrt(exp_base - exp_sqrbracket)
    exp_full = exp_first + exp_second
    return exp_full    

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

def execute(configs):
    var_map=configs["VAR_IDS"]

    # Load surrogate PaO2 regression model
    with open(configs["sur_reg_model_path"],'rb') as fp:
        model_dict=pickle.load(fp)
        sur_reg_base_model=model_dict["reg_base_model"]
        sur_meta_model=model_dict["meta_model"]
        sur_base_scaler=model_dict["base_scaler"]
        sur_meta_scaler=model_dict["meta_scaler"]

    # Loop over all patients in the test set, apply the model per patient,
    # estimate MAE at the time-points of real PaO2 measurements.
    with open(configs["split_path"],'rb') as fp:
        test_desc=pickle.load(fp)

    test_pids=test_desc[configs["temp_split"]]["test"]

    with open(configs["batch_desc"],'rb') as fp:
        batch_map=pickle.load(fp)["pid_to_chunk"]

    all_errors=[]
    all_baseline_errors=[]
    all_pop_errors=[]

    random.shuffle(test_pids)

    for pix,pid in enumerate(test_pids):
        if (pix+1)%100==0:
            print("{}/{} PIDs".format(pix+1,len(test_pids)))
            print("Model MAE: {:.3f} [{:.3f},{:.3f}]".format(np.median(all_errors),np.percentile(all_errors,25), np.percentile(all_errors,75)))
            print("Pop. Model MAE: {:.3f} [{:.3f},{:.3f}]" .format(np.median(all_pop_errors), np.percentile(all_pop_errors,25), np.percentile(all_pop_errors,75)))
            print("Baseline MAE: {:.3f} [{:.3f},{:.3f}]".format(np.median(all_baseline_errors), np.percentile(all_baseline_errors,25), np.percentile(all_baseline_errors,75)))
            
        batch_pid=batch_map[pid]
        df_pid=pd.read_hdf(os.path.join(configs["imputed_path"],configs["temp_split"],"batch_{}.h5".format(batch_pid)),
                           mode='r',where="PatientID={}".format(pid))
        fio2_col=np.array(df_pid[var_map["FiO2"]])
        spo2_col=np.array(df_pid[var_map["SpO2"]])
        pao2_col=np.array(df_pid[var_map["PaO2"]])
        sao2_col=np.array(df_pid[var_map["SaO2"]])
        ph_col=np.array(df_pid[var_map["pH"]])
        spo2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SpO2"])])
        pao2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["PaO2"])])        
        sao2_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["SaO2"])])
        ph_meas_cnt=np.array(df_pid["{}_IMPUTED_STATUS_CUM_COUNT".format(var_map["pH"])])        
        pao2_sur_est, pao2_pop_est = apply_regression_model(spo2_col, spo2_meas_cnt,
                                                            pao2_col, pao2_meas_cnt,
                                                            sao2_col, sao2_meas_cnt,
                                                            ph_col, ph_meas_cnt,
                                                            sur_reg_base_model, sur_meta_model,
                                                            sur_base_scaler, sur_meta_scaler)

        pat_errors=[]
        pat_baseline_errors=[]

        time_pao2=[]
        pred_pao2=[]
        true_pao2=[]
        baseline_pao2=[]

        for idx in range(1,len(pao2_meas_cnt)):
            if pao2_meas_cnt[idx]-pao2_meas_cnt[idx-1]>0:
                abs_error=abs(pao2_sur_est[idx]-pao2_col[idx])
                abs_pop_error=abs(pao2_pop_est[idx]-pao2_col[idx])
                prev_spo2=spo2_col[idx]
                baseline_pred=ellis(np.array([prev_spo2]))[0]
                baseline_pao2.append(baseline_pred)
                pred_pao2.append(pao2_sur_est[idx])
                true_pao2.append(pao2_col[idx])
                baseline_error=abs(baseline_pred-pao2_col[idx])
                all_errors.append(abs_error)
                all_baseline_errors.append(baseline_error)
                pat_errors.append(abs_error)
                pat_baseline_errors.append(baseline_error)
                all_pop_errors.append(abs_pop_error)
                time_pao2.append(5*idx)

        if len(pat_errors)>=15:
            if abs(np.median(pat_errors)-6.25)<=1 and abs(np.median(pat_baseline_errors)-12.90)<=1:
                print("Found typical patient...")
                time_axis=5*np.arange(len(pao2_col))
                plt.plot(time_axis,spo2_col,label="SpO2")
                plt.plot(time_pao2,baseline_pao2,'.',label="Pred. PaO2 baseline")
                plt.plot(time_pao2,pred_pao2,'+',label="Pred. PaO2 model")
                plt.plot(time_pao2,true_pao2,'x',label="PaO2 (Ground-truth)")
                plt.xlabel("Time since ICU admission [mins]")
                plt.ylabel("SpO2 [%] or PaO2 [mmHg] value ")
                plt.legend()
                plt.savefig(os.path.join(configs["plot_path"],"pao2_model_example.png"),dpi=400)
                plt.savefig(os.path.join(configs["plot_path"],"pao2_model_example.pdf"))
                sys.exit(0)
                

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--sur_reg_model_path", default="../../data/endpoints/pao2_est_models.pickle",
                        help="PaO2 estimation model")

    parser.add_argument("--imputed_path", default="../../data/imputed/impute_hirid2/reduced",
                        help="Imputed data-path")

    parser.add_argument("--split_path", default="../../data/exp_design/temp_splits_hirid2.pickle",
                        help="Split descriptor path")

    parser.add_argument("--batch_desc", default="../../data/exp_design/hirid2_chunking_100.pickle",
                        help="Split descriptor path")

    # Output paths

    parser.add_argument("--plot_path", default="../../data/plots/pao2_estimation",
                        help="Plot output path")

    # Arguments
    parser.add_argument("--temp_split", default="temporal_1", help="Split to evaluate")

    configs=vars(parser.parse_args())

    configs["VAR_IDS"]={"PaO2":"vm140","SpO2": "vm20","SaO2": "vm141","pH": "vm138",
                        "FiO2": "vm58"}

    execute(configs)
