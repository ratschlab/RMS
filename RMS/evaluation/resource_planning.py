#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np
import pickle
import h5py
import gc


from os import listdir, makedirs
from os.path import join, exists


from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from lightgbm import LGBMRegressor
import lightgbm as lgb


import shap


from precision_recall import read_mh_pd, read_ep_resp, align_time


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", choices=["ml", "feature_extraction"])
    parser.add_argument("--dataset", default="HiRID", choices=["HiRID", "UMC"])
    parser.add_argument("--split_name", default="point_est")
    
    parser.add_argument("--feature_dir", default="ml_features_one_hour")
    parser.add_argument("--prediction_horizon", nargs="+", type=int, default=[4,8])
    parser.add_argument("--min_ventilation_min", type=int, default=5)
    parser.add_argument("--rf_res", default=".")
    
    parser.add_argument("--num_leaves", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--stopping_rounds", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--max_ffill", type=int, default=1)
    parser.add_argument("--only_emergency", action="store_true")
    parser.add_argument("--predict_emergency", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument("--selected_set", default=None)
    parser.add_argument("--k_anonymous", action="store_true")
    args = parser.parse_args()
    
    mode = args.mode
    dataset = args.dataset
    
    feature_dir = args.feature_dir
    prediction_horizon = args.prediction_horizon
    rf_res = args.rf_res
    min_ventilation_min = args.min_ventilation_min
    num_leaves = args.num_leaves
    num_rounds = args.num_rounds
    stopping_rounds = args.stopping_rounds
    random_state = args.random_state
    max_ffill = args.max_ffill
     
    only_emergency = args.only_emergency
    predict_emergency = args.predict_emergency
    selected_set = args.selected_set
    DEBUG = args.DEBUG    
    k_anonymous = args.k_anonymous
    split_name = args.split_name


#     mode = "ml"
#     dataset = "HiRID"
    
#     feature_dir = "tmp"
#     prediction_horizon = [4,8] 
#     rf_res = "."
#     min_ventilation_min = 10
#     num_leaves = 32
#     num_rounds = 1000
#     stopping_rounds = 100
#     random_state = 2010
#     max_ffill = 1
     
#     only_emergency = False
#     predict_emergency = False
#     selected_set = "val"
#     DEBUG = True
#     k_anonymous = True
#     split_name = "temporal_1"


    # hirid data path
    hirid_path = "/cluster/work/grlab/clinical/hirid2/research"

    feature_path = join(hirid_path, "vent_planning", feature_dir)
    if not exists(feature_path):
        makedirs(feature_path)
    if not exists(rf_res):
        makedirs(rf_res)
    # if not exists(rf_res+"_figs"):
    #     makedirs(rf_res+"_figs")

    win_size = prediction_horizon[1] - prediction_horizon[0]
    feature_cols = ["Prediction_vent", "Prediction_extu", "Prediction_fail", "Prediction_exfa", "VentUse", "weekday", "total_vent_use", "total_patient", "idx_wind"]
    elective_pids = pd.read_hdf("/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5")[["PatientID", "Emergency"]]
    elective_pids = np.sort(elective_pids[elective_pids.Emergency==0].PatientID.unique())

    lst_features = np.concatenate([ ["min_%s"%col for col in feature_cols[:4]], 
                                    ["p25_%s"%col for col in feature_cols[:4]],
                                    ["median_%s"%col for col in feature_cols[:4]],
                                    ["p75_%s"%col for col in feature_cols[:4]],
                                    ["max_%s"%col for col in feature_cols[:4]],
                                    ["current_%s"%col for col in feature_cols[:4]],
                                    ["vent_last_hour", "vent_since_admission"],
                                    ["min_total_vent", "max_total_vent"],
                                    ["min_total_patient", "max_total_patient"],
                                    ["weekday", "hourOftheDay","timeSinceAdmission"]])



    if mode=="feature_extraction": # Extract features
        hirid_path = "/cluster/work/grlab/clinical/hirid2/research/RESP_RELEASE"
        hirid_pd_path = join(hirid_path, "predictions", "reduced")
        hirid_ep_path = join(hirid_path, "endpoints", "hirid2_endpoints")

        # umc data path
        umc_path = "/cluster/work/grlab/clinical/umcdb/preprocessed"
        umc_pd_path = join(umc_path,"8c_predictions_resp","211007","reduced")
        umc_ep_path = join(umc_path,"3c_endpoints_resp","endpoints_210630")

        if dataset=="UMC":
            pid_map_file = "temp_splits_UMCDB_210630.pickle"
            split_name = "exploration_1"
            ep_path = join(umc_ep_path, split_name) # endpoint path
            fail_path = join(umc_pd_path,"point_est","Label_WorseStateFromZeroOrOne0To24Hours_retrain_lightgbm") # path to prediction scores for respiratory failure
            vent_path = join(umc_pd_path,"point_est","Label_VentilationMulticlass_retrain_lightgbm") # path to prediction scores for ventilation
            extu_path = join(umc_pd_path,"point_est","Label_ReadyExtubateMulticlass_retrain_lightgbm") # path to prediction scores for readiness to extubate

        else:
            ep_path = join(hirid_ep_path, split_name)
            pid_map_file = join(hirid_path, "exp_design", "temp_splits_hirid2.pickle")

            fail_path = join(hirid_pd_path,split_name,"Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm")
            vent_path = join(hirid_pd_path,split_name,"Label_Ventilation0To24Hours_internal_rmsVENT_lightgbm")
            extu_path = join(hirid_pd_path,split_name,"Label_ReadyExtubate0To24Hours_internal_rmsREXP_lightgbm")
            exfa_path = join(hirid_pd_path,split_name,"Label_ExtubationFailureSimple_internal_rmsEF_lightgbm")

        with open(pid_map_file, "rb") as tmp:
            pid_mapping = pickle.load(tmp) # reading patient-split mapping

        win_size = prediction_horizon[1] - prediction_horizon[0]
        lst_dset = ["train", "val", "test"] if selected_set is None else [selected_set]

        for dset in lst_dset:
            lst_pids = pid_mapping[split_name][dset]
            print(dset)
            if dset=="train":
                ep_all = read_ep_resp(ep_path, ["batch_%d.h5"%n for n in np.arange(45,60)], "ventilation")
                ep_all = ep_all[ep_all.PatientID.isin(lst_pids)]  
                ep_all = ep_all.rename(columns={col: col+"_vent" for col in ep_all.columns[2:]})
                pd_all = pd.DataFrame(np.zeros((ep_all.shape[0],4)), columns=feature_cols[:4])
                pd_all.loc[:,'PatientID'] = ep_all.PatientID
                pd_all.loc[:,'Datetime'] = ep_all.Datetime
                pd_all = pd_all[["PatientID","Datetime"]+feature_cols[:4]]
            else:
                lst_files = []
                for f in listdir(extu_path):
                    if ".h5" not in f:
                        continue
                    with h5py.File(join(extu_path, f), "r") as tmp:
                        lst_pid_f = [int(k[1:]) for k in tmp.keys()]
                    if len(set(lst_pid_f)&set(lst_pids))  > 0:
                        lst_files.append(f)
                lst_files = np.array(lst_files)[np.argsort([int(f.split("_")[1][:-3]) for f in lst_files])]
                if DEBUG:
                    idx_batch = 0
                    lst_files = lst_files[idx_batch:idx_batch+1]
                print(lst_files)
                ep_all = read_ep_resp(ep_path, lst_files, "ventilation")
                ep_all = ep_all.rename(columns={col: col+"_vent" for col in ep_all.columns[2:]})
                pd_fail = read_mh_pd(fail_path, lst_files).rename(columns={"Prediction": "Prediction_fail"})
                pd_vent = read_mh_pd(vent_path, lst_files).rename(columns={"Prediction": "Prediction_vent"})
                pd_extu = read_mh_pd(extu_path, lst_files).rename(columns={"Prediction": "Prediction_extu"})
                pd_exfa = read_mh_pd(exfa_path, lst_files).rename(columns={"Prediction": "Prediction_exfa"})
                pd_exfa.loc[:,"Prediction_exfa"] = pd_exfa.loc[:,"Prediction_exfa"].fillna(0)

                for df in [pd_vent, pd_extu, pd_fail, pd_exfa, ep_all]:
                    df.set_index(["PatientID", "Datetime"], inplace=True)
                pd_all = pd.concat([pd_fail,pd_vent,pd_extu,pd_exfa], axis=1, join="inner").reset_index()
                ep_all = ep_all.reset_index()

            if dset=="train":
                lst_pids = np.array(lst_pids)[np.isin(lst_pids, ep_all.PatientID.unique())]


            ep_all.loc[:,'Datetime'] = ep_all.Datetime.dt.ceil('5T')


            ep_all.loc[:,'Elective'] = False
            ep_all.loc[ep_all.index[ep_all.PatientID.isin(elective_pids)],"Elective"] = True

            ep_all.loc[:,'Datetime_h'] = ep_all.Datetime.dt.floor('H')
            
            icu_bed_all = ep_all[["Datetime_h","PatientID"]].drop_duplicates(["Datetime_h","PatientID"]).groupby("Datetime_h").agg({"PatientID": lambda x: set(x)})
            vent_all = ep_all[ep_all.InEvent_vent][["Datetime_h","PatientID"]].drop_duplicates(["Datetime_h","PatientID"]).groupby("Datetime_h").agg({"PatientID": lambda x: set(x)})
            vent_elec = ep_all[(ep_all.Elective)&(ep_all.InEvent_vent)][["Datetime_h","PatientID"]].drop_duplicates(["Datetime_h","PatientID"]).groupby("Datetime_h").agg({"PatientID": lambda x: set(x)})
            vent_emer = ep_all[(~ep_all.Elective)&(ep_all.InEvent_vent)][["Datetime_h","PatientID"]].drop_duplicates(["Datetime_h","PatientID"]).groupby("Datetime_h").agg({"PatientID": lambda x: set(x)})
 
            icu_bed_all = icu_bed_all.rename(columns={"PatientID": "ICUBed"})
            
            vent_elec = vent_elec.rename(columns={"PatientID": "VentUse_elective"})
            vent_emer = vent_emer.rename(columns={"PatientID": "VentUse_emergency"})
            vent_all = vent_all.rename(columns={"PatientID": "VentUse"})
            vent = pd.concat([vent_all, vent_emer, vent_elec], axis=1)
            for col in ["VentUse_emergency", "VentUse_elective"]:
                tmp = []
                tmp_pid = []
                for dt in vent.index:
                    dt_inXh = dt+np.timedelta64(int(prediction_horizon[0]),"h")
                    dt_inYh = dt+np.timedelta64(int(prediction_horizon[1]),"h")
                    # tmp.append(vent.loc[dt_inXh:dt_inYh][col].max())
                    if len(vent.loc[dt_inXh:dt_inYh][col].dropna()) > 0:
                        if type(vent.loc[dt,"VentUse"])==set:
                            tmp.append(len(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values) - vent.loc[dt,"VentUse"]))
                            tmp_pid.append(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values) - vent.loc[dt,"VentUse"])
                        else:
                            tmp.append(len(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values)))
                            tmp_pid.append(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values))
                    else:
                        tmp.append(np.nan)
                        tmp_pid.append(set())

                vent.loc[:,"{}_future".format(col)] = tmp   
                vent.loc[:,"{}_future_pid".format(col)] = tmp_pid

        
            for col in ["VentUse"]:
                tmp = []
                tmp_pid = []
                for dt in vent.index:
                    dt_inXh = dt+np.timedelta64(int(prediction_horizon[0]),"h")
                    dt_inYh = dt+np.timedelta64(int(prediction_horizon[1]),"h")
                    # tmp.append(vent.loc[dt_inXh:dt_inYh][col].max())
                    if len(vent.loc[dt_inXh:dt_inYh][col].dropna()) > 0:
                        if type(vent.loc[dt,"VentUse"])==set:
                            tmp.append(len(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values) & vent.loc[dt,"VentUse"]))
                            tmp_pid.append(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values) & vent.loc[dt,"VentUse"])
                        else:
                            tmp.append(np.nan)
                            tmp_pid.append(set())
                    else:
                        tmp.append(np.nan)
                        tmp_pid.append(set())
                vent.loc[: ,"{}_icu_future".format(col)] = tmp                            
                vent.loc[: ,"{}_icu_future_pid".format(col)] = tmp_pid

            for col in ["VentUse"]:
                tmp = []
                tmp_pid = []
                for dt in vent.index:
                    dt_inXh = dt+np.timedelta64(int(prediction_horizon[0]),"h")
                    dt_inYh = dt+np.timedelta64(int(prediction_horizon[1]),"h")
                    if len(vent.loc[dt_inXh:dt_inYh][col].dropna()) > 0:
                        tmp.append(len(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values)))
                        tmp_pid.append(set.union(*vent.loc[dt_inXh:dt_inYh][col].dropna().values))
                    else:
                        tmp.append(np.nan)
                        tmp_pid.append(set())
                vent.loc[: ,"{}_future".format(col)] = tmp                            
                vent.loc[: ,"{}_future_pid".format(col)] = tmp_pid

            pd_all.loc[:,'Datetime'] = pd_all.Datetime.dt.ceil('5T')
            pd_all.loc[:,'Datetime_h'] = pd_all.Datetime.dt.floor('H')
            min_score = pd_all.drop(columns=["Datetime"]).groupby(["PatientID", "Datetime_h"]).min()
            p25_score = pd_all.drop(columns=["Datetime"]).groupby(["PatientID", "Datetime_h"]).quantile(q=0.25)
            med_score = pd_all.drop(columns=["Datetime"]).groupby(["PatientID", "Datetime_h"]).median()
            p75_score = pd_all.drop(columns=["Datetime"]).groupby(["PatientID", "Datetime_h"]).quantile(q=0.75)
            max_score = pd_all.drop(columns=["Datetime"]).groupby(["PatientID", "Datetime_h"]).max()
            cur_score = pd_all.drop(columns=["Datetime"]).drop_duplicates(["PatientID", "Datetime_h"], keep='last').set_index(["PatientID", "Datetime_h"])
            vent_time = ep_all[["PatientID", "Datetime_h", "InEvent_vent"]].groupby(["PatientID", "Datetime_h"]).sum() 
            all_time = ep_all[["PatientID", "Datetime_h", "InEvent_vent"]].groupby(["PatientID", "Datetime_h"]).count()
            vent_frac_win = vent_time / all_time
            vent_frac_his = vent_time.cumsum() / all_time.cumsum()
            pat_feat = pd.concat([min_score, p25_score, med_score, p75_score, max_score, cur_score, vent_frac_win, vent_frac_his], axis=1)
            pat_feat = pd.DataFrame(pat_feat.values, columns=lst_features[:26], index=pat_feat.index)

            vent_all_5min =  ep_all[ep_all.InEvent_vent][["Datetime","PatientID"]].drop_duplicates(["Datetime","PatientID"]).groupby("Datetime").count()
            vent_all_5min = vent_all_5min.reset_index()
            vent_all_5min.loc[:,'Datetime'] = vent_all_5min.Datetime.dt.floor('H')
            vent_all_min = vent_all_5min.groupby('Datetime').min()
            vent_all_max = vent_all_5min.groupby('Datetime').max()


            pat_all_5min = ep_all[["Datetime","PatientID"]].drop_duplicates(["Datetime","PatientID"]).groupby("Datetime").count()
            pat_all_5min = pat_all_5min.reset_index()
            pat_all_5min.loc[:,'Datetime'] = pat_all_5min.Datetime.dt.floor('H')
            pat_all_min = pat_all_5min.groupby('Datetime').min()
            pat_all_max = pat_all_5min.groupby('Datetime').max()
            icu_feat = pd.concat([vent_all_min, vent_all_max, pat_all_min, pat_all_max], axis=1)
            icu_feat = pd.DataFrame(icu_feat.values, columns=lst_features[26:-3], index=icu_feat.index)
            icu_feat.loc[:,'weekday'] = icu_feat.index.weekday
            icu_feat.loc[:,'hourOftheDay'] = icu_feat.index.hour

            pat_feat = pat_feat.reset_index().rename(columns={'Datetime_h':'Datetime'})
            icu_feat = icu_feat.reset_index()
            pat_feat = pat_feat.merge(icu_feat, how='outer', left_on='Datetime', right_on='Datetime')
            pat_feat = pat_feat.sort_values(['PatientID','Datetime'])
            adm_time = pat_feat[['PatientID', 'Datetime']].drop_duplicates(['PatientID'], keep='first').rename(columns={'Datetime':'Admtime'})
            pat_feat = pat_feat.merge(adm_time, left_on='PatientID', right_on='PatientID', how='outer')
            pat_feat.loc[:,lst_features[-1]] = (pat_feat.Datetime - pat_feat.Admtime)/np.timedelta64(1,'h')
            pat_feat = pat_feat.drop(columns=['Admtime'])
            pat_feat = pat_feat.merge(vent[["VentUse_future_pid"]], left_on="Datetime", right_index=True, how="outer")
            pat_feat.loc[:,"Label"] = 0
            pat_feat.loc[pat_feat.index[pat_feat["VentUse_future_pid"].notnull()],'Label'] = pat_feat.loc[pat_feat.index[pat_feat["VentUse_future_pid"].notnull()]].apply(lambda row: row["PatientID"] in row["VentUse_future_pid"], axis=1).astype(float)
            pat_feat = pat_feat.drop(columns=['VentUse_future_pid'])
            pat_feat.to_hdf(join(feature_path, "%s_%d_%d.h5"%(dset, prediction_horizon[0], prediction_horizon[1])), 
                                "data", complib="blosc:lz4", complevel=5, format="table", data_columns=True)
            
            vent.loc[:,"VentUse"] = vent["VentUse"].apply(lambda x: len(x))
            icu_bed_all.loc[:,"ICUBed"] = icu_bed_all["ICUBed"].apply(lambda x: len(x))
            
            vent = vent[["VentUse", "VentUse_emergency_future", "VentUse_elective_future", "VentUse_icu_future", "VentUse_future"]]

            vent.to_hdf(join(feature_path, "label_%s_%d_%d.h5"%(dset, prediction_horizon[0], prediction_horizon[1])), "data",
                        complib="blosc:lz4", complevel=5, format="table", data_columns=True)

            icu_bed_all.to_hdf(join(feature_path, "icu_bed_%s_%d_%d.h5"%(dset, prediction_horizon[0], prediction_horizon[1])), 
                               "data", complib="blosc:lz4", complevel=5, format="table", data_columns=True)


    elif mode=="ml": # Extract features

        if predict_emergency:
            df_train = pd.read_hdf(join(feature_path, "train_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))
            df_val = pd.read_hdf(join(feature_path, "val_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))
            df_test = pd.read_hdf(join(feature_path, "test_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))

            lbl_train = pd.read_hdf(join(feature_path, "label_train_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))
            lbl_val = pd.read_hdf(join(feature_path, "label_val_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))
            lbl_test = pd.read_hdf(join(feature_path, "label_test_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))

            df_train.loc[:,"weekday"] = df_train.weekday.apply(lambda x: 1 if x<=4 else 0)
        else:
            df_val = pd.read_hdf(join(feature_path, "val_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))
            df_test = pd.read_hdf(join(feature_path, "test_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))

            lbl_val = pd.read_hdf(join(feature_path, "label_val_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))
            lbl_test = pd.read_hdf(join(feature_path, "label_test_%d_%d.h5"%(prediction_horizon[0], prediction_horizon[1])))

        df_val.loc[:,"weekday"] = df_val.weekday.apply(lambda x: 1 if x<=4 else 0) # weekday or not
        df_test.loc[:,"weekday"] = df_test.weekday.apply(lambda x: 1 if x<=4 else 0)

        elective_pids = pd.read_hdf("/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5")[["PatientID", "Emergency"]]
        elective_pids = np.sort(elective_pids[elective_pids.Emergency==0].PatientID.unique())

        gc.collect()
        lst_features = np.concatenate([ ["min_%s"%col for col in feature_cols[:4]], 
                                        ["p25_%s"%col for col in feature_cols[:4]],
                                        ["median_%s"%col for col in feature_cols[:4]],
                                        ["p75_%s"%col for col in feature_cols[:4]],
                                        ["max_%s"%col for col in feature_cols[:4]],
                                        ["current_%s"%col for col in feature_cols[:4]],
                                        ["vent_last_hour", "vent_since_admission"],
                                        ["min_total_vent", "max_total_vent"],
                                        ["min_total_patient", "max_total_patient"],
                                        ["weekday", "hourOftheDay","timeSinceAdmission"]])


        if predict_emergency:
            lst_features = lst_features[26:-1]
            print(lst_features)
            label_emergency = dict()
            feats_emergency = dict()
            lst_dt = dict()
            for dset, df, lbl in [("train", df_train, lbl_train), ("val", df_val, lbl_val), ("test", df_test, lbl_test)]:
                label_emergency.update({dset: lbl["VentUse_emergency_future"].fillna(0)})

                tmp = df[np.concatenate((['Datetime'],lst_features))].drop_duplicates()
                tmp = tmp[tmp.notnull().all(axis=1)]
                assert(len(tmp)==len(lbl))

                feats_emergency.update({dset: tmp[lst_features]})
                lst_dt.update({dset: tmp['Datetime'].values})

            param = {'num_leaves': num_leaves, 'objective': 'regression_l2', "seed": random_state}
            param['metric'] = 'l2'
            train_data =  lgb.Dataset(feats_emergency["train"].values, label=label_emergency["train"].values)
            val_data =  lgb.Dataset(feats_emergency["val"].values, label=label_emergency["val"].values)
            num_rounds = num_rounds
            reg = lgb.train(param, train_data, num_rounds, valid_sets=val_data, callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds)])
            median_gt_train = np.nanmedian(label_emergency["train"].values)
            mean_gt_train = np.nanmedian(label_emergency["train"].values)
            for dset, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
                pred = reg.predict(feats_emergency[dset].values)
                gt = label_emergency[dset].values
                MAE = np.mean(np.abs(gt-pred))
                MSE = np.mean(np.abs(gt-pred)**2)
                np.savez("%s/res_emer_%s_%d_%d_l%d_r%d_s%d.npz"%(rf_res, dset, prediction_horizon[0], prediction_horizon[1], num_leaves, num_rounds, stopping_rounds), 
                         lst_dt=lst_dt[dset], groundtruth=gt, prediction=pred,
                         MAE=MAE, MSE=MSE, lst_features=lst_features, mean_gt_train=mean_gt_train, median_gt_train=median_gt_train)                

        else:
            ################################################
            ### Training Random Forest with validation data
            ################################################
            df_val_tmp = df_val[df_val[lst_features].notnull().sum(axis=1)>1].fillna(0)
            df_val_pid_unique = df_val_tmp.sort_values(["Datetime"]).drop_duplicates("PatientID", keep='last')
            val_pids = df_val_pid_unique.PatientID.values
            val_val_pids = val_pids[:int(0.8*len(val_pids))]
            val_test_pids = val_pids[int(0.8*len(val_pids)):]
            X_train = df_val_tmp[df_val_tmp.PatientID.isin(val_val_pids)][lst_features].values
            y_train = df_val_tmp[df_val_tmp.PatientID.isin(val_val_pids)]["Label"].values
            X_val = df_val_tmp[df_val_tmp.PatientID.isin(val_test_pids)][lst_features].values
            y_val = df_val_tmp[df_val_tmp.PatientID.isin(val_test_pids)]["Label"].values
            param = {'num_leaves': num_leaves, 'objective': 'cross_entropy', "seed": random_state}
            param['metric'] = 'auc'
            train_data =  lgb.Dataset(X_train, label=y_train)
            val_data =  lgb.Dataset(X_val, label=y_val)
            num_rounds = num_rounds
            clf = lgb.train(param, train_data, num_rounds, valid_sets=val_data, callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds)])

            # feature_importance = base_clf.feature_importances_  
            # prob_val = clf.predict_proba(X_val)[:,1]
            feature_importance = clf.feature_importance()
            prob_val = clf.predict(X_val)
            auprc_val = average_precision_score(y_val, prob_val)
            auroc_val = roc_auc_score(y_val, prob_val)
            prev_val = (y_val==1).sum() / len(y_val)
            explainer = shap.TreeExplainer(clf)
            df_X_val = pd.DataFrame(X_val, columns=lst_features)
            shap_values = explainer.shap_values(df_X_val)
            plt.figure(figsize=(5,5))
            shap.summary_plot(shap_values, df_X_val, show=False)

            # Evaluation
            for dset, df, lbl in [("train", df_val, lbl_val), ("val", df_val, lbl_val), ("test", df_test, lbl_test)]:
                if dset=="val":
                    df = df[df.PatientID.isin(val_test_pids)].copy()
                    lbl = lbl[lbl.index>=df.Datetime.min()]
                elif dset=="train":
                    df = df[df.PatientID.isin(val_val_pids)].copy()
                    lbl = lbl[lbl.index<=df.Datetime.max()]

                prob = clf.predict(df[lst_features].values)
                df.loc[:,"Prediction"] = prob

                df_vent_groundtruth = lbl[["VentUse_icu_future"]]
                df_vent_prediction = df[["PatientID", "Datetime", "Prediction"]].copy()
                df_vent_prediction.loc[:,"Prediction"] = (df_vent_prediction.Prediction>0.5).astype(float)
                df_vent_prediction = df_vent_prediction[df_vent_prediction.Prediction==True][["PatientID", "Datetime"]].groupby("Datetime").agg({"PatientID": lambda x: len(set(x))})
                df_vent_prediction = df_vent_prediction.rename(columns={"PatientID": "VentUse_icu_future_pred"})

                df_vent_ffill = lbl[["VentUse"]].rename(columns={'VentUse': 'VentUse_icu_future_ffill'})


                df_output = pd.concat([df_vent_groundtruth.VentUse_icu_future, 
                                       df_vent_prediction.VentUse_icu_future_pred,
                                       df_vent_ffill.VentUse_icu_future_ffill], join="outer", axis=1)

                MAE_FFill = np.abs(df_output.VentUse_icu_future_ffill[:-prediction_horizon[1]] - df_output.VentUse_icu_future[:-prediction_horizon[1]]).mean()
                MAE_RF = np.abs(df_output.VentUse_icu_future_pred[:-prediction_horizon[1]] - df_output.VentUse_icu_future[:-prediction_horizon[1]]).mean()

                MSE_FFill = (np.abs(df_output.VentUse_icu_future_ffill[:-prediction_horizon[1]] - df_output.VentUse_icu_future[:-prediction_horizon[1]])**2).mean()
                MSE_RF = (np.abs(df_output.VentUse_icu_future_pred[:-prediction_horizon[1]] - df_output.VentUse_icu_future[:-prediction_horizon[1]])**2).mean()

                ffill = df_output.VentUse_icu_future_ffill[:-prediction_horizon[1]].values
                pred = df_output.VentUse_icu_future_pred[:-prediction_horizon[1]].values
                gt = df_output.VentUse_icu_future[:-prediction_horizon[1]].values

                lst_dt = df_output.index[:-prediction_horizon[1]].values
                pred_int = np.round(df_output.VentUse_icu_future_pred[:-prediction_horizon[1]])

                np.savez("%s/res_%s_%d_%d_l%d_r%d_s%d.npz"%(rf_res, dset, prediction_horizon[0], prediction_horizon[1], num_leaves, num_rounds, stopping_rounds), 
                         lst_dt=lst_dt, groundtruth=gt, prediction=pred, prediction_ffill=ffill, prediction_int=pred_int,
                         val_val_auprc=auprc_val,  val_val_auroc=auroc_val, val_val_prev=prev_val,
                         MAE_baseline=MAE_FFill,MAE_RF=MAE_RF,MSE_baseline=MSE_FFill,MSE_RF=MSE_RF, feature_importance=feature_importance, lst_features=lst_features)

                if dset=="train":
                    df.to_hdf("%s/res_%s_%d_%d_l%d_r%d_s%d.h5"%(rf_res, dset, prediction_horizon[0], prediction_horizon[1], num_leaves, num_rounds, stopping_rounds),"data", 
                              format="table", complib="blosc:lz4", complevel=5)

