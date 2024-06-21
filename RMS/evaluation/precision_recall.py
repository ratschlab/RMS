import os
import gc
import sys
import h5py
import pickle 
import pandas as pd
import numpy as np
from sklearn import metrics 
from functools import reduce

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn'])
sns.set_palette("tab10")

import utils_plot

import utils
    
import ipdb

use_cols = ["PatientID", "Datetime", "Prediction"]

ABS_DT_COL = 'AbsDatetime'
REL_DT_COL = 'RelDatetime'
DT_COL = "Datetime"
PID_COL = "PatientID"
LBL_COL = "Label"
PD_COL = "Prediction"
FURO_ID = "pm69"
SPO2_ID = "vm20"
FIO2_ID = "vm58"
IMPUTE_DATA_PATH = "/cluster/work/grlab/clinical/hirid2/research/5c_imputed_resp/210407_noimpute/reduced/point_est"
STATIC_DATA_PATH = "/cluster/work/grlab/clinical/hirid2/research/1a_hdf5_clean/v8/static.h5"


def read_furosemide_records(datapath,
                            batch_name):
    
    df_iter = pd.read_hdf(os.path.join(datapath, batch_name),
                          chunksize=10**5,
                          mode="r",
                          usecols=[PID_COL, ABS_DT_COL, REL_DT_COL, FURO_ID])
    
    df = []    
    for tmp in df_iter:
        pids_furo = tmp[tmp[FURO_ID]>0].PatientID.unique() # patients with positive furosemide drug records
        if len(pids_furo) > 0:
            df.append(tmp[tmp.PatientID.isin(pids_furo)])
        gc.collect()
    df = pd.concat(df)
    
    df = df.rename(columns={ABS_DT_COL: DT_COL})
    df = df.sort_values([PID_COL, DT_COL])
    df = df.reset_index(drop=True)
    return df


def read_mh_pd(datapath, lst_batches, calibrated=False):
    """
    Reading prediction scores from Matthias' prediction folder
    """
    if calibrated:
        df = pd.read_hdf(datapath, mode="r")
        df = df.rename(columns={ABS_DT_COL: DT_COL, 
                                "TrueLabel": LBL_COL, 
                                "CalibratedPredScore": PD_COL})
        df = df[[PID_COL, DT_COL, PD_COL]]
        df = df.sort_values([PID_COL, DT_COL])
        df = df.reset_index(drop=True)
        
    else:
        df = []

        for i, b in enumerate(lst_batches):
            batch_path = os.path.join(datapath, b)
            with h5py.File(batch_path, mode="r") as f:
                lst_pids = [k for k in f.keys()]
            num_pids = len(lst_pids)

            for j, pid in enumerate(lst_pids):
                try:
                    df_p = pd.read_hdf(batch_path, pid, mode="r")
                except:
                    ipdb.set_trace()
                    
                df_p = df_p.rename(columns={ABS_DT_COL: DT_COL, 
                                            "TrueLabel": LBL_COL, 
                                            "PredScore": PD_COL})
                
                if PD_COL not in df_p.columns:
                    df_p = df_p.rename(columns={"PredScore_0": PD_COL})
                    
                # if the time interval is not 5 minutes, debug
                assert( df_p.Datetime.dt.floor("5T").duplicated().sum()==0 ) 
                df_p = df_p.sort_values(DT_COL)
                df.append(df_p[[PID_COL, DT_COL, PD_COL]])
                gc.collect()

                sys.stdout.write("Reading Matthias's prediction results: ") 
                sys.stdout.write("Batch %02d, Patient %05d / %05d \r"%(i+1, j+1, num_pids))
                sys.stdout.flush()
                
        df = pd.concat(df, axis=0)
        df = df.reset_index(drop=True)
        
    print("\nFinished reading Matthias' prediction results")    
    return df


def read_bw_pd(datapath,
               prediction_col):
    """
    Reading prediction scores from Bowen's prediction folder
    """
    
    if ".csv" in datapath:
        df = pd.read_csv(datapath, index_col=0)
        df = df.reset_index()
        df = df.rename(columns={ABS_DT_COL: DT_COL, 
                                "risk": PD_COL,
                                "adverse_risk_within_48h": PD_COL})
        
    else:
        if ".h5" in datapath:
            df = pd.read_hdf(datapath)
            df = df.reset_index()
            
        elif ".parquet" in datapath:
            df = pd.read_parquet(datapath)
            df = df.reset_index()
            
        deepmind_model = "deepmind" in datapath.lower()
        if deepmind_model:
            df = df.rename(columns={"Type %d"%i: "Type%d"%i for i in np.arange(1,5)})
            df = df.rename(columns={ABS_DT_COL: DT_COL, 
                                    "timestamp": REL_DT_COL, 
                                    prediction_col: PD_COL})
        else:
            df = df.rename(columns={ABS_DT_COL: DT_COL, 
                                    "label": LBL_COL, 
                                    "pred": PD_COL, 
                                    "score": PD_COL,
                                    "risk": PD_COL})

    df.loc[:,DT_COL] = df[DT_COL].astype(np.datetime64)
    df = df.sort_values([PID_COL, DT_COL])
    df = df.reset_index(drop=True)

    if REL_DT_COL not in df.columns:
        df.loc[:,REL_DT_COL] = 0
        for pid in df[PID_COL].unique():
            idx_p = df.index[df[PID_COL]==pid]
            df.loc[idx_p,REL_DT_COL] = df.loc[idx_p,DT_COL] - df.loc[idx_p[0],DT_COL]
        df.loc[:,REL_DT_COL] /= np.timedelta64(1,"s")
    print("\nFinished reading Bowen's prediction results.")
    
    return df[[PID_COL, DT_COL, PD_COL]]


def read_tg_pd(datapath,
               lst_batches,
               prediction_col=None):
    """
    Reading prediction scores from Thomas' prediction folder
    """    
    df = []
    for i, b in enumerate(lst_batches):
        df_b = pd.read_hdf(os.path.join(datapath, b))
        df_b = df_b.rename(columns={"geq1_48": LBL_COL, 
                                    # "geq0_48": LBL_COL, 
                                    "point_est_set": "Dataset"})
        if prediction_col is None:
            df_b = df_b.rename(columns={"raw_point_est": PD_COL})
        else:
            df_b = df_b.rename(columns={prediction_col: PD_COL})
            
        df_b = df_b.sort_values([PID_COL, DT_COL])
        df_b = df_b.reset_index(drop=True)
        
        for pid in df_b[PID_COL].unique():
            df_p = df_b[df_b[PID_COL]==pid]
            if df_p[DT_COL].duplicated(keep=False).sum() > 0:
                df_b = df_b.drop(df_p.index[df_p[DT_COL].duplicated(keep="last")])
                
        df.append(df_b[[PID_COL, DT_COL, PD_COL, "Dataset"]])
        gc.collect()

        sys.stdout.write("Reading Thomas' prediction results: ")
        sys.stdout.write("Batch %02d \r"%(i+1))
        sys.stdout.flush()
        
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    
    print("\nFinished reading Thomas' results.")
    return df


def read_clinical_baseline(datapath,
                           lst_batches,
                           spo2=None,
                           fio2=None):
    """
    Reading clinical baseline prediction
    """
    
    df = []
    for i, b in enumerate(lst_batches):
        batch_path = os.path.join(datapath, b)
        df_b = pd.read_hdf(batch_path, chunksize=10**5, mode="r")
        for tmp in df_b:
            df.append(tmp[[PID_COL, ABS_DT_COL, REL_DT_COL, SPO2_ID, FIO2_ID]])
            gc.collect()
            
        sys.stdout.write("Reading clinical baseline results: ")
        sys.stdout.write("Batch %02d \r"%(i+1))
        sys.stdout.flush()
        
    df = pd.concat(df)
    df.loc[:,PD_COL] = (df[SPO2_ID]<spo2)|(df[FIO2_ID]>fio2)
    df = df.rename(columns={ABS_DT_COL: DT_COL})
    df = df.sort_values([PID_COL, DT_COL])
    df = df.reset_index(drop=True)
    
    print("\nFinished reading clinical predictions.")
    return df[[PID_COL, DT_COL, PD_COL]]

def read_clinical_baseline2(datapath, lst_batches, spo2=None, fio2=None):
    """
    Reading clinical baseline prediction
    """            
    usecols = ["PatientID", "AbsDatetime", "RelDatetime", "vm20", "vm58"]
    df = []
    for i, b in enumerate(lst_batches):
        batch_path = os.path.join(datapath, b)
        df_b = pd.read_hdf(batch_path, chunksize=10**5, mode="r")
        for tmp in df_b:
            df.append(tmp[usecols])
            gc.collect()
        sys.stdout.write("Reading clinical baseline results: ")
        sys.stdout.write("Batch %02d \r"%(i+1))
        sys.stdout.flush()
    df = pd.concat(df)

    # df.loc[:,"Prediction"] = (df[FIO2_ID]/df[SPO2_ID]).values
    df.loc[:,"Prediction"] = (df[SPO2_ID]/df[FIO2_ID]).values
    min_ratio = 0.21
    max_ratio = 10.0
    df.loc[:,"Prediction"] = 1 - (df.Prediction - min_ratio)/(max_ratio - min_ratio)    
    # df.loc[:,"Prediction"] = (df.Prediction - min_ratio)/(max_ratio - min_ratio)    
    df = df.rename(columns={"AbsDatetime": "Datetime"})
    df = df.sort_values(["PatientID", "Datetime"]).reset_index(drop=True)
    print("Finished reading clinical predictions.")
    return df[["PatientID", "Datetime", "Prediction"]]


def get_intersection(dict_df):
    """
    Compute the intersection on time grid for all models considered
    """
    
    for k, df in dict_df.items():
        df.loc[:,DT_COL] = df[DT_COL].dt.floor("5T")
        if 'bw' in k: # why do I shift it???
            df.loc[:,DT_COL] = df[DT_COL] + np.timedelta64(5,"m")
            
    lst_pids = [df[df[PD_COL].notnull()][PID_COL].unique() for df in lst_df]
    lst_pids = reduce(np.intersect1d, tuple(lst_pids))

    df_i = []
    for df in lst_df:
        tmp = df[df[PID_COL].isin(lst_pids)]
        tmp = tmp.sort_values([PID_COL, DT_COL])
        tmp = tmp.set_index([PID_COL, DT_COL])
        df_i.append(tmp[[PD_COL]])

    df_i = pd.concat(df_i, join="inner", axis=1)
    df_i = df_i[df_i.isnull().values.sum(axis=1)==0]

    # get the intersection value of the files
    dict_df_i = {k: df_i.iloc[:,i].reset_index() for i, k in enumerate(dict_df.keys())}
    gc.collect()
    return dict_df_i

def get_only_urine_ep(df_ep):
    
    gap2merge_m = 1500 # manimal gap length allowed in minutes (25 hours)
    time_step = np.timedelta64(5,"m")
    df_ep.loc[:,"geq1_urine"] = df_ep["1.u"].copy() # label based on only urine
    
    for pid in df_ep[PID_COL].unique():
        tmp = df_ep[df_ep[PID_COL]==pid]
        
        # if there is no positive label, skip (i.e. we don't change the endpoint information)
        if tmp.geq1.sum()==0:
            continue
        
        if tmp["1.u"].sum()==0:
            continue

        tdiff = tmp[tmp['geq1_urine']==1][DT_COL].diff() # the time difference between timepoints with positive endpoint label based on urine
        if np.sum((tdiff>time_step)&(tdiff<np.timedelta64(gap2merge_m,"m"))) > 0:
            dt_all_gap_end = tdiff[(tdiff>time_step)].index # if the gap length is larger than 5 minutes
            dt_all_gap_start = np.array([tmp[tmp[DT_COL]==(tmp.loc[l][DT_COL]-tdiff.loc[l])].index[0] for l in dt_all_gap_end])
            
            assert(len(dt_all_gap_start)==len(dt_all_gap_end))
            
            dt_short_gap_end = tdiff[(tdiff>time_step)&(tdiff<np.timedelta64(gap2merge_m,"m"))].index
            idx_short_gap_end = np.where(np.isin(dt_all_gap_end, dt_short_gap_end))[0]
            
            for i in idx_short_gap_end:
                df_ep.loc[dt_all_gap_start[i]:dt_all_gap_end[i],"geq1_urine"] = 1
                
    return df_ep

def get_only_creatinine_ep(df_ep):
    
    gap2merge_m = 1500
    time_step = np.timedelta64(5,"m")    
    df_ep.loc[:,"geq1_creatinine"] = (df_ep["1.b"].astype(bool)|df_ep["1.i"].astype(bool))
    for pid in df_ep[PID_COL].unique():
        tmp = df_ep[df_ep[PID_COL]==pid]

        # if there is no positive label, skip (i.e. we don't change the endpoint information)        
        if tmp.geq1.sum()==0:
            continue
        
        # if some endpoint is caused by creatinine (either increase in value or compared to the baseline)???
        if tmp["1.b"].sum()==0 and tmp["1.i"].sum()==0:
            continue
        
        tdiff = tmp[tmp['geq1_creatinine']==1][ABS_DT_COL].diff()
        if np.sum((tdiff>time_step)&(tdiff<np.timedelta64(gap2merge_m,"m"))) > 0:
            loc_all_gap_end = tdiff[(tdiff>time_step)].index
            loc_all_gap_start = np.array([tmp[tmp[ABS_DT_COL]==(tmp.loc[l][ABS_DT_COL]-tdiff.loc[l])].index[0] for l in loc_all_gap_end])
            
            loc_short_gap_end = tdiff[(tdiff>time_step)&(tdiff<np.timedelta64(gap2merge_m,"m"))].index
            idx_short_gap_end = np.where(np.isin(loc_all_gap_end, loc_short_gap_end))[0]
            for i in idx_short_gap_end:
                df_ep.loc[loc_all_gap_start[i]:loc_all_gap_end[i],"geq1_creatinine"] = 1
                
    return df_ep

def read_ep_renal(datapath,
                  lst_batches,
                  endpoint_name,
                  no_urine=False,
                  only_urine=False):
    """
    Read renal endpoints
    """
    df = []
    for i, b in enumerate(lst_batches):
        df_b = pd.read_hdf(os.path.join(datapath, b)).reset_index(drop=True)
            
        if no_urine:
            df_b = df_b.drop(columns=['geq1'])
            df_b = df_b.rename(columns={"geq1_creatinine": "geq1"})

        if endpoint_name == "012-3":
            df_b.loc[:,"Stable"] = (df_b.geq3 == False) & (df_b.geq2 == True)
            df_b.loc[:,"InEvent"] = df_b.geq3 == True
            df_b.loc[:,"Unknown"] = df_b.geq3.isnull() | (df_b.geq2 == False)
            
        elif endpoint_name == "01-23":
            df_b.loc[:,"Stable"] = (df_b.geq2 == False) & (df_b.geq1 == True)
            df_b.loc[:,"InEvent"] = df_b.geq2 == True
            df_b.loc[:,"Unknown"] = df_b.geq2.isnull() | (df_b.geq1 == False)
            
        elif endpoint_name == "0-123":
            if only_urine:
                df_b = get_only_urine_ep(df_b)
                df_b.loc[:,"Stable"] = (df_b.geq1_urine == False)
                df_b.loc[:,"InEvent"] = (df_b.geq1_urine == True)
                df_b.loc[:,"Unknown"] = df_b.geq1_urine.isnull()
                
            elif no_urine:
                df_b = get_only_creatinine_ep(df_b)
                df_b.loc[:,"Stable"] = (df_b.geq1_creatinine == False)
                df_b.loc[:,"InEvent"] = (df_b.geq1_creatinine == True)
                df_b.loc[:,"Unknown"] = df_b.geq1_creatinine.isnull()
                
            else:
                df_b.loc[:,"Stable"] = (df_b.geq1 == False)
                df_b.loc[:,"InEvent"] = (df_b.geq1 == True)
                df_b.loc[:,"Unknown"] = df_b.geq1.isnull()
                
        df.append(df_b)
        sys.stdout.write("Reading KDIGO endpoint: Batch %02d \r"%(i+1))

    df = pd.concat(df)
    df = df.rename(columns={ABS_DT_COL: DT_COL})
    
    df = df.sort_values([PID_COL, DT_COL])
    df = df.reset_index(drop=True)

    print("\nFinished reading endpoints.")
    return df[[PID_COL, DT_COL, "Stable", "InEvent", "Unknown"]]


def read_ep_resp(datapath,
                 lst_batches,
                 endpoint_name, 
                 stable_status=None,
                 unstable_status=None):
    """
    Read respiratory endpoints
    """
    df = []
    for i, b in enumerate(lst_batches):
        df_b = pd.read_hdf(os.path.join(datapath, b))
        df_b = df_b.reset_index(drop=True)
        
        if endpoint_name=="resp_failure":
            str_stable = ["event_%s"%x for x in stable_status]
            str_fail = ["event_%s"%x for x in unstable_status]
            df_b.loc[:,"Stable"] = df_b.endpoint_status.isin(str_stable).values
            df_b.loc[:,"InEvent"] = df_b.endpoint_status.isin(str_fail).values
            
            status_of_interest = np.concatenate((stable_status, unstable_status))
            if len(np.unique(status_of_interest)) == 4:
                df_b.loc[:,"Unknown"] = (df_b.endpoint_status=="UNKNOWN").values
            else:
                unknown_status = set(["0","1","2","3"]) - set(status_of_interest)
                unknown_status = list(unknown_status) + ["UNKNOWN"]
                df_b.loc[:,"Unknown"] = df_b.endpoint_status.isin(unknown_status).values
                
        elif "extu" in endpoint_name and endpoint_name not in ["extu_failure", "extu_success"]:
            df_b.loc[:,"Stable"] = (df_b.readiness_ext==0).values
            df_b.loc[:,"InEvent"] = (df_b.readiness_ext==1).values
            df_b.loc[:,"Unknown"] = df_b.readiness_ext.isnull()
            
        elif "vent" in endpoint_name:
            df_b.loc[:,"Stable"] = (df_b.vent_period==0).values
            df_b.loc[:,"InEvent"] = (df_b.vent_period==1).values
            df_b.loc[:,"Unknown"] = df_b.vent_period.isnull()
            
        elif endpoint_name in ["extu_failure", "extu_success"]:
            for pid in df_b[df_b.ext_failure.notnull()][PID_COL].unique():
                df_p = df_b[df_b[PID_COL]==pid]
                for idx in df_p[df_p.ext_failure.notnull()].index:

                    # create a short event of 30 minutes after the extubation 
                    # success or failure event
                    dt_extu =  df_p.loc[idx,ABS_DT_COL]
                    dt_extu_end = dt_extu + np.timedelta64(30,"m")
                    idx_tmp = df_p.index[( (df_p[ABS_DT_COL]>dt_extu)
                                          &(df_p[ABS_DT_COL]<dt_extu_end))]
                    df_b.loc[idx_tmp,"ext_failure"] = df_b.loc[idx,"ext_failure"]

                    # the prediction score only happen at the time point of 
                    # extubation
                    if "ExtubationFailure" in out_path:
                        df_b.loc[idx,"ext_failure"] = 0
                    else:
                        df_b.loc[idx,"ext_failure"] = 1

            if endpoint_name=="extu_failure":
                df_b.loc[:,"Stable"] = (df_b.ext_failure==0).values
                df_b.loc[:,"InEvent"] = (df_b.ext_failure==1).values
            else:
                df_b.loc[:,"Stable"] = (df_b.ext_failure==1).values
                df_b.loc[:,"InEvent"] = (df_b.ext_failure==0).values                    
            df_b.loc[:,"Unknown"] = df_b.ext_failure.isnull()
        else:
            raise Exception("endpoint %s is not defined!"%endpoint_name)
            exit(0)
            
        df.append(df_b)
        sys.stdout.write("Reading respiratory endpoint: Batch %02d \r"%(i+1))

    print("\nFinished reading endpoints.")
    df = pd.concat(df)
    df = df.rename(columns={ABS_DT_COL: DT_COL})
    df = df.sort_values([PID_COL, DT_COL])
    df = df.reset_index(drop=True)
    return df[[PID_COL, DT_COL, "Stable", "InEvent", "Unknown"]]

def align_time(df1, df2):
    df1.loc[:,DT_COL] = df1[DT_COL].dt.floor("5T")
    df2.loc[:,DT_COL] = df2[DT_COL].dt.floor("5T")
    df1 = df1.set_index(DT_COL)
    df2 = df2.set_index(DT_COL)
    
    df = df1.merge(df2.drop(columns=[PID_COL]),
                   how="outer", 
                   left_index=True,
                   right_index=True)
    
    df.loc[:,REL_DT_COL] = (df.index-df.index[0])/np.timedelta64(1,"s")
    assert(df[REL_DT_COL].isnull().sum()==0)
    df = df.sort_values(REL_DT_COL)
    df = df.reset_index()
    df = df.rename(columns={"index": DT_COL})
    df = df.set_index(REL_DT_COL)
    return df

def get_threshold(datapath,
                  configs, 
                  rec=0.9, 
                  prec=None, 
                  is_first_onset=False, 
                  is_random=False):

    df = utils_plot._get_df(datapath, 
                            configs, 
                            RANDOM=is_random, 
                            onset_type="first" if is_first_onset else None)
    
    if rec is not None and prec is None:
        tmp = df.iloc[np.argmin(np.abs(df.rec.values-rec))]
    elif rec is None and prec is not None:
        tmp = df.iloc[np.argmin(np.abs(df.prec.values-prec))]
    return tmp.tau, tmp.rec, tmp.prec

def get_event_info(df,
                   prediction_score_interval,
                   beginning_hours2ignore=None):
    df.loc[:,"Onset"] = False
    df.loc[:,"EventEnd"] = False
    df.loc[:,"Merged"] = False

    if df["InEvent"].sum() > 0:
        dt_event = df.index[df["InEvent"]]
        beg_event = np.concatenate((dt_event[[0]],
                                    dt_event[np.where(np.diff(dt_event)>prediction_score_interval)[0]+1]))
        end_event = np.concatenate((dt_event[np.where(np.diff(dt_event)>prediction_score_interval)[0]],
                                    dt_event[[-1]]))

        assert(len(beg_event)==len(end_event))
        if beginning_hours2ignore is not None:
            end_event = end_event[beg_event>beginning_hours2ignore*3600]
            beg_event = beg_event[beg_event>beginning_hours2ignore*3600]

        df.loc[beg_event,"Onset"] = True
        df.loc[end_event,"EventEnd"] = True

        # merge events with gap shorter than t_mingap
        if t_mingap > 0:
            gap_witdth = (beg_event[1:] - end_event[:-1])
            idx_gap2merge = np.where(gap_width <= t_mingap * 60)[0]
            for igap in idx_gap2merge:
                df.loc[end_event[igap]:beg_event[1+igap]-300,"Merged"] = True
                df.loc[beg_event[1+igap],"Onset"] = False
                df.loc[end_event[igap],"EventEnd"] = False
            beg_event = np.delete(beg_event, idx_gap2merge+1)
            end_event = np.delete(end_event, idx_gap2merge)

        lst_event_end = np.concatenate(([beginning_hours2ignore*3600-t_reset_sec], end_event))
        for kk, dt_onset in enumerate(beg_event):
            win_pre_event = df[np.logical_and(df.index>=max(dt_onset-max_sec,lst_event_end[kk]+t_reset_sec),
                                              df.index<dt_onset-min_sec)] 
            if win_pre_event[PD_COL].notnull().sum()==0:
                # if len(win_pre_event) in [0, win_pre_event[PD_COL].isnull().sum()]:
                # if prior to the event, the status are unknown and there is no prediction score,                                                                                                           
                # this event is listed unpredictable at all, hence delete it from the Onset list                                                                                                            
                # print("reset.")
                df.loc[dt_onset, "Onset"] = False
                df.loc[end_event[kk], "EventEnd"] = False
    try:
        assert(df["Onset"].sum()==df["EventEnd"].sum())
    except:
        ipdb.set_trace()

def get_alarm_info(df, tau, prediction_score_interval):
    # Compute true alarms and false alarms                                                                                 
    df.loc[:,"GeqTau"] = (df[PD_COL] >= tau)
    df.loc[:,"Alarm"] = False
    df.loc[:,"TrueAlarm"] = False

    if df["GeqTau"].sum() > 0:
        dt_geq_tau = df.index[df["GeqTau"]]
        if t_silence == 5 and t_reset == 0 or df[PD_COL].notnull().sum() < 2:
            df.loc[dt_geq_tau,"Alarm"] = True
            df.loc[df.index[df.Merged],"Alarm"] = False

        elif df["Onset"].sum()==0:
            dt = 0
            while dt <= dt_geq_tau[-1]:
                dt = dt_geq_tau[dt_geq_tau>=dt][0]
                reset = False
                while not reset:
                    if dt in df.index and df.loc[dt,"GeqTau"]:
                        df.loc[dt,"Alarm"] = True
                        dt += ts
                    else:
                        reset = True
            df.loc[df.index[df.Merged],"Alarm"] = False
        else:
            dt = 0
            while dt <= dt_geq_tau[-1]:
                dt = dt_geq_tau[dt_geq_tau>=dt][0]
                reset = False
                while not reset:
                    if dt in df.index and df.loc[dt,"GeqTau"]:
                        if dt < df.index[df["EventEnd"]].values[0]:
                        # before the first event
                            df.loc[dt,"Alarm"] = True
                            dt = min( dt+ts , df.index[df["EventEnd"]].values[0]+t_reset_sec)
                        else:
                            # after the first event
                            dist2event = dt - df.index[df["EventEnd"]].values
                            last_end = df.index[df["EventEnd"]].values[np.where(dist2event>0)[0][-1]]
                            
                            if np.where(dist2event>0)[0][-1] == df["EventEnd"].sum()-1:
                            # after the last event
                                next_end = df.index[-1]
                            else:
                            # before the last event
                                next_end = df.index[df["EventEnd"]].values[np.where(dist2event>0)[0][-1]+1]

                            if dt < last_end+t_reset_sec:
                                # if the current dt falls in the reset time period
                                dt = last_end+t_reset_sec
                            else:
                                # otherwise
                                df.loc[dt,"Alarm"] = True
                                dt = min( dt+ts , next_end+t_reset_sec)
                    else:
                        reset = True
        df.loc[df.index[df.Merged],"Alarm"] = False
        for dt_alarm in df.index[df["Alarm"]]:
            win_post_alarm = df[np.logical_and(df.index<=dt_alarm+max_sec,df.index>dt_alarm+min_sec)]
            if len(win_post_alarm) in [0, win_post_alarm.Unknown.sum()]:
                # if there is no more time after the alarm or the status are completely unknown, we                        
                # consider the alarm neither true or false, hence disable it.                                              
                df.loc[dt_alarm,"Alarm"] = False
            else:
                df.loc[dt_alarm,"TrueAlarm"] = win_post_alarm["InEvent"].sum()>0

    # Compute captured events and missed events                                                                            
    df.loc[:,"CatchedOnset"] = False
    if df["Onset"].sum() > 0:
        lst_event_end = np.concatenate(([0], df.index[df["EventEnd"]]))
        for kk, dt_onset in enumerate(df.index[df["Onset"]]):
            win_pre_event = df[np.logical_and(df.index>=max(dt_onset-max_sec,lst_event_end[kk]),
                                              df.index<dt_onset-min_sec)]
            df.loc[dt_onset,"CatchedOnset"] = win_pre_event.Alarm.sum() > 0
    return df


def plot_ts(df_p):
    """
    plot the time series of a patient
    """
    tvals = df_p.index/3600
    plt.fill_between(tvals, df_p.Stable.astype(int)-0, step="pre", color="C2", alpha=0.2, zorder=1, label="Stable")
    plt.fill_between(tvals, df_p.InEvent.astype(int)-0, step="pre", color="C3", alpha=0.2, zorder=2, label="Renal Failure")
    plt.plot(tvals, df_p.Prediction, color="k", marker='.', markersize=2**2, zorder=3, label="Prediction score")
    plt.scatter(tvals[df_p.Onset]-300/3600, [1]*df_p.Onset.sum(), marker=11, color="C3", zorder=4, label="Onset")
#     plt.scatter(tvals[df_p.EventEnd], [1]*df_p.EventEnd.sum(), marker=11, color="C2", zorder=5, label="_nolegend_")
    plt.axhline(y=tau, linestyle="--", color='gray', zorder=6, label="Threshold")
    plt.scatter(tvals[df_p.Alarm], df_p.Prediction[df_p.Alarm]+0.05, marker='d', color="C2", s=5**2, zorder=7, label="Alarm")
    plt.scatter(tvals[df_p.TrueAlarm], df_p.Prediction[df_p.TrueAlarm]+0.05, marker='d', color="C3", zorder=8, label="True Alarm")
    plt.scatter(tvals[df_p.CatchedOnset]-300/3600, [0.85]*df_p.CatchedOnset.sum(), marker='*', s=16**2, color="C3", zorder=9, label="Catched Onset")
    plt.xlim(tvals.min()-0.05, tvals.max()+0.05)
    if tvals.max() > 48:
        tickvals = np.arange(0,tvals.max(),24)
    else:
        tickvals = np.arange(0,tvals.max(),4)
    plt.xticks(tickvals, ["%d"%x for x in tickvals])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time since admission (h)')
    plt.ylabel('Prediction score')

def sys_print(line):
    sys.stdout.write(line+"\n")
    sys.stdout.flush()
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    ### essential parameters
    parser.add_argument("-ep_path")
    parser.add_argument("-pd_path")
    parser.add_argument("-out_path")
    parser.add_argument("-split_file", default="")

    parser.add_argument("--t_delta", type=int, default=0)
    parser.add_argument("--t_window", type=int, default=480)
    parser.add_argument("--t_silence", type=int, default=30)
    parser.add_argument("--t_mingap", type=int, default=0)
    parser.add_argument("--t_reset", type=int, default=0)

    parser.add_argument("--RANDOM", action="store_true")
    parser.add_argument("--FIRST_ONSET", action="store_true")
    parser.add_argument("--random_model", default=None)
    parser.add_argument("--random_seed", type=int, default=2021)

    ### types of enpoints
    parser.add_argument("--stable_status", nargs="+", default=["0", "1"])
    parser.add_argument("--unstable_status", nargs="+", default=["2", "3"])
    parser.add_argument("--ep_problem", default="renal", choices=["respiratory", "renal"])
    parser.add_argument("--ep_type", default="resp_failure")

    ### baseline parameters
    parser.add_argument("--spo2", type=float, default=90)
    parser.add_argument("--fio2", type=float, default=60)
    parser.add_argument("--THRESHOLD_BASELINE", action="store_true")
    
    ### intersection parameters
    parser.add_argument("--tg_path", default=None)
    parser.add_argument("--mh_path", default=None)
    parser.add_argument("--bw_rf_path", default=None)
    parser.add_argument("--bw_hmm_path", default=None)
    parser.add_argument("--pred_col", default="WorseStateFromZero0To48Hours")
    parser.add_argument("--tg_ep_type", default="1_1")
    parser.add_argument("--set_type", default="test")
    parser.add_argument("--idx_method", type=int, nargs="+", default=[0,1])
    parser.add_argument("--lst_w", type=float, nargs="+", default=[0.333,0.333,0.334])
    parser.add_argument("--INTERSECT", action="store_true")

    ### parallelization parameters
    parser.add_argument("--idx_batch", type=int, default=None)
    parser.add_argument("--idx_threshold", type=int, default=None)

    parser.add_argument("--kdigo_intervent_type", default=None, choices=["furosemide", "fluid", "no_furosemide", "no_fluid", "wo_invasive_bp", "w_invasive_bp"])

    ### parameters for getting alarm information with fixed precision/recall
    parser.add_argument("--fixed_recall", type=float, default=None)
    parser.add_argument("--fixed_precision", type=float, default=None)
    parser.add_argument("--alarm_path", default=None)
    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument("--ext_tsplit", default='exploration_1')
    parser.add_argument("--correct_false_alarm", action="store_true")
    parser.add_argument("--correct_false_alarm_win", default=4, type=int)
    parser.add_argument("--correct_false_alarm_mode", default="FA2TA", choices=["DelFA", "FA2TA"])

    parser.add_argument("--select_cohort", default=None)
    parser.add_argument("--bootstrap_seed", default=None, type=int)

    parser.add_argument("--no_urine", action="store_true")
    parser.add_argument("--only_urine", action="store_true")

    args = parser.parse_args()
    configs = vars(args)
    for key, val in configs.items():
        exec("%s=%s"%(key, "'%s'"%val if type(val)==str else val))
    t_reset_sec = t_reset*60
    lst_w = np.array(lst_w) / 10

    ts = t_silence * 60
    # max_sec = (t_delta+t_window) * 60
    max_sec = t_window * 60
    min_sec = t_delta * 60
    t_unit = np.timedelta64(1,"s")
    
    tsplit = ext_tsplit
    
    if ep_problem=="renal" and INTERSECT:
        lst_files = np.array([f for f in os.listdir(mh_path) if ".h5" in f])
        lst_files = lst_files[np.argsort([int(f.split('_')[-1].split('.')[0]) for f in lst_files])]
    elif ep_problem=="renal":
        lst_files = np.array([f for f in os.listdir(mh_path) if ".h5" in f])
        lst_files = lst_files[np.argsort([int(f.split('_')[-1].split('.')[0]) for f in lst_files])]
    else:
        lst_files = np.array([f for f in os.listdir(pd_path) if ".h5" in f and "batch" in f])
        lst_files = lst_files[np.argsort([int(f.split('_')[-1].split('.')[0]) for f in lst_files])]

    if idx_batch is not None:
        lst_files = lst_files[idx_batch:idx_batch+1]

    if DEBUG:
        lst_files =  lst_files[-1:]


    ### select only valiation/test set
    if ep_problem == "renal":
        with open(split_file, "rb")  as tmp:
            pid_mapping = pickle.load(tmp)
        lst_pids = pid_mapping[tsplit][set_type]
                
        lst_files_selected = []
        if "mimic" in pd_path and "internal" not in pd_path: 
            tsplit = 'exploration_1'
        else:
            lst_tsplit = ["point_est"] + ["temporal_%d"%i for i in np.arange(1,6)]
            tsplit = [s for s in lst_tsplit if s in mh_path][0]
        for f in lst_files:
            try:
                with h5py.File(os.path.join(mh_path, f), mode="r") as tmp:
                    pids = [k for k in tmp.keys()]
            except:
                continue
            if len(set([int(x[1:]) for x in pids]) & set(pid_mapping[tsplit][set_type]))==0:
                continue
            lst_files_selected.append(f)
        lst_files = lst_files_selected
        
    elif ep_problem=="respiratory":
        
        with open(split_file, "rb")  as tmp:
            pid_mapping = pickle.load(tmp)
        lst_pids = pid_mapping[tsplit][set_type]
        
        lst_files_selected = []
        lst_tsplit = ["point_est"] + ["temporal_%d"%i for i in np.arange(1,6)]
        for f in lst_files:
            try:
                with h5py.File(os.path.join(pd_path, f), mode="r") as tmp:
                    pids = [k for k in tmp.keys()]
                if len(pids) == 1:
                    pids = pd.read_hdf(os.path.join(pd_path, f), columns=['PatientID']).PatientID.unique()
                else:
                    pids = [int(x[1:]) for x in pids]
            except:
                continue
            print(pids[:10], lst_pids[:10])
            if len(set(pids) & set(lst_pids))==0:
                continue
            lst_files_selected.append(f)
        print(lst_files)
        print(lst_files_selected)
        lst_files = lst_files_selected

        if THRESHOLD_BASELINE:
            # df_pd = read_clinical_baseline(pd_path, lst_files, spo2=spo2, fio2=fio2)
            
            df_pd = read_clinical_baseline2(pd_path, lst_files, spo2=spo2, fio2=fio2)
        else:
            df_pd = read_mh_pd(pd_path, lst_files)

        df_ep = read_ep_resp(ep_path, lst_files, ep_type, stable_status=stable_status, unstable_status=unstable_status)
    else:
        if INTERSECT:
            
            dict_df = dict()
            if tg_path is not None:
                lst_tg_files = np.array([f for f in os.listdir(tg_path) if any([ff in f for ff in lst_files]) and tg_ep_type in f])
                if len(lst_tg_files) == 0:
                    lst_tg_files = np.array([f for f in os.listdir(tg_path) if any([ff in f for ff in lst_files])])
                lst_tg_files = lst_tg_files[np.argsort([int(f.split('_')[-1].split('.')[0]) for f in lst_tg_files])]
                df_tg = read_tg_pd(tg_path, lst_tg_files, prediction_col=random_model)
                df_tg = df_tg.drop(df_tg.index[~df_tg[PID_COL].isin(lst_pids)])
                df_tg.loc[df_tg.index[df_tg[PD_COL]>1e6],"Prediction"] = 1
                dict_df.update({'tg': df_tg})

            if bw_hmm_path is not None:
                df_bw_hmm = read_bw_pd(bw_hmm_path, pred_col)
                df_bw_hmm = df_bw_hmm.drop(df_bw_hmm.index[~df_bw_hmm[PID_COL].isin(lst_pids)])
                dict_df.update({'bw_hmm': df_bw_hmm})

            if bw_rf_path is not None:
                df_bw_rf = read_bw_pd(bw_rf_path, pred_col)
                df_bw_rf = df_bw_rf.drop(df_bw_rf.index[~df_bw_rf[PID_COL].isin(lst_pids)])
                dict_df.update({'bw_rf': df_bw_rf})

            if mh_path is not None:
                if "calib" in pd_path:
                    df_mh = read_mh_pd(mh_path, lst_files, calibrated=True)
                else:
                    df_mh = read_mh_pd(mh_path, lst_files)
                    
                df_mh = df_mh.drop(df_mh.index[~df_mh[PID_COL].isin(lst_pids)])
                dict_df.update({'mh': df_mh})
            
            dict_df = get_intersection(dict_df)
            
            if "mh" in out_path.split("/")[-1] and "tg" not in out_path.split("/")[-1] and "bw" not in out_path.split("/")[-1]:
                df_pd = dict_df['mh']
                print("mh")
                
            elif "tg" in out_path:
                df_pd = dict_df['tg']
                print("tg")
                
            elif "bw" in out_path:
                df_pd = dict_df['bw_hmm']
                print("bw")
                
            elif "ensemble_mean" in out_path:
                df_pd = lst_df[0].copy()
                df_pd.loc[:,"Prediction"] = np.nanmean([x[PD_COL].values for x in lst_df[:-1]], axis=0)
                
            elif "ensemble_weighted_mean" in out_path:
                df_pd = lst_df[0].copy()
                df_pd.loc[:,"Prediction"] = np.nansum([x[PD_COL].values * lst_w[i] for i, x in enumerate(lst_df[:-1])], axis=0)
                
            elif "ensemble_max" in out_path:
                df_pd = lst_df[0].copy()
                df_pd.loc[:,"Prediction"] = np.nanmax([x[PD_COL].values for x in lst_df[:-1]], axis=0)
                
            elif "ensemble" in out_path:
                df_pd = lst_df[idx_method[0]].copy()
                df_pd.loc[:,"Prediction"] = np.nanmean([x[PD_COL].values for n, x in enumerate(lst_df) if n in idx_method], axis=0)
                
        else:
            if "thomas" in pd_path.lower():
                df_pd = read_tg_pd(pd_path, lst_files, prediction_col=random_model)
                
            elif "bowen" in pd_path.lower():
                df_pd = read_bw_pd(pd_path, pred_col)
                
            else:
                df_pd = read_mh_pd(pd_path, lst_files)

        df_ep = read_ep_renal(ep_path, lst_files, ep_type, no_urine=no_urine)
        
    sys_print("Finish reading endpoint files")
    # if ep_problem == "renal" or "umcdb" in pd_path:
    df_ep = df_ep.drop(df_ep.index[~df_ep[PID_COL].isin(lst_pids)])
    df_pd = df_pd.drop(df_pd.index[~df_pd[PID_COL].isin(lst_pids)])


    if kdigo_intervent_type is not None:
        if kdigo_intervent_type in ["furosemide", "no_furosemide"]:
            pids_with_intervention = pd.read_csv("patient_list_with_furosemide.csv")[PID_COL].values
        elif kdigo_intervent_type in ["fluid", "no_fluid"]:
            pids_with_intervention = pd.read_csv("patient_list_with_fluid.csv")[PID_COL].values
        elif kdigo_intervent_type in ["wo_invasive_bp", "w_invasive_bp"]:
            pids_with_intervention = pd.read_csv("pids_without_invasive_bp.csv")[PID_COL].values
            
        if "no" in kdigo_intervent_type or kdigo_intervent_type == "w_invasive_bp":
            df_ep = df_ep.drop(df_ep.index[df_ep[PID_COL].isin(pids_with_intervention)])
            df_pd = df_pd.drop(df_pd.index[df_pd[PID_COL].isin(pids_with_intervention)])      
        else:
            df_ep = df_ep.drop(df_ep.index[~df_ep[PID_COL].isin(pids_with_intervention)])
            df_pd = df_pd.drop(df_pd.index[~df_pd[PID_COL].isin(pids_with_intervention)])      
    
    if "kdigo" in out_path:
        prediction_score_interval = 3600
    else:
        prediction_score_interval = 300
        
    lst_pids = np.sort(df_pd[PID_COL].unique())
    if DEBUG:
        lst_pids = lst_pids[:20]
    if bootstrap_seed is not None:
        lst_pids = np.random.permutation(lst_pids)[:int(len(lst_pids)*0.5)]
        
    np.random.seed(random_seed)
    df_with_event =  []
    max_score = 0

    print('Number of patients in split %s :%d'%(tsplit, len(lst_pids)))
    for i, pid in enumerate(lst_pids):
        if (df_ep[PID_COL]==pid).sum()==0:
            continue
        
        df = align_time(df_ep[df_ep[PID_COL]==pid].copy(), df_pd[df_pd[PID_COL]==pid].copy())
        df.loc[df.index[~df.Stable.astype(bool)], PD_COL] = np.nan
            
        if df[PD_COL].notnull().sum()==0:
            print("%d, PatientID %d does not have prediction score"%(i, pid))
            continue
            
        if RANDOM:
            df.loc[df.index[df[PD_COL].notnull()], PD_COL] = np.random.rand(df[PD_COL].notnull().sum())
            
        df = df.drop(df.index[df.InEvent.isnull()])

        get_event_info(df, prediction_score_interval, beginning_hours2ignore=2)

        for i, dt in enumerate(df.index[df.Onset]):
            if i==0:
                max_score = max(df[(df.index>=0)&(df.index>=dt-max_sec)&(df.index<dt-min_sec)][PD_COL].max(), max_score)
            else:
                max_score = max(df[(df.index>=df.index[df.EventEnd][i-1])&(df.index>=dt-max_sec)&(df.index<dt-min_sec)][PD_COL].max(), max_score)
                
        df_with_event.append(df)
        
    df_with_event = pd.concat(df_with_event)
    df_with_event = df_with_event.reset_index()

    if THRESHOLD_BASELINE:
        # thresholds = [0,0.5,1.1]
        # if fixed_recall is not None and fixed_precision is None:
        #     thresholds = [0.5]

        pa_with_events = df_with_event[df_with_event.InEvent==True][PID_COL].unique()
        pred_before_onset = []
        for pid in pa_with_events:
            df_tmp  = df_with_event[df_with_event[PID_COL]==pid]
            for dt in df_tmp[REL_DT_COL][df_tmp.Onset]:
                pred_tmp = df_tmp[(df_tmp[REL_DT_COL]>=dt-max_sec)&(df_tmp[REL_DT_COL]<dt-min_sec)][PD_COL]
                pred_tmp = pred_tmp[pred_tmp.notnull()]
                pred_before_onset.extend(pred_tmp)
                
        thresholds = [np.nanpercentile(pred_before_onset, i) for i in range(100)]        
        thresholds = np.concatenate(([0], thresholds, [max_score, df_with_event[PD_COL].max()]))
        thresholds = np.interp(np.arange(0, len(thresholds)+0.25, 0.25), range(len(thresholds)), thresholds)        
        thresholds = np.concatenate((thresholds[:315:3], thresholds[315:]))
        thresholds = thresholds[::2]
        if fixed_recall is not None and fixed_precision is None:
            tau, rec, prec = get_threshold(out_path, configs, rec=fixed_recall)
            thresholds = [tau]
            
        elif fixed_recall is None and fixed_precision is not None:
            tau, rec, prec = get_threshold(out_path, configs, rec=None, prec=fixed_precision, is_first_onset=FIRST_ONSET)
            thresholds = [tau]
        print('select threshold:',thresholds)
        
    elif fixed_recall is not None and fixed_precision is None:
        tau, rec, prec = get_threshold(out_path, configs, rec=fixed_recall)
        thresholds = [tau]
        
    elif fixed_recall is None and fixed_precision is not None:
        tau, rec, prec = get_threshold(out_path, configs, rec=None, prec=fixed_precision, is_first_onset=FIRST_ONSET)
        thresholds = [tau]
        
    else:
        pa_with_events = df_with_event[df_with_event.InEvent==True][PID_COL].unique()
        pred_before_onset = []
        for pid in pa_with_events:
            df_tmp  = df_with_event[df_with_event[PID_COL]==pid]
            for dt in df_tmp[REL_DT_COL][df_tmp.Onset]:
                pred_tmp = df_tmp[(df_tmp[REL_DT_COL]>=dt-max_sec)&(df_tmp[REL_DT_COL]<dt-min_sec)][PD_COL]
                pred_tmp = pred_tmp[pred_tmp.notnull()]
                pred_before_onset.extend(pred_tmp)
                
        thresholds = [np.nanpercentile(pred_before_onset, i) for i in range(100)]        
        thresholds = np.concatenate(([0], thresholds, [max_score, df_with_event[PD_COL].max()]))
        thresholds = np.interp(np.arange(0, len(thresholds)+0.25, 0.25), range(len(thresholds)), thresholds)        
        thresholds = np.concatenate((thresholds[:315:3], thresholds[315:]))
        thresholds = thresholds[::2]

        if idx_threshold >= len(thresholds):
            exit(0)
            
    stable_frac = [df_with_event[df_with_event[PID_COL]==pid].Stable.sum()/df_with_event[df_with_event[PID_COL]==pid].shape[0] for pid in df_with_event[PID_COL].unique()]
    inevent_frac = [df_with_event[df_with_event[PID_COL]==pid].InEvent.sum()/df_with_event[df_with_event[PID_COL]==pid].shape[0] for pid in df_with_event[PID_COL].unique()]
    unknown_frac = [df_with_event[df_with_event[PID_COL]==pid].Unknown.sum()/df_with_event[df_with_event[PID_COL]==pid].shape[0] for pid in df_with_event[PID_COL].unique()]
    event_ratio = np.array(inevent_frac) / (np.array(inevent_frac)+np.array(stable_frac))
        
    if fixed_recall is None and fixed_precision is None:
        TA = []
        FA = []
        CE = []
        ME = []

        TA_first =  []
        FA_first =  []
        CE_first =  []
        ME_first =  []
        
        if idx_threshold is None:
            pass
        else:
            thresholds  = thresholds[[idx_threshold]]
            
    else:
        res_f = ("rand%d_"%random_seed if RANDOM else "")+"tg-%d_tr-%d_dt-%d_ws-%d_ts-%d"%(t_mingap, t_reset, t_delta, t_window, t_silence)
        if idx_threshold is None and idx_batch is None:
            pass
        elif idx_batch is None:
            res_f = res_f + "_cnts_i%d"%idx_threshold
        else:
            res_f = res_f + "_batch_i%s"%lst_files[0].split("_")[1][:-3]
            
        if not os.path.exists(alarm_path):
            os.mkdir(alarm_path)            
            
        if fixed_recall is not None and fixed_precision is None:
            alarm_path = os.path.join(alarm_path, (res_f+"_rec-%g.h5"%fixed_recall))
        elif fixed_recall is None and fixed_precision is not None:
            alarm_path = os.path.join(alarm_path, (res_f+"_prec-%g.h5"%fixed_precision))

    if correct_false_alarm and ep_problem=="renal":
        df_in = [read_furosemide_records(IMPUTE_DATA_PATH, f) for f in lst_files]
        df_in = pd.concat(df_in)
        df_in = df_in.reset_index(drop=True)

        
    df_static = pd.read_hdf(STATIC_DATA_PATH)
    if select_cohort is None or select_cohort.lower() in ['all', 'none']:
        pass
        
    elif select_cohort.lower()=="case":
        lst_pid = np.load("lst_pids_furo_new.npy")
        
    elif select_cohort.lower()=="ctrl":
        lst_pid = np.load("lst_pids_no_furo_new.npy")
        
    elif select_cohort.lower()=="male":
        lst_pid = [p for p in lst_pid if p in df_static[df_static.Sex=="M"][PID_COL]]
        
    elif select_cohort.lower()=="female":
        lst_pid = [p for p in lst_pid if p in df_static[df_static.Sex=="F"][PID_COL]]
        
    elif select_cohort.lower()=="emergency":
        lst_pid = [p for p in lst_pid if p in df_static[df_static.Emergency==1][PID_COL]]
        
    elif select_cohort.lower()=="nonemergency":
        lst_pid = [p for p in lst_pid if p in df_static[df_static.Emergency==0][PID_COL]]
        
    elif select_cohort.lower()=="surgical":
        lst_pid = [p for p in lst_pid if p in df_static[df_static.Surgical>0][PID_COL]]
        
    elif select_cohort.lower()=="nonsurgical":
        lst_pid = [p for p in lst_pid if p in df_static[df_static.Surgical==0][PID_COL]]
        
    elif 'apachepatgroup' in select_cohort:
        apachecode=int(select_cohort[14:])
        lst_pid = [p for p in lst_pid if p in df_static[df_static.APACHEPatGroup==apachecode][PID_COL]]
        
    elif 'age' in select_cohort:
        min_age = int(select_cohort.split("_")[1])
        max_age = int(select_cohort.split("_")[2])
        lst_pid = [p for p in lst_pid if p in df_static[(df_static.Age>=min_age)&(df_static.Age<=max_age)][PID_COL]]
        
    else:
        raise Exception("Not implemented.")

    for tau in thresholds:
        for pid in lst_pids:
            if (df_with_event[PID_COL]==pid).sum() == 0:
                continue
            
            df = df_with_event[df_with_event[PID_COL]==pid].set_index(REL_DT_COL)
            if correct_false_alarm and ep_problem=="renal" and correct_false_alarm_mode=="DelFA": 
                df_in_tmp  = df_in[df_in[PID_COL]==pid]
                if (df_in_tmp[FURO_ID]>0).sum()>0:
                    dt_drug = df_in_tmp[df_in_tmp[FURO_ID]>0][REL_DT_COL].values
                    dt_drug_beg = dt_drug[1:][np.diff(dt_drug)>300]
                    dt_drug_beg = np.concatenate((dt_drug[:1], dt_drug_beg))
                    
                    dt_drug_end = dt_drug[:-1][np.diff(dt_drug)>300]
                    dt_drug_end = np.concatenate((dt_drug_end, dt_drug[-1:]))

                    
                    for i in range(len(dt_drug_beg)):
                        if i==0:
                            tmp  = df[(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                            
                        else:
                            tmp  = df[(df.index>=dt_drug_end[i-1])&(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                            
                        df.loc[tmp.index, PD_COL] = np.nan

            if len(df)==0:
                continue
            
            get_alarm_info(df, tau, prediction_score_interval)
            
            if correct_false_alarm and ep_problem=="renal" and correct_false_alarm_mode=="FA2TA": 
                df_in_tmp  = df_in[df_in[PID_COL]==pid]
                
                if (df_in_tmp[FURO_ID]>0).sum()>0:
                    dt_drug = df_in_tmp[df_in_tmp[FURO_ID]>0][REL_DT_COL].values
                    dt_drug_beg = dt_drug[1:][np.diff(dt_drug)>300]
                    dt_drug_beg = np.concatenate((dt_drug[:1], dt_drug_beg))
                    
                    dt_drug_end = dt_drug[:-1][np.diff(dt_drug)>300]
                    dt_drug_end = np.concatenate((dt_drug_end, dt_drug[-1:]))
    
                    for i in range(len(dt_drug_beg)):
                        if i==0:
                            tmp  = df[(df["Alarm"])&(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                            
                        else:
                            tmp  = df[(df["Alarm"])&(df.index>=dt_drug_end[i-1])&(df.index<dt_drug_beg[i]+correct_false_alarm_win*3600)&(df.index>=dt_drug_beg[i])]
                            
                        if ((tmp["Alarm"])&(~tmp["TrueAlarm"])).sum() > 0:
                            df.loc[tmp.index,"TrueAlarm"] = True

            elif correct_false_alarm and ep_problem=="renal" and correct_false_alarm_mode=="DelFA":
                for dt in df[df["CatchedOnset"]].index:
                    if df.loc[dt-max_sec:dt-min_sec,"Alarm"].sum() == 0:
                        df.loc[dt,"Onset"] = False
                        df.loc[dt,"CatchedOnset"] = False
                        
                    else:
                        pass

            if fixed_recall is None and fixed_precision is None:
                TA.append([tau, pid, df["TrueAlarm"].sum()])
                FA.append([tau, pid, df["Alarm"].sum() - df["TrueAlarm"].sum()])
                CE.append([tau, pid, df["CatchedOnset"].sum()])
                ME.append([tau, pid, df["Onset"].sum() - df["CatchedOnset"].sum()])

                if df["Onset"].sum() > 0:
                    dt_be4_first = df.index[df.index<=df.index[df["Onset"]][0]]
                    TA_first.append([tau, pid, df.loc[dt_be4_first]["TrueAlarm"].sum()])
                    FA_first.append([tau, pid, df.loc[dt_be4_first]["Alarm"].sum() - df.loc[dt_be4_first]["TrueAlarm"].sum()])
                    CE_first.append([tau, pid, df.loc[dt_be4_first]["CatchedOnset"].sum()])
                    ME_first.append([tau, pid, df.loc[dt_be4_first]["Onset"].sum() - df.loc[dt_be4_first]["CatchedOnset"].sum()])
                    
                else:
                    TA_first.append([tau, pid, 0])
                    FA_first.append([tau, pid, df["Alarm"].sum()])
                    CE_first.append([tau, pid, 0])
                    ME_first.append([tau, pid, 0])
                    
            else:
                df_alarm = df[df["Alarm"]].copy()
                if len(df_alarm) > 0:
                    df_alarm.loc[:,"t2onset_h"] = np.nan
                for i, dt_onset in enumerate(df.index[df["Onset"]]):
                    if i == 0:
                        idx_alarm = df_alarm.index[df_alarm.index<dt_onset]
                    else:
                        idx_alarm = df_alarm.index[(df_alarm.index>df.index[df["Onset"]][i-1])&(df_alarm.index<dt_onset)]
                    df_alarm.loc[idx_alarm,"t2onset_h"] = (dt_onset - idx_alarm) / 3600
                if len(df_alarm) > 0:
                    df_alarm = pd.concat([df_alarm, df[df["Onset"]|df.EventEnd]]).sort_index()
                else:
                    df_alarm =  df[df["Onset"]|df.EventEnd].copy()
                    if len(df_alarm) > 0:
                        df_alarm.loc[:,"t2onset_h"] = np.nan
                
                if len(df_alarm) == 0:
                    df_alarm = df.iloc[[-1]].copy()
                    df_alarm.loc[:,"t2onset_h"] = np.nan

                df_alarm = df_alarm[[PID_COL, "Alarm","TrueAlarm","Onset","CatchedOnset", "EventEnd", "t2onset_h", PD_COL, DT_COL]]
                df_alarm.loc[:,"DiscRelDatetime"] = df.index[-1]
                df_alarm.to_hdf(alarm_path, "p%d"%pid, complevel=5, complib="blosc:lz4")

    if fixed_recall is None and fixed_precision is None:
        TA = pd.DataFrame(TA, columns=['tau',PID_COL, 'TA'])
        FA = pd.DataFrame(FA, columns=['tau',PID_COL, 'FA'])
        ME = pd.DataFrame(ME, columns=['tau',PID_COL, 'ME'])
        CE = pd.DataFrame(CE, columns=['tau',PID_COL, 'CE'])

        TA = TA[['tau', 'TA']].groupby('tau').sum()
        FA = FA[['tau', 'FA']].groupby('tau').sum()
        ME = ME[['tau', 'ME']].groupby('tau').sum()
        CE = CE[['tau', 'CE']].groupby('tau').sum()

        TA_first = pd.DataFrame(TA_first, columns=['tau',PID_COL, 'TA'])
        FA_first = pd.DataFrame(FA_first, columns=['tau',PID_COL, 'FA'])
        ME_first = pd.DataFrame(ME_first, columns=['tau',PID_COL, 'ME'])
        CE_first = pd.DataFrame(CE_first, columns=['tau',PID_COL, 'CE'])

        TA_first = TA_first[['tau', 'TA']].groupby('tau').sum()
        FA_first = FA_first[['tau', 'FA']].groupby('tau').sum()
        ME_first = ME_first[['tau', 'ME']].groupby('tau').sum()
        CE_first = CE_first[['tau', 'CE']].groupby('tau').sum()


        cnts = pd.concat([TA,FA,CE,ME], axis=1).reset_index()
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        res_f = ("rand%d_"%random_seed if RANDOM else "")+"tg-%d_tr-%d_dt-%d_ws-%d_ts-%d"%(t_mingap, t_reset, t_delta, t_window, t_silence)
        if idx_threshold is None and idx_batch is None:
            pass
        
        elif idx_batch is None:
            res_f = res_f + "_cnts_i%d"%idx_threshold
            
        else:
            res_f = res_f + "_batch_i%d"%idx_batch
            
        if not DEBUG:
            cnts.to_csv(os.path.join(out_path, (res_f+".csv")), index=False)
            
        print(os.path.join(out_path, res_f))


        cnts_first = pd.concat([TA_first,FA_first,CE_first,ME_first], axis=1).reset_index()
        res_f_first = ("rand%d_"%random_seed if RANDOM else "")+"tg-%d_tr-%d_dt-%d_ws-%d_ts-%d_first"%(t_mingap, t_reset, t_delta, t_window, t_silence)
        if idx_threshold is None and idx_batch is None:
            pass
        
        elif idx_batch is None:
            res_f_first = res_f_first + "_cnts_i%d"%idx_threshold
            
        else:
            res_f_first = res_f_first + "_batch_i%d"%idx_batch
