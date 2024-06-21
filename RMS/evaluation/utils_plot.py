 import os
import gc
import h5py
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sys
import json
from copy import copy

import matplotlib.pyplot as plt
cm = 1/2.54 
from matplotlib import gridspec 
import seaborn as sns

def compute_precision_recall(df, calibrated_s=None):
    if calibrated_s is not None:
        df.loc[:,'FA'] = df.FA * calibrated_s    
    
    if df.shape[1] == 5:
        df.loc[:,"rec"] = df.CE / (df.CE+df.ME)
        df.loc[:,"prec"] = df.TA / (df.TA+df.FA)

        df.drop(df.index[df.rec.isnull()|df.prec.isnull()], inplace=True)
        df.sort_values(["rec", "tau", "prec"], inplace=True)
        #     df.sort_values("prec", inplace=True, ascending=False)
        df.drop_duplicates(["rec", "prec"], inplace=True)
        df.drop_duplicates("rec", keep="first", inplace=True)
        df.sort_values(["rec", "prec"], inplace=True)
        if df.iloc[0].rec > 0.02:
            new_idx = df.index.max()+1
            df.loc[new_idx, "rec"] = 0
            df.loc[new_idx, "prec"] = df.iloc[0].prec
            df.sort_values(["rec", "tau", "prec"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df
    else:
        lst_df = [] 
        num_run = int((df.shape[1]-1)/4)
        for i in range(num_run):
            df.loc[:,"rec%d"%i] = df["CE%d"%i] / (df["CE%d"%i]+df["ME%d"%i])
            df.loc[:,"prec%d"%i] = df["TA%d"%i] / (df["TA%d"%i]+df["FA%d"%i])
            lst_df.append(df[["rec%d"%i, "prec%d"%i]].dropna().sort_values(["rec%d"%i, "prec%d"%i],ascending=False).drop_duplicates("rec%d"%i, keep="last").set_index("rec%d"%i).sort_index(ascending=False))
            df_m = pd.concat(lst_df, axis=1)
            df_m.sort_index(inplace=True)
            df_m["prec"] = df_m[[col for col in df_m.columns if "prec" in col]].mean(axis=1)
            df_m.reset_index(inplace=True)
            df_m.rename(columns={"index": "rec"}, inplace=True)
            df_m = df_m[["rec", "prec"]].copy()
        for col in df_m.columns:
            df_m.loc[:,col] = df_m[col].interpolate(method="index")
            lst_df_rec1 = []
        for i in range(num_run):
            lst_df_rec1.append(df[df["rec%d"%i]==1][["rec%d"%i, "prec%d"%i]].reset_index(drop=True))
            df_m_rec1 = pd.concat(lst_df_rec1, axis=1)
            df_m_rec1["prec"] = df_m_rec1[[col for col in df_m_rec1.columns if "prec" in col]].mean(axis=1)
            df_m_rec1["rec"] = 1
            df_m_rec1 = df_m_rec1[["rec", "prec"]].copy()
            df = pd.concat([df_m, df_m_rec1]).reset_index(drop=True).sort_values(["rec"], ascending=False)
        return df

def get_keystr(configs):
    try:
        f_key = "tg-%d"%configs["t_mingap_m"]
        f_key += "_tr-%d"%configs["t_reset_m"]
        f_key += "_dt-0"
        f_key += "_ws-%d"%configs["t_window_m"]
        f_key += "_ts-%d"%configs["t_silence_m"]
    except:
        f_key = "tg-%d"%configs["t_mingap"]
        f_key += "_tr-%d"%configs["t_reset"]
        f_key += "_dt-0"
        f_key += "_ws-%d"%configs["t_window"]
        f_key += "_ts-%d"%configs["t_silence"]
    return f_key

def _get_df(respath, configs, calibrated_s=None, RANDOM=False, random_seed=None, onset_type=None):
    '''
    read TA, FA, ME, CE information and compute rec and prec
    '''
    f_key = get_keystr(configs)
    if RANDOM:
        if random_seed is None:
            f_key = "rand_" + f_key
        else:
            f_key = "rand%d_"%random_seed + f_key
    else:
        pass
    
    if onset_type == "first":
        f_key += "_first"
        
    aggrfile = [f for f in os.listdir(respath) if f==(f_key+".csv")]  
    if len(aggrfile) > 0:
        df = pd.read_csv(os.path.join(respath, aggrfile[0]))
        df = compute_precision_recall(df, calibrated_s=calibrated_s)
    else:
        if onset_type=="first":
            batchfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "batch" in f]
        else:
            batchfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "batch" in f and "first" not in f]
            
        if len(batchfiles) > 0:
            df = [pd.read_csv(os.path.join(respath, f)) for f in batchfiles]
            df = pd.concat(df).groupby("tau").sum().reset_index()
        else:
            if onset_type=="first":
                cntfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "cnts" in f]
            else:
                cntfiles = [f for f in os.listdir(respath) if f[:len(f_key)]==f_key and "cnts" in f and "first" not in f]
                df = [pd.read_csv(os.path.join(respath, f)) for f in cntfiles]    
                df = pd.concat(df)
                
        df.sort_values("tau", inplace=True)
        df = df.reset_index(drop=True)
        df2write = df.copy()
        df = compute_precision_recall(df, calibrated_s=calibrated_s)
        df2write.to_csv(os.path.join(respath, f_key+".csv"), index=False)
        # if RANDOM:
        #     df.loc[df.index[df.rec<0.6],"prec"] = df.loc[df.index[df.rec>=0.6],"prec"].iloc[0]
    return df

def get_df(datapath, res_dir, tsplit, configs, calibrated_s=None, RANDOM=False, random_seed=2021, onset_type=None):
    respath = os.path.join(datapath, res_dir, tsplit)
    if onset_type is None or onset_type=="first":
        return _get_df(respath, configs, calibrated_s=calibrated_s, RANDOM=RANDOM, random_seed=random_seed, onset_type=onset_type)
    elif onset_type=="sequential":
        df_a = _get_df(respath, configs, calibrated_s=calibrated_s, RANDOM=RANDOM, random_seed=random_seed, onset_type=None).set_index("tau")
        df_f = _get_df(respath, configs, calibrated_s=calibrated_s, RANDOM=RANDOM, random_seed=random_seed, onset_type="first").set_index("tau")
        df = (df_a - df_f).iloc[:,:4].reset_index()
        df = compute_precision_recall(df, calibrated_s=calibrated_s)
        return df
    else:
        raise Exception("not implemented")

def plot_prc(datapath,
             lst_res_dir, 
             lst_label, 
             lst_color, 
             lst_configs,
             lst_linestyle=None,
             onset_type=None, 
             figsize=(10,10), 
             title=None, 
             fig_path=None, 
             errorbar=False):
    
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    for i, res_dir in enumerate(lst_res_dir):            
        if type(lst_configs) == dict:
            configs = lst_configs
        elif type(lst_configs) == list:
            configs = lst_configs[i]
            
        RANDOM = True if "random" in lst_label[i].lower() else False
        
        try:
            if RANDOM:
                try:
                    auc_rand = []
                    for random_seed in np.arange(2017,2022):
                        df = get_df(datapath, res_dir, "point_est", configs, RANDOM=True, random_seed=random_seed, onset_type=onset_type)
                        auc_rand.append( metrics.auc(df.rec, df.prec) )
                        df.loc[:,"prec"] = np.mean(auc_rand)
                except:
                    df = get_df(datapath, res_dir, "point_est", configs, RANDOM=True, onset_type=onset_type)
            else:
                df = get_df(datapath, res_dir, "point_est", configs, RANDOM=False, onset_type=onset_type)
        except:
            print(res_dir)
            raise Exception("Error")
        if "fio" in res_dir.lower():
            plt.scatter(df[df.tau==0.5].iloc[0].rec, 
                        df[df.tau==0.5].iloc[0].prec, 
                        label=lst_label[i], 
                        color=lst_color[i], 
                        marker="d")
        else:
            auc = metrics.auc(df.rec, df.prec)
            plt.plot(df.rec, df.prec,
                     label=lst_label[i]+("" if errorbar else " (AUPRC=%3.3f)"%auc), 
                     color=lst_color[i], 
                     marker=".",
                     linestyle=":" if RANDOM else ("solid" if lst_linestyle is None else lst_linestyle[i]))
            
        if "fio" in res_dir.lower():
            continue
        if errorbar:
            df_tsplit = []
            for j in np.arange(1,6):
                df_tmp = get_df(datapath, res_dir, "temporal_%d"%j, configs, RANDOM=RANDOM, onset_type=onset_type)
                df_tmp.set_index("rec", inplace=True)
                df_tmp.rename(columns={"prec": "prec_%d"%j}, inplace=True)
                df_tsplit.append(df_tmp[["prec_%d"%j]])
                
            df_merge = df.set_index("rec")[["prec"]]
            for tmp in df_tsplit:
                df_merge = df_merge.merge(tmp, how="outer", left_index=True, right_index=True)
                df_merge.sort_index(inplace=True)
            for col in df_merge.columns:
                df_merge.loc[df_merge.index,col] = df_merge[col].interpolate(method="index")
                df.sort_values("rec", inplace=True)
                df_merge = df_merge.loc[df.rec.values]
                
                
            auc_std = np.nanstd([metrics.auc(df_merge[df_merge.index.notnull()&df_merge["prec_%d"%j].notnull()].index, 
                                             df_merge[df_merge.index.notnull()&df_merge["prec_%d"%j].notnull()]["prec_%d"%j]) for j in np.arange(1,6)])
            prec_std = df_merge[["prec_%d"%j for j in np.arange(1,6)]].std(axis=1)
            plt.fill_between(df_merge.index, 
                             df_merge.prec-prec_std, 
                             df_merge.prec+prec_std, 
                             label="_nolegend_", 
                             color=lst_color[i],
                             alpha=0.2)
            plt.text(0.02,0.25-0.05*(i+1), "%3.3f $\pm$ %3.3f"%(auc, auc_std), color=lst_color[i])
            
        else:
            pass
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.axis("equal")
        plt.ylim([0,1])
        plt.xlim([0,1])
        plt.grid()
    if errorbar:
        plt.text(0.02,0.25, "AUPRC:")
        plt.legend()
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if title is not None:
        plt.title(title)
    if fig_path is not None:
        plt.savefig(fig_path)
    if figsize is not None:
        plt.show()
        
        
def interpolate_prec(df, lst_rec):
    df.set_index("rec", inplace=True)
    for rec in lst_rec:
        df.loc[rec,"prec"] = np.nan
        df.sort_index(inplace=True)
        df.loc[:,"prec"] = df.prec.interpolate(method="index")
        df.reset_index(inplace=True)
    return df


def get_alarm(datapath, res_dir, tsplit, configs, tbins, fixed_rec=0.9, fixed_prec=None, CUMUL=True, RANDOM=False):
    f_key = get_keystr(configs)
    if RANDOM:
        f_key = "rand_" + f_key
        
    if fixed_prec is None:
        alarm_path = os.path.join(datapath, res_dir, "%s_alarm"%tsplit, (f_key+"_rec-%g.h5"%fixed_rec))
    else:
        alarm_path = os.path.join(datapath, res_dir, "%s_alarm"%tsplit, (f_key+"_prec-%g.h5"%fixed_prec))
        
    if os.path.exists(alarm_path):
        try:
            all_df = pd.read_hdf(alarm_path, "data")
        except:
            
            df_store = pd.HDFStore(alarm_path, "r")
            lst_pids = [k for k in df_store.keys()]
            df_store.close()
            all_df = []
            for pid in lst_pids:
                all_df.append(pd.read_hdf(alarm_path, pid))
                all_df = pd.concat(all_df).reset_index()
    else:
        print("Merging alarm files")
        all_df = []
        if RANDOM:
            if fixed_prec is None:
                alarm_batchfiles = [f for f in os.listdir(os.path.join(datapath, res_dir, "%s_alarm"%tsplit)) if "_rec-%g."%fixed_rec in f and "batch" in f and "rand_" in f and f_key in f]
            else:
                alarm_batchfiles = [f for f in os.listdir(os.path.join(datapath, res_dir, "%s_alarm"%tsplit)) if "_prec-%g."%fixed_prec in f and "batch" in f and "rand_" in f and f_key in f]
                print(alarm_batchfiles)
                
        else:
            if fixed_prec is None:
                alarm_batchfiles = [f for f in os.listdir(os.path.join(datapath, res_dir, "%s_alarm"%tsplit)) if "_rec-%g."%fixed_rec in f and "batch" in f and "rand_" not in f and f_key in f]
            else:
                alarm_batchfiles = [f for f in os.listdir(os.path.join(datapath, res_dir, "%s_alarm"%tsplit)) if "_prec-%g."%fixed_prec in f and "batch" in f and "rand_" not in f and f_key in f]
                print(alarm_batchfiles)
                
        for i, f in enumerate(alarm_batchfiles):
            tmp = h5py.File(os.path.join(datapath, res_dir, "%s_alarm"%tsplit, f), mode="r")
            pids = [k for k in tmp.keys() if k!="data"]
            tmp.close()
            all_df.extend([pd.read_hdf(os.path.join(datapath, res_dir, "%s_alarm"%tsplit, f), pid) for pid in pids])
            print("Finish reading batch %d"%(i+1))
            
        
        all_df = pd.concat(all_df).reset_index()
        all_df.to_hdf(alarm_path, "data", complevel=5, complib="blosc:lz4")
        print("Finish writing")
      
    pids = np.sort(all_df.PatientID.unique())
    FA_e = 0
    LS_e = 0
    FA_n = 0
    LS_n = 0
    ce_bins = np.zeros(tbins.shape)
    me_bins = np.zeros(tbins.shape)
    
    if not CUMUL:
        fa_bins_s = np.zeros(tbins.shape)
        ta_bins_s = np.zeros(tbins.shape)
        fa_bins_l = np.zeros(tbins.shape)
        ta_bins_l = np.zeros(tbins.shape)
        a1_time_s = []
        a1_time_l = []
        aa_time_s = []
        aa_time_l = []
        
    for pid in pids:
        #         df_p = pd.read_hdf(alarm_path, pid).reset_index()
        df_p = all_df[all_df.PatientID==pid].copy()
        all_onset = df_p.RelDatetime[df_p.Onset].values
        all_end = df_p.RelDatetime[df_p.EventEnd].values
        if df_p.Onset.sum() == 0:
            FA_n += df_p.Alarm.sum()
            LS_n += (df_p.iloc[-1].DiscRelDatetime / 3600)
        else:
            for j, tBE4 in enumerate(tbins):
                if j==len(tbins)-1:
                    continue

                for i, tONS in enumerate(all_onset):
                    if CUMUL:
                        # pre_event_ta = df_p[(df_p.RelDatetime<=tONS)&(df_p.t2onset_h>=tBE4)&(df_p.t2onset_h<configs["t_window_h"])]
                        # pre_event_ta = df_p[(df_p.RelDatetime<=tONS)&(df_p.t2onset_h>=tBE4)&(df_p.t2onset_h<np.max(tbins))]
                        pre_event_ta = df_p[(df_p.RelDatetime<=tONS)&(df_p.t2onset_h>=tBE4)&(df_p.t2onset_h<np.max(tbins))]
                    else:
                        pre_event_ta = df_p[(df_p.RelDatetime<=tONS)&(df_p.t2onset_h>=tBE4)&(df_p.t2onset_h<tbins[j+1])]
                        
                        
                    pre_event_fa = df_p[(df_p.RelDatetime<=tONS-configs["t_window_h"]*3600)]
                    pre_event_aa = df_p[df_p.RelDatetime<=tONS]
                        

                    if i==0:
                        LS_tmp = tONS / 3600
                    else:
                        pre_event_ta = pre_event_ta[pre_event_ta.RelDatetime>all_end[i-1]]
                        pre_event_fa = pre_event_fa[pre_event_fa.RelDatetime>all_end[i-1]]
                        pre_event_aa = pre_event_aa[pre_event_aa.RelDatetime>all_end[i-1]]
                        LS_tmp = (tONS - all_end[i-1]) / 3600

                    if not CUMUL:
                        if LS_tmp <= configs["t_window_h"]:
                            if tbins[j+1]<=configs["t_window_h"]:
                                ta_bins_s[j]+=pre_event_ta.Alarm.sum()
                            else:
                                fa_bins_s[j]+=pre_event_ta.Alarm.sum()
                            if j==0:
                                a1_time_s.extend(pre_event_aa[pre_event_aa.Alarm].t2onset_h.values[:1])
                                aa_time_s.extend(pre_event_aa[pre_event_aa.Alarm].t2onset_h.values)
                        else:
                            if tbins[j+1]<=configs["t_window_h"]:
                                ta_bins_l[j]+=pre_event_ta.Alarm.sum()
                            else:
                                fa_bins_l[j]+=pre_event_ta.Alarm.sum()
                            if j==0:
                                a1_time_l.extend(pre_event_aa[pre_event_aa.TrueAlarm].t2onset_h.values[:1])
                                aa_time_l.extend(pre_event_aa[pre_event_aa.Alarm].t2onset_h.values)
                                

                    if LS_tmp < tBE4:
                        continue
                    
                    if tbins[j] <= 24:
                        if pre_event_ta.TrueAlarm.sum() > 0:
                            ce_bins[j]+=1

                        else:
                            me_bins[j]+=1
                    else:
                        if pre_event_ta.Alarm.sum() > 0:
                            ce_bins[j]+=1

                        else:
                            me_bins[j]+=1
                        
                        
                    if j == 0:
                        LS_e += max((LS_tmp-configs["t_window_h"]), 0)
                        FA_e += (pre_event_fa.Alarm.sum() - pre_event_fa.TrueAlarm.sum())
                        
                if j == 0:
                    pre_event_fa = df_p[df_p.RelDatetime>all_end[-1]]
                    LS_e += max((df_p.iloc[-1].DiscRelDatetime-all_end[-1])/3600, 0)
                    FA_e += (pre_event_fa.Alarm.sum() - pre_event_fa.TrueAlarm.sum())
    
    print(fixed_rec, all_df.CatchedOnset.sum(), all_df.Onset.sum())
    print(ce_bins[0], ce_bins[0]+me_bins[0])
    rec_bins = ce_bins / (ce_bins+me_bins)
    fa_rate_e = FA_e / LS_e
    fa_rate_n = FA_n / LS_n
    if CUMUL:
        return rec_bins, fa_rate_e, fa_rate_n
    else:
        return ta_bins_s, fa_bins_s, a1_time_s, aa_time_s, ta_bins_l, fa_bins_l, a1_time_l, aa_time_l

def plot_cumulative_recall(datapath, res_dir, configs, tbins, endpoint_name=None, lst_rec=[0.9], errorbar=False, fig_path=None, RANDOM=False):
    lst_split = ["point_est"] 
    if errorbar:
        lst_split += ["temporal_%d"%i for i in np.arange(1,6)]
        dict_recall = dict()
        dict_farate = dict()
        dict_prec = dict()
    for tsplit in lst_split:
        dict_recall.update({tsplit:dict()})
        dict_farate.update({tsplit:dict()})
        dict_prec.update({tsplit:dict()})
        df = get_df(datapath, res_dir, tsplit, configs)
        for n, fixed_recall in enumerate(lst_rec):
            rec_bins, fa_rate_e, fa_rate_n = get_alarm(datapath, res_dir, tsplit, configs, tbins, fixed_rec=fixed_recall, RANDOM=RANDOM)
            dict_recall[tsplit].update({fixed_recall:rec_bins})
            dict_farate[tsplit].update({fixed_recall:fa_rate_e})
            dict_prec[tsplit].update({fixed_recall: df.iloc[np.argmin(np.abs(df.rec.values-fixed_recall))].prec})
            gc.collect()

    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([],[])
    ax.set_xlabel("Alarm time before event - $\\mathbf{t}$ (h)")
    ax = ax.twinx()
    cmaps = sns.color_palette("tab20c")[4:7][::-1]
    markers = ["d", "*", "."]
    for i, key in enumerate(lst_rec):
        rec_bins = dict_recall["point_est"][key]
        prec = dict_prec["point_est"][key]
        
        if errorbar:
            rec_bins_std = np.std([dict_recall[tsplit][key] for tsplit in lst_split[1:]], axis=0)
            prec_std = np.std([dict_prec[tsplit][key] for tsplit in lst_split[1:]])

            plt.fill_between(-tbins[::-1], 
                             rec_bins[::-1]-rec_bins_std[::-1], 
                             rec_bins[::-1]+rec_bins_std[::-1], alpha=0.2, 
                             label="_nolegend_", 
                             color=cmaps[i])
            
        plt.plot(-tbins[::-1], 
                 rec_bins[::-1], 
                 marker=markers[i], 
                 label="Overall recall=%0.2f, Precision=%0.2f"%(key,prec)+("$\pm$%0.2f"%prec_std if errorbar else ""),
                 color=cmaps[i])
        plt.axvline(-tbins[0], color="C3", linestyle="--")
        plt.legend(loc=4)
        plt.ylim([0.3,1.05])

    xleft = plt.xlim()[0]
    xright = plt.xlim()[1]
    xscale = ( xright - xleft ) / 10
    if endpoint_name is not None:
        plt.text(xright-5*xscale, 1, endpoint_name, horizontalalignment="left", color="C3", fontweight="bold")
        plt.arrow(xright-5*xscale+3.5*xscale, 1.01, 0.8*xscale, 0, color="C3", width=0.003*xscale, head_width=0.006*xscale,head_length=0.1*xscale, zorder=10)

    
    ax.set_ylabel("Recall rate by alarms at $\\mathbf{t}$ h before event")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    
    d = .4  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax.plot([1, 1], [0.08, 0.06], transform=ax.transAxes, **kwargs)
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='w', mec='w', mew=5, clip_on=False, zorder=11)
    ax.plot([1], [0.07], transform=ax.transAxes, **kwargs)

    plt.xticks(-tbins[::-2],tbins[::-2])
    plt.yticks(np.arange(0.3, 1.1, 0.1), ["%0.1f"%x for x in [0.0]+np.arange(0.4, 1.1, 0.1).tolist()])
    
    plt.grid(axis="y")
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.show()
        
    
def merge_alarm_files(alarm_path, batch_files, merged_file):
    print("Merging alarm files")
    df = []
    for i, f in enumerate(batch_files):
        tmp =  h5py.File(os.path.join(alarm_path, f), mode="r")
        pids = [k for k in tmp.keys() if k!="data"]
        tmp.close()
        df.extend([pd.read_hdf(os.path.join(alarm_path, f), pid) for pid in pids])
        sys.stdout.write("Finish reading batch %d\r"%(i+1))
        sys.stdout.flush()
        df = pd.concat(df).reset_index()
        df.to_hdf(os.path.join(alarm_path, merged_file), "data", complevel=5, complib="blosc:lz4")
        print("\nFinish merging")
    return df

def get_alarm_dataframe(datapath, 
                        res_dir, 
                        tsplit, 
                        configs, 
                        fixed_rec=None, 
                        fixed_prec=None, 
                        FIRST_ONSET=False,
                        RANDOM=False):
    
    f_key = get_keystr(configs)
    f_key = ("rand2021_" if RANDOM else "") + f_key
    if fixed_rec is not None and fixed_prec is None:
        alarm_file = f_key+"_rec-%g.h5"%fixed_rec
    elif fixed_rec is None and fixed_prec is not None:
        alarm_file = f_key+"_prec-%g.h5"%fixed_prec
    else:
        raise Exception("Pick a fixed recall or precision.")    
    alarm_dir = "%s_alarm"%tsplit + ("_first" if FIRST_ONSET else "")    
    alarm_path = os.path.join(datapath, res_dir, alarm_dir)
    
    if os.path.exists(os.path.join(alarm_path, alarm_file)):
        try:
            df = pd.read_hdf(os.path.join(alarm_path, alarm_file), "data")
        except:
            
            df_store = pd.HDFStore(os.path.join(alarm_path, alarm_file), "r")
            lst_pids = [k for k in df_store.keys()]
            df_store.close()
            df = []
            for pid in lst_pids:
                df.append(pd.read_hdf(os.path.join(alarm_path, alarm_file), pid))
                df = pd.concat(df).reset_index()
    else:
        if fixed_rec is not None and fixed_prec is None:
            batch_files = [f for f in os.listdir(alarm_path) if f[:len(f_key)]==f_key and "_rec-%g"%fixed_rec in f and "batch" in f]
        else:
            batch_files = [f for f in os.listdir(alarm_path) if f[:len(f_key)]==f_key and "_prec-%g"%fixed_prec in f and "batch" in f]
            df = merge_alarm_files(alarm_path, batch_files, alarm_file)
    return df

def get_alarm_info_first_other(datapath, 
                               res_dir, 
                               tsplit, 
                               configs, 
                               fixed_rec=None, 
                               fixed_prec=None, 
                               FIRST_ONSET=False):
    
    df = get_alarm_dataframe(datapath, 
                             res_dir, 
                             tsplit, 
                             configs, 
                             fixed_rec=fixed_rec, 
                             fixed_prec=fixed_prec, 
                             FIRST_ONSET=FIRST_ONSET)
    #     f_key = get_keystr(configs)
    #     if fixed_rec is not None and fixed_prec is None:
    #         alarm_file = f_key+"_rec-%g.h5"%fixed_rec
    #     elif fixed_rec is None and fixed_prec is not None:
    #         alarm_file = f_key+"_prec-%g.h5"%fixed_prec
    #     else:
    #         raise Exception("Pick a fixed recall or precision.")    
    #     alarm_dir = "%s_alarm"%tsplit + ("_first" if FIRST_ONSET else "")    
    #     alarm_path = os.path.join(datapath, res_dir, alarm_dir)
    
#     if os.path.exists(os.path.join(alarm_path, alarm_file)):
#         df = pd.read_hdf(os.path.join(alarm_path, alarm_file), "data")
#     else:
#         if fixed_rec is not None and fixed_prec is None:
#             batch_files = [f for f in os.listdir(alarm_path) if f[:len(f_key)]==f_key and "_rec-%g"%fixed_rec in f and "batch" in f]
#         else:
#             batch_files = [f for f in os.listdir(alarm_path) if f[:len(f_key)]==f_key and "_prec-%g"%fixed_prec in f and "batch" in f]
#         df = merge_alarm_files(alarm_path, batch_files, alarm_file)
        
    dict_onsets = dict(first=dict(cnt_all=0,cnt_catched=0),
                       other=dict(cnt_all=0,cnt_catched=0))
    dict_alarms = dict(first=dict(cnt_true=0,time_true=0,cnt_false=0,time_false=0),
                       other=dict(cnt_true=0,time_true=0,cnt_false=0,time_false=0))
    
    for pid in df.PatientID.unique():
        if df[df.PatientID==pid].Onset.sum() >= 1:
            tmp = df[df.PatientID==pid]
            tmp = tmp[tmp.index<=tmp.index[tmp.Onset][0]]
            dict_onsets["first"]["cnt_all"] += 1
            if tmp.CatchedOnset.sum() > 0:
                dict_onsets["first"]["cnt_catched"] += 1

            window_sec = configs["t_window_h"]*3600
            timeperiod_true = tmp[(tmp.RelDatetime>=0)&(tmp.RelDatetime<tmp.iloc[-1].RelDatetime)&(tmp.RelDatetime>=(tmp.iloc[-1].RelDatetime-window_sec))]
            timeperiod_fals = tmp[(tmp.RelDatetime>=0)&(tmp.RelDatetime<(tmp.iloc[-1].RelDatetime-window_sec))]
            dict_alarms["first"]["cnt_true"] += timeperiod_true.Alarm.sum()
            dict_alarms["first"]["cnt_false"] += timeperiod_fals.Alarm.sum()
            if tmp.iloc[-1].RelDatetime < window_sec:
                dict_alarms["first"]["time_true"] += tmp.iloc[-1].RelDatetime
            else:
                dict_alarms["first"]["time_true"] += window_sec
                dict_alarms["first"]["time_false"] += (tmp.iloc[-1].RelDatetime - window_sec)
                
        if df[df.PatientID==pid].Onset.sum() > 1:
            tmp = df[df.PatientID==pid]
            prev_onset = tmp.index[tmp.Onset][0]
            tmp = tmp[tmp.index>tmp.index[tmp.Onset][0]]
            dict_onsets["other"]["cnt_all"] += tmp.Onset.sum()
            if tmp.CatchedOnset.sum() > 0:
                dict_onsets["other"]["cnt_catched"] += tmp.CatchedOnset.sum()

            for curr_onset in tmp.RelDatetime[tmp.Onset]:
                timeperiod_true = tmp[(tmp.RelDatetime>=prev_onset+configs["t_reset_m"]*60)&(tmp.RelDatetime<curr_onset)&(tmp.RelDatetime>=(curr_onset-window_sec))]
                timeperiod_fals = tmp[(tmp.RelDatetime>=prev_onset+configs["t_reset_m"]*60)&(tmp.RelDatetime<curr_onset)]
                dict_alarms["other"]["cnt_true"] += timeperiod_true.Alarm.sum()
                dict_alarms["other"]["cnt_false"] += timeperiod_fals.Alarm.sum()
                if (curr_onset-prev_onset) < window_sec:
                    dict_alarms["other"]["time_true"] += (curr_onset-prev_onset)-configs["t_reset_m"]*60
                else:
                    dict_alarms["other"]["time_true"] += window_sec
                    dict_alarms["other"]["time_false"] += ((curr_onset-prev_onset-configs["t_reset_m"]*60) - window_sec)
                    prev_onset = curr_onset
                    

    for k in dict_onsets.keys():
        dict_onsets[k].update(dict(recall=dict_onsets[k]["cnt_catched"]/dict_onsets[k]["cnt_all"]))
        dict_alarms[k].update(dict(alarm_freq_true=dict_alarms[k]["cnt_true"]/(dict_alarms[k]["time_true"]/3600),
                                   alarm_freq_false=dict_alarms[k]["cnt_false"]/(dict_alarms[k]["time_false"]/3600)))
        

    return dict_onsets, dict_alarms


def plot_heatmap_silencing_reset(res_dir, lst_t_silencing, lst_t_reset, configs, lst_recall=[0.8, 0.9, 0.95], title=None, fig_path=None):
    if lst_t_silencing.max() > 60:
        lst_t_silencing /= 60    
    if lst_t_reset.max() > 60:
        lst_t_reset /= 60
        
    info_mat = []
    for t_silence_h in lst_t_silencing:
        for t_reset_h in lst_t_reset:
            configs.update(dict(t_silence_m=t_silence_h*60, t_reset_m=t_reset_h*60))
            df = interpolate_prec(get_df(res_dir, "point_est", configs, RANDOM=False), lst_recall).set_index("rec")
            df_rand = interpolate_prec(get_df(res_dir, "point_est", configs, RANDOM=True), lst_recall).set_index("rec")
            for fixed_recall in lst_recall:
                info_mat.append([t_reset_h, t_silence_h, df.loc[fixed_recall].prec, "Precision@Recall=%d%%"%(fixed_recall*100)])
                info_mat.append([t_reset_h, t_silence_h, df_rand.loc[fixed_recall].prec, "rand_Precision@Recall=%d%%"%(fixed_recall*100)])
                
            auc = metrics.auc(np.sort(df.index.values), df.iloc[np.argsort(df.index.values)].prec)
            auc_rand = metrics.auc(np.sort(df_rand.index.values), df_rand.iloc[np.argsort(df_rand.index.values)].prec)
            info_mat.append([t_reset_h, t_silence_h, auc, "AUPRC"])
            info_mat.append([t_reset_h, t_silence_h, auc_rand, "rand_AUPRC"])
            info_mat = pd.DataFrame(info_mat, columns=["$t_{reset}$ (h)", "$t_{silence}$ (h)", "value", "metric"], )

    ncols = len(lst_recall)+1
    nrows = 2
    ylabel = info_mat.columns[0]
    xlabel = info_mat.columns[1]
    fig = plt.figure(figsize=(6*ncols,4*nrows))
    for n, fixed_recall in enumerate(lst_recall):
        metric_col = "Precision@Recall=%d%%"%(fixed_recall*100)
        plt.subplot(nrows, ncols, 0*(ncols)+n+1)
        df_metric = info_mat[info_mat.metric==metric_col][[ylabel, xlabel,"value"]]
        mat_metric = df_metric.pivot_table(index=ylabel, columns=xlabel, values="value")        
        sns.heatmap(mat_metric, annot=True, fmt="3.3f")
        plt.title(metric_col)
        plt.tight_layout()
        
        plt.subplot(nrows, ncols, 1*(ncols)+n+1)
        df_metric_ratio = (df_metric.set_index([ylabel, xlabel])["value"] / info_mat[info_mat.metric=="rand_"+metric_col].set_index([ylabel, xlabel])["value"]).reset_index()
        mat_metric_ratio = df_metric_ratio.pivot_table(index=ylabel, columns=xlabel, values="value")
        sns.heatmap(mat_metric_ratio, annot=True, fmt="3.3f")
        plt.title("$\\frac{\\mathrm{Precision}_m}{\\mathrm{Precision}_r}$@Recall=%d%%"%(fixed_recall*100))
        plt.tight_layout()
        
    n += 1
    metric_col = "AUPRC"        
    plt.subplot(nrows, ncols, 0*(ncols)+n+1)
    df_metric = info_mat[info_mat.metric==metric_col][[ylabel, xlabel,"value"]]
    mat_metric = df_metric.pivot_table(index=ylabel, columns=xlabel, values="value")        
    sns.heatmap(mat_metric, annot=True, fmt="3.3f")
    plt.title(metric_col)
    plt.tight_layout()

    plt.subplot(nrows, ncols, 1*(ncols)+n+1)
    df_metric_ratio = (df_metric.set_index([ylabel, xlabel])["value"] / info_mat[info_mat.metric=="rand_"+metric_col].set_index([ylabel, xlabel])["value"]).reset_index()
    mat_metric_ratio = df_metric_ratio.pivot_table(index=ylabel, columns=xlabel, values="value")
    sns.heatmap(mat_metric_ratio, annot=True, fmt="3.3f")
    plt.title("$\\frac{\\mathrm{AUPRC}}{\\mathrm{Prevalence}}$")
    plt.tight_layout()

    if title is not None:
        plt.suptitle(title, y=1.05)
    if fig_path is not None:
        plt.savefig(fig_path)
    if SHOW:
        plt.show()
    else:
        plt.close()

def plot_heatmap_merge_del(lst_t_gapmerge, lst_t_eventdel, configs, lst_recall=[0.8, 0.9, 0.95], title=None, fig_path=None):
    info_mat = []
    for i, m in enumerate(lst_t_gapmerge):
        for d in lst_t_eventdel:
            res_dir = "mh210517_merged_%dh_deleted_%dh_KidneyFailure0To%dHours"%(m,d,t_window_h)
            df = interpolate_prec(get_df(res_dir, "point_est", configs, RANDOM=False), lst_recall).set_index("rec")
            df_rand = interpolate_prec(get_df(res_dir, "point_est", configs, RANDOM=True), lst_recall).set_index("rec")
            for fixed_recall in lst_recall:
                info_mat.append([m, d, df.loc[fixed_recall].prec, "Precision@Recall=%d%%"%(fixed_recall*100)])
                info_mat.append([m, d, df_rand.loc[fixed_recall].prec, "rand_Precision@Recall=%d%%"%(fixed_recall*100)])

            auc = metrics.auc(np.sort(df.index.values), df.iloc[np.argsort(df.index.values)].prec)
            auc_rand = metrics.auc(np.sort(df_rand.index.values), df_rand.iloc[np.argsort(df_rand.index.values)].prec)
            info_mat.append([m, d, auc, "AUPRC"])
            info_mat.append([m, d, auc_rand, "rand_AUPRC"])
            info_mat = pd.DataFrame(info_mat, columns=["Gap merging length", "Short event deletion length", "value", "metric"])

    ncols = len(lst_recall)+1
    nrows = 2
    ylabel = info_mat.columns[0]
    xlabel = info_mat.columns[1]
    fig = plt.figure(figsize=(6*ncols,4*nrows))
    for n, fixed_recall in enumerate(lst_recall):
        metric_col = "Precision@Recall=%d%%"%(fixed_recall*100)
        plt.subplot(nrows, ncols, 0*(ncols)+n+1)
        df_metric = info_mat[info_mat.metric==metric_col][[ylabel, xlabel,"value"]]
        mat_metric = df_metric.pivot_table(index=ylabel, columns=xlabel, values="value")        
        sns.heatmap(mat_metric, annot=True, fmt="3.3f")
        plt.title(metric_col)
        plt.tight_layout()

        plt.subplot(nrows, ncols, 1*(ncols)+n+1)
        df_metric_ratio = (df_metric.set_index([ylabel, xlabel])["value"] / info_mat[info_mat.metric=="rand_"+metric_col].set_index([ylabel, xlabel])["value"]).reset_index()
        mat_metric_ratio = df_metric_ratio.pivot_table(index=ylabel, columns=xlabel, values="value")
        sns.heatmap(mat_metric_ratio, annot=True, fmt="3.3f")
        plt.title("$\\frac{\\mathrm{Precision}_m}{\\mathrm{Precision}_r}$@Recall=%d%%"%(fixed_recall*100))
        plt.tight_layout()

    n += 1
    metric_col = "AUPRC"        
    plt.subplot(nrows, ncols, 0*(ncols)+n+1)
    df_metric = info_mat[info_mat.metric==metric_col][[ylabel, xlabel,"value"]]
    mat_metric = df_metric.pivot_table(index=ylabel, columns=xlabel, values="value")        
    sns.heatmap(mat_metric, annot=True, fmt="3.3f")
    plt.title(metric_col)
    plt.tight_layout()

    plt.subplot(nrows, ncols, 1*(ncols)+n+1)
    df_metric_ratio = (df_metric.set_index([ylabel, xlabel])["value"] / info_mat[info_mat.metric=="rand_"+metric_col].set_index([ylabel, xlabel])["value"]).reset_index()
    mat_metric_ratio = df_metric_ratio.pivot_table(index=ylabel, columns=xlabel, values="value")
    sns.heatmap(mat_metric_ratio, annot=True, fmt="3.3f")
    plt.title("$\\frac{\\mathrm{AUPRC}}{\\mathrm{Prevalence}}$")
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, y=1.05)
    if fig_path is not None:
        plt.savefig(fig_path)
    if SHOW:
        plt.show()
    else:
        plt.close()
        

    
def get_model_prc(respath, 
                  model,
                  ews_configs,
                  lst_tsplit,
                  calibrated_s=None,
                  RANDOM=False):
    print(calibrated_s)
    get_df_split = lambda s: get_df(respath, model, s, ews_configs, calibrated_s=calibrated_s, RANDOM=RANDOM).set_index('rec').rename(columns={'prec': s})

    lst_df = []
    for tsplit in lst_tsplit:
        lst_df.append(get_df_split(tsplit)[[tsplit]])
        
    lst_df = pd.concat(lst_df, axis=1).sort_index()
    lst_df = lst_df.interpolate(method='index', 
                                axis=0)
    lst_auc = [metrics.auc(lst_df.index,lst_df[tsplit]) for tsplit in lst_df.columns]
    auc_mean = np.nanmean(lst_auc)
    auc_std = np.nanstd(lst_auc)
        
    df = pd.concat([lst_df.mean(axis=1).to_frame(name='prec_mean'),
                    lst_df.std(axis=1).to_frame(name='prec_std')], axis=1)
    return df, auc_mean, auc_std

def plot_prc_curves(respath, json_file, h=5.8, w=6, fig_path=None):
    
    with open(json_file, 'r') as outfile:
        curves = json.load(outfile)    

    plt.figure(figsize=(h*cm, w*cm))
    num_solid_c = np.sum([(1-v['random']) for _,v in curves.items()])
    for i, model in enumerate(curves.keys()):

        df, auc_mean, auc_std = get_model_prc(respath, 
                                              curves[model]['model'], 
                                              curves[model]['ews_configs'],
                                              curves[model]['lst_tsplit'], 
                                              calibrated_s=curves[model]['calibrated_s'] if 'calibrated_s' in curves[model] else None,
                                              RANDOM=curves[model]['random'])
        
        if 'threshold_baseline' in curves[model] and curves[model]['threshold_baseline']:
            plt.scatter(df.index[1], 
                        df.iloc[1].prec_mean,
                        color=curves[model]['color'], 
                        marker='d', 
                        s=8,
                        label=model)
            
        else:
            plt.plot(df.index, 
                     df.prec_mean, 
                     color=curves[model]['color'],
                     linestyle=curves[model]['linestyle'],
                     label='_nolegend_' if curves[model]['random'] else model)

            plt.fill_between(df.index, 
                             df.prec_mean-df.prec_std,
                             df.prec_mean+df.prec_std,
                             color=curves[model]['color'],
                             alpha=0.3)
            # if curves[model]['random']:
            #     continue
            plt.text(0.03, 
                     (num_solid_c-1-i)*0.07, 
                     '%3.3f$\pm$%3.3f'%(auc_mean, auc_std) if len(curves[model]['lst_tsplit'])>1 else '%3.3f'%auc_mean,
                     color=curves[model]['color'],
                     horizontalalignment='left', 
                     verticalalignment='bottom')
    plt.text(0.03, 
             (num_solid_c)*0.07, 
             'AUPRC:',
             horizontalalignment='left', 
             verticalalignment='bottom')

    plt.grid(alpha=0.5)    
    plt.axis('equal')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.legend(loc=1, facecolor='white')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(json_file.replace('.json',''))
    plt.show()
