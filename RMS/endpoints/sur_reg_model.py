''' Fit a surrogate regression model to estimate the current Pa2 from
    context information'''

import argparse
import os
import os.path
import glob
import ipdb
import pickle
import collections

import numpy as np
import sklearn.metrics as skmetrics
import sklearn.linear_model as sklm
import sklearn.neural_network as sknn
import sklearn.svm as sksvm
import sklearn.preprocessing as skpproc


def weight_fn(y_arr,configs=None,produce_trace=False):
    ''' Returns the weighting function of samples giving the true Y-array'''
    out_arr=configs["sample_weight_base"]+configs["sample_weight_scale"]/(1+np.exp(0.025*(y_arr-110)))
    all_weight=np.sum(out_arr)
    
    if produce_trace:
        bins=[(0,50),(50,75),(75,100),(100,125),(125,150),(150,200),
              (200,250),(250,300),(300,350),(350,400),(400,450),
              (450,500)]
        for bin_desc in bins:
            bin_sweights=out_arr[(y_arr>=bin_desc[0]) & (y_arr<=bin_desc[1])]
            print("PaO2 bin: [{},{}], Cumulative weight: {:.3f}, rel: {:.3f}".format(bin_desc[0],bin_desc[1],np.sum(bin_sweights),
                                                                                     np.sum(bin_sweights)/all_weight))
    
    return out_arr

def ellis(x):
    x=x/100
    x[x==1]=0.999
    exp_base = (11700/((1/x)-1))
    exp_sqrbracket = np.sqrt(pow(50,3)+(exp_base**2))
    exp_first = np.cbrt(exp_base + exp_sqrbracket)
    exp_second = np.cbrt(exp_base - exp_sqrbracket)
    exp_full = exp_first + exp_second
    return exp_full


def f1_score(pf_test,pf_test_pred):
    fp_cnt=np.sum((pf_test>=200) & (pf_test_pred<200))
    tp_cnt=np.sum((pf_test<200) & (pf_test_pred<200))
    tn_cnt=np.sum((pf_test>=200) & (pf_test_pred>=200))
    fn_cnt=np.sum((pf_test<200) & (pf_test_pred>=200))
    recall=tp_cnt/(tp_cnt+fn_cnt)
    prec=tp_cnt/(tp_cnt+fp_cnt)
    f1score=2*prec*recall/(prec+recall)
    return f1score


def execute(configs):
    reg_data=sorted(glob.glob(os.path.join(configs["data_path"],"reg_data*.pickle")),key=lambda fpath: int(fpath.split("/")[-1].split("_")[-1][:-7]))
    X_collect=[]
    y_collect=[]
    fio2_collect=[]
    for rix,rf in enumerate(reg_data):
        #print("File: {}/{}".format(rix+1,len(reg_data)))
        with open(rf,'rb') as fp:
            obj=pickle.load(fp)
            X_collect.extend(obj["X_reg"])
            y_collect.extend(obj["y_reg"])
            fio2_collect.extend(obj["fio2_reg"])
        #print("Batch: {}, Cum. patients: {}".format(int(rf.split("/")[-1].split("_")[-1][:-7]), len(X_collect)))
        
    X_train=np.vstack(X_collect[:int(0.5*len(X_collect))])
    y_train=np.concatenate(y_collect[:int(0.5*len(y_collect))])

    scaler=skpproc.StandardScaler()

    # Generate poly

    if configs["use_poly_features_base"]:
        poly=skpproc.PolynomialFeatures(degree=3)
        X_train=poly.fit_transform(X_train)
        
    X_train_std=scaler.fit_transform(X_train)
    
    X_val=np.vstack(X_collect[int(0.5*len(X_collect)):int(0.7*len(X_collect))])

    if configs["use_poly_features_base"]:
        X_val=poly.transform(X_val)
    
    X_val_std=scaler.transform(X_val)
    y_val=np.concatenate(y_collect[int(0.5*len(y_collect)):int(0.7*len(y_collect))])
    fio2_val=np.concatenate(fio2_collect[int(0.5*len(X_collect)):int(0.7*len(X_collect))])

    # First test set to observe the mistakes on, collect by patient
    X_test_1=X_collect[int(0.7*len(X_collect)):int(0.85*len(X_collect))]

    if configs["use_poly_features_base"]:
        X_test_1=list(map(lambda X_mat: poly.transform(X_mat),X_test_1))
    
    X_test_std_1=list(map(lambda X_mat: scaler.transform(X_mat), X_test_1))
    y_test_1=y_collect[int(0.7*len(X_collect)):int(0.85*len(X_collect))]
    fio2_test_1=fio2_collect[int(0.7*len(X_collect)):int(0.85*len(X_collect))]

    print("Test 2 index: {}".format(int(0.85*len(X_collect))))
    
    # Second test set to apply the final model on, by patient
    X_test_2=X_collect[int(0.85*len(X_collect)):]

    if configs["use_poly_features_base"]:
        X_test_2=list(map(lambda X_mat: poly.transform(X_mat),X_test_2))
    
    X_test_std_2=list(map(lambda X_mat: scaler.transform(X_mat), X_test_2))
    y_test_2=y_collect[int(0.85*len(X_collect)):]
    fio2_test_2=fio2_collect[int(0.85*len(X_collect)):]

    alpha_cands=configs["ALPHA_CANDS"]
    best_alpha=None
    best_score=np.inf
    
    print("Training base model...")
    for alpha in alpha_cands:
        print("Testing HP: {}".format(alpha))
        model_cand=sklm.SGDRegressor(alpha=alpha,random_state=2021,penalty=configs["reg_type"],
                                     loss=configs["loss_type"],learning_rate=configs["learning_rate"],
                                     early_stopping=configs["early_stopping"])

        if configs["use_weighted_model"]:
            sample_weights=weight_fn(y_train,configs=configs)
            model_cand.fit(X_train_std,y_train,sample_weight=sample_weights)
        else:
            model_cand.fit(X_train_std,y_train)
            
        y_val_pred=model_cand.predict(X_val_std)
        mae=np.median(np.absolute(y_val-y_val_pred))
        if mae<best_score:
            best_score=mae
            best_alpha=alpha

    print("Best alpha: {}".format(best_alpha))
            
    best_model=sklm.SGDRegressor(alpha=best_alpha,random_state=2021,penalty=configs["reg_type"],
                                 loss=configs["loss_type"],learning_rate=configs["learning_rate"],
                                 early_stopping=configs["early_stopping"])

    if configs["use_weighted_model"]:
        sample_weights=weight_fn(y_train,configs=configs,produce_trace=True)
        best_model.fit(X_train_std,y_train,sample_weight=sample_weights)
    else:
        best_model.fit(X_train_std,y_train)        

    X_meta_collect=[]
    y_meta_collect=[]
    
    # Create predictions on 1st set for each patient
    for X_pat_unstd, X_pat,y_pat in zip(X_test_1, X_test_std_1,y_test_1):
        mistake_queue=collections.deque([0]*configs["history_len"],maxlen=configs["history_len"])
        y_pred_pat=best_model.predict(X_pat)

        if configs["use_only_ellis"]:
            y_pred_pat=ellis(X_pat_unstd[:,1])
        
        queue_cnt=0
        for pix in range(len(y_pred_pat)):

            if queue_cnt>=configs["usage_offset"]:
                X_meta_collect.append(np.concatenate([X_pat_unstd[pix,:], np.array([y_pred_pat[pix]]+list(mistake_queue))]))
                y_meta_collect.append(y_pat[pix])
                
            bias=y_pred_pat[pix]-y_pat[pix]

            if not configs["ignore_first_abga_mistake"] or pix>0:
                mistake_queue.append(bias)
                queue_cnt+=1

    X_meta=np.vstack(X_meta_collect)
    y_meta=np.array(y_meta_collect)

    # Now split in proportion 75:25 % to find the best meta-model
    X_meta_train=X_meta[:int(X_meta.shape[0]*0.75),:]
    X_meta_val=X_meta[int(X_meta.shape[0]*0.75):,:]
    y_meta_train=y_meta[:int(X_meta.shape[0]*0.75)]
    y_meta_val=y_meta[int(X_meta.shape[0]*0.75):]

    scaler_meta=skpproc.StandardScaler()

    if configs["use_poly_features_meta"]:
        poly_meta=skpproc.PolynomialFeatures()
        X_meta_train=poly_meta.fit_transform(X_meta_train)
        X_meta_val=poly_meta.transform(X_meta_val)
    
    X_meta_train_std=scaler_meta.fit_transform(X_meta_train)
    X_meta_val_std=scaler_meta.transform(X_meta_val)

    best_score=np.inf
    best_alpha=None
    print("Training meta model")
    for alpha in alpha_cands:
        print("Testing HP: {}".format(alpha))
        model_cand=sklm.SGDRegressor(alpha=alpha,random_state=2021,penalty=configs["reg_type"],loss=configs["loss_type"],
                                     learning_rate=configs["learning_rate"], early_stopping=configs["early_stopping"])

        if configs["use_weighted_model"]:
            sample_weights=weight_fn(y_meta_train,configs=configs)
            model_cand.fit(X_meta_train_std,y_meta_train,sample_weight=sample_weights)
        else:
            model_cand.fit(X_meta_train_std,y_meta_train)
            
        y_val_pred=model_cand.predict(X_meta_val_std)
        mae=np.median(np.absolute(y_meta_val-y_val_pred))
        if mae<best_score:
            best_score=mae
            best_alpha=alpha

    print("Best alpha: {}".format(best_alpha))
            
    meta_model=sklm.SGDRegressor(alpha=best_alpha,random_state=2021,penalty=configs["reg_type"],loss=configs["loss_type"],
                                 learning_rate=configs["learning_rate"], early_stopping=configs["early_stopping"])

    if configs["use_weighted_model"]:
        sample_weights=weight_fn(y_meta_train,configs=configs)
        meta_model.fit(X_meta_train_std,y_meta_train,sample_weight=sample_weights)
    else:
        meta_model.fit(X_meta_train_std,y_meta_train)        

    # Generate predictions on test set by using both models

    local_pred_reg_medians=[]
    local_pred_online_medians=[]

    local_pred_reg_above_5=[]
    local_pred_online_above_5=[]
    
    print("Generating predictions on the second test set")
    y_test_pred_collect=[]
    y_test_pred_base_collect=[]
    for X_pat,X_pat_unstd,y_pat in zip(X_test_std_2,X_test_2,y_test_2):
        mistake_queue=collections.deque([0]*configs["history_len"],maxlen=configs["history_len"])

        if configs["use_only_ellis"]:
            y_pred_base=ellis(X_pat_unstd[:,1])
        else:
            y_pred_base=best_model.predict(X_pat) 
        
        queue_cnt=0

        # Local array collecting predictions and baseline
        local_pred_online=[]
        local_pred_reg=[]
        
        for pix in range(len(y_pred_base)):
            y_pred_pre=float(y_pred_base[pix])
            x_inst=np.concatenate([X_pat_unstd[pix,:], np.array([y_pred_base[pix]]+list(mistake_queue))])
            x_inst=x_inst.reshape((1,len(x_inst)))

            if configs["use_poly_features_meta"]:
                x_inst=poly_meta.transform(x_inst)
            
            x_inst_std=scaler_meta.transform(x_inst)
            y_final=float(meta_model.predict(x_inst_std))

            # Last offset baseline
            if configs["use_last_offset"]:
                y_test_pred_collect.append(y_pred_pre+list(mistake_queue)[-1])
                local_pred_online.append(y_pred_pre+list(mistake_queue)[-1])

            # Use corrected estimate
            elif queue_cnt>=configs["usage_offset"]:
                y_test_pred_collect.append(y_final)
                local_pred_online.append(float(y_final)-float(y_pat[pix]))
                local_pred_online_above_5.append(float(y_final)-float(y_pat[pix]))
                local_pred_reg_above_5.append(float(y_pred_pre)-float(y_pat[pix]))

            # Use original estimate from the base model
            else:
                y_test_pred_collect.append(y_pred_pre)
                local_pred_online.append(float(y_pred_pre)-float(y_pat[pix]))

            if not configs["ignore_first_abga_mistake"] or pix>0:
                mistake_queue.append(y_pred_base[pix]-y_pat[pix])
            
            local_pred_reg.append(float(y_pred_pre)-float(y_pat[pix]))
            queue_cnt+=1

        if len(local_pred_reg)>0:
            local_pred_reg_medians.append(np.median(local_pred_reg))
            local_pred_online_medians.append(np.median(local_pred_online))
            
        y_test_pred_base_collect.append(y_pred_base)
    y_test_pred_2=np.array(y_test_pred_collect).flatten()
    y_test_pred_reg_2=np.concatenate(y_test_pred_base_collect)
    y_test_baseline_2=np.vstack(X_test_2)[:,5]
    y_test_2=np.concatenate(y_test_2)
    fio2_test_2=np.concatenate(fio2_test_2)

    pf_test=y_test_2/fio2_test_2
    pf_test_pred=y_test_pred_2/fio2_test_2
    pf_test_pred_reg=y_test_pred_reg_2/fio2_test_2    
    pf_test_baseline=y_test_baseline_2/fio2_test_2

    result_dict={"pao2_true": y_test_2, "pao2_model_online": y_test_pred_2, "pao2_model_reg": y_test_pred_reg_2,
                 "pf_true": pf_test, "pf_model_online": pf_test_pred, "pf_model_reg": pf_test_pred_reg,
                 "pao2_online_medians": np.array(local_pred_online_medians), "pao2_reg_medians": np.array(local_pred_reg_medians),
                 "pao2_reg_above_5": np.array(local_pred_reg_above_5), "pao2_online_above_5": np.array(local_pred_online_above_5)}

    model_dict={"reg_base_model": best_model, "meta_model": meta_model, "base_scaler": scaler, "meta_scaler": scaler_meta}
    

    abs_error_reg=np.absolute(y_test_pred_reg_2-y_test_2)

    for model_key,preds in [("corrected",pf_test_pred),("pop",pf_test_pred_reg),("baseline", pf_test_baseline)]:
        fp_cnt=np.sum((pf_test>=200) & (preds<200))
        tp_cnt=np.sum((pf_test<200) & (preds<200))
        tn_cnt=np.sum((pf_test>=200) & (preds>=200))
        fn_cnt=np.sum((pf_test<200) & (preds>=200))
        recall=tp_cnt/(tp_cnt+fn_cnt)
        prec=tp_cnt/(tp_cnt+fp_cnt)
        spec=tn_cnt/(tn_cnt+fp_cnt)
        f1score=2*prec*recall/(prec+recall)
        print("{}: PF ratio Recall: {:.3f}".format(model_key,recall))
        print("{}: PF ratio Precision: {:.3f}".format(model_key,prec))
        print("{}: PF ratio Spec: {:.3f}".format(model_key,spec))
        print("{}: PF ratio F1 score: {:.3f}".format(model_key,f1score))

    abs_error=np.absolute(y_test_pred_2-y_test_2)    
    for rangepair in [(0,50),(50,100),(100,150),(150,np.inf)]:
        print("Evaluation range: {}".format(rangepair))
        abs_error_meta=abs_error[(y_test_2>=rangepair[0])&(y_test_2<=rangepair[1])]
        print("Range: [25: {:.3f}, 50: {:.3f}, 75: {:.3f}, Mean: {:.3f}]".format(np.percentile(abs_error_meta,25), np.percentile(abs_error_meta,50), np.percentile(abs_error_meta,75), np.mean(abs_error_meta)))

    print("Overall: [25: {:.3f}, 50: {:.3f}, 75: {:.3f}, Mean: {:.3f}]".format(np.percentile(abs_error,25), np.percentile(abs_error,50), np.percentile(abs_error,75), np.mean(abs_error)))
    
    # with open(os.path.join(configs["eval_path"], "reg_results.pickle"),'wb') as fp:
    #     pickle.dump(result_dict,fp)

    # with open(os.path.join(configs["eval_path"], "reg_models.pickle"), 'wb') as fp:
    #     pickle.dump(model_dict,fp)
    
if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    parser.add_argument("--data_path", default="/cluster/work/grlab/clinical/hirid2/research/3c_endpoints_resp/endpoints_210112/point_est",help="Regression data")

    # Output paths
    parser.add_argument("--eval_path", default="/cluster/work/grlab/clinical/hirid2/research/10c_results_resp/surrogate_pao2")

    # Arguments
    parser.add_argument("--history_len", type=int, default=10, help="History of mistakes to remember")
    parser.add_argument("--use_last_offset", default=False, help="Use last offset baseline",action="store_true")
    parser.add_argument("--reg_type", default="l2", help="Regularizer to use")
    parser.add_argument("--loss_type", default="huber", help="Loss to use")
    parser.add_argument("--learning_rate", default="adaptive", help="Learning rate schedule to use")
    parser.add_argument("--early_stopping", default=False, action="store_true", help="Should val. early stopping be used?")
    parser.add_argument("--use_poly_features_base", default=True, action="store_true")
    parser.add_argument("--use_poly_features_meta", default=False, action="store_true")
    parser.add_argument("--use_only_ellis", default=False, action="store_true")

    parser.add_argument("--ignore_first_abga_mistake", default=False, action="store_true", help="Ignore first ABGA mistake in cor. model")

    parser.add_argument("--use_weighted_model", default=True, action="store_true", help="Weighted model?")

    # Sample weight
    parser.add_argument("--sample_weight_base", default=10, type=int, help="Constant base offset")
    parser.add_argument("--sample_weight_scale", default=100, type=int, help="Max. weight scale")

    parser.add_argument("--usage_offset", default=0, type=int, help="Minimum number of elements in the queue to use the model")

    configs=vars(parser.parse_args())

    configs["ALPHA_CANDS"]=[1.0,0.1,0.01,0.001,0.0001]
    
    execute(configs)
