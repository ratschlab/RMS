''' Post-processes evaluation results to give
    error bars to different configurations'''

import os
import os.path
import argparse
import ipdb

import pandas as pd
import numpy as np

def execute(configs):
    df_eval_all=pd.read_csv(configs["eval_path"],sep='\t')

    unique_tasks=list(df_eval_all["task"].unique())

    out_df_tasks=[]
    out_df_auprc_mean=[]
    out_df_auprc_std=[]
    out_df_prec80_mean=[]
    out_df_prec80_std=[]
    out_df_n_patients=[]
    
    for task in unique_tasks:
        df_task=df_eval_all[df_eval_all.task==task]
        assert df_task.shape[0]==5
        auprc_mean=df_task["auprc"].mean()
        auprc_std=df_task["auprc"].std()
        prec80_mean=df_task["prec_at_80"].mean()
        prec80_std=df_task["prec_at_80"].std()
        out_df_tasks.append(task)
        out_df_auprc_mean.append(auprc_mean)
        out_df_auprc_std.append(auprc_std)
        out_df_prec80_mean.append(prec80_mean)
        out_df_prec80_std.append(prec80_std)
        out_df_n_patients.append(df_task["test_set_pids"].mean())

    out_df=pd.DataFrame({"task": out_df_tasks,
                         "n_patients": out_df_n_patients,
                         "auprc_mean": out_df_auprc_mean,
                         "auprc_std": out_df_auprc_std,
                         "prec80_mean": out_df_prec80_mean,
                         "prec80_std": out_df_prec80_std})

    out_df.to_csv(configs["output_path"],sep='\t',index=False)

    
if __name__=="__main__":

    parser=argparse.ArgumentParser()

    # Input paths
    #parser.add_argument("--eval_path",
    #                    default="../../data/evaluation/time_point_based/eval_fail_prefix_model/task_results.tsv",
    #                    help="Evaluation file with all splits")

    parser.add_argument("--eval_path",
                        default="../../data/evaluation/time_point_based/eval_fail_prefix_model_no_pharma/task_results.tsv",
                        help="Evaluation file with all splits")    

    #parser.add_argument("--eval_path",
    #                    default="../../data/evaluation/time_point_based/eval_fail_prefix_model_val/task_results.tsv",
    #                    help="Evaluation file with all splits")

    #parser.add_argument("--eval_path",
    #                    default="../../data/evaluation/time_point_based/eval_fail_prefix_model_val_no_pharma/task_results.tsv",
    #                    help="Evaluation file with all splits")

    #parser.add_argument("--eval_path",
    #                    default="../../data/evaluation/time_point_based/eval_ef_cohorts/task_results.tsv",
    #                    help="Evaluation file with all splits")            

    # Output paths
    #parser.add_argument("--output_path",
    #                    default="../../data/evaluation/time_point_based/eval_fail_prefix_model/task_results_with_error_bars.tsv",
    #                    help="Evaluation file with all splits")

    parser.add_argument("--output_path",
                        default="../../data/evaluation/time_point_based/eval_fail_prefix_model_no_pharma/task_results_with_error_bars.tsv",
                        help="Evaluation file with all splits")    

    #parser.add_argument("--output_path",
    #                    default="../../data/evaluation/time_point_based/eval_fail_prefix_model_val/task_results_with_error_bars.tsv",
    #                    help="Evaluation file with all splits")

    #parser.add_argument("--output_path",
    #                    default="../../data/evaluation/time_point_based/eval_fail_prefix_model_val_no_pharma/task_results_with_error_bars.tsv",
    #                    help="Evaluation file with all splits")

    #parser.add_argument("--output_path",
    #                    default="../../data/evaluation/time_point_based/eval_ef_cohorts/task_results_with_error_bars.tsv",
    #                    help="Evaluation file with all splits")                

    # Arguments

    configs=vars(parser.parse_args())

    execute(configs)
