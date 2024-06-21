''' Cluster dispatcher for respiratory endpoint generation'''

import subprocess
import os
import os.path
import sys
import argparse
import gin
import ipdb

from RMS.utils.filesystem import delete_if_exist


def execute(configs):
    ''' Computes all endpoints for all patient batches'''
    compute_script_path=configs["compute_script_path"]
    job_index=0
    mem_in_mbytes=configs["compute_mem"]
    n_cpu_cores=configs["compute_n_cores"]
    n_compute_hours=configs["compute_n_hours"]
    bad_hosts=["le-g3-003","le-g3-007"]
    is_dry_run=configs["dry_run"]

    if not is_dry_run and not configs["preserve_logs"]:
        print("Deleting previous log-files...")
        for logf in os.listdir(configs["log_dir"]):
            os.remove(os.path.join(configs["log_dir"], logf))

    if configs["reliability_analysis"]:
        random_seeds=configs["RANDOM_STATES_REL"]
    else:
        random_seeds=[configs["random_state"]]
            
    for split_key in configs["SPLITS"]:
        for batch_idx in configs["BATCH_IDXS"]:
            for rseed in random_seeds:
                print("Create {} endpoint data for batch, split {}, {} with seed {}".format(configs["endpoint"], split_key, batch_idx, rseed))
                job_name="resp_endpoint_gen_{}_{}_seed{}".format(split_key,batch_idx,rseed)
                log_stdout_file=os.path.join(configs["log_dir"], "{}.stdout".format(job_name))
                log_stderr_file=os.path.join(configs["log_dir"], "{}.stderr".format(job_name))               
                delete_if_exist(log_stdout_file)
                delete_if_exist(log_stderr_file)                
                
                cmd_line=" ".join(["sbatch", "--mem-per-cpu {}".format(mem_in_mbytes),
                                   "-n", "{}".format(n_cpu_cores),
                                   "--time", "{}:00:00".format(n_compute_hours),
                                   "--mail-type FAIL",
                                   "--exclude compute-biomed-18",
                                   "--job-name","{}".format(job_name), "-o", log_stdout_file, "-e", log_stderr_file, "--wrap",
                                   '\"python3', compute_script_path, "--run_mode CLUSTER", "--endpoint {}".format(configs["endpoint"]), "--random_state {}".format(rseed),
                                   "--split {}".format(split_key),"--batch_idx {}".format(batch_idx), "--gin_config {}".format(configs["base_gin_config"]), '\"'])
                assert(" rm " not in cmd_line)
                job_index+=1

                if is_dry_run:
                    print("CMD: {}".format(cmd_line))
                else:
                    subprocess.call([cmd_line], shell=True)

                    if configs["debug_mode"]:
                        sys.exit(0)

    print("Number of generated jobs: {}".format(job_index))    

@gin.configurable
def parse_gin_args(old_configs,gin_configs=None):
    gin_configs=gin.query_parameter("parse_gin_args.gin_configs")
    for k in old_configs.keys():
        if old_configs[k] is not None:
            gin_configs[k]=old_configs[k]
    gin.bind_parameter("parse_gin_args.gin_configs",gin_configs)
    return gin_configs

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--dry_run", default=None, action="store_true", help="Should a dry run be run, without dispatching the jobs?")
    parser.add_argument("--debug_mode", default=None, action="store_true", help="Debug mode?")
    parser.add_argument("--preserve_logs", default=None, action="store_true", help="Should logs be preserved?")
    
    #parser.add_argument("--gin_config", default="./configs/cluster_ep.gin", help="GIN config to use")
    parser.add_argument("--gin_config", default="./configs/cluster_ep_extval.gin", help="GIN config to use")
    #parser.add_argument("--gin_config", default="./configs/cluster_ep_circ_ref.gin", help="GIN config to use")

    args=parser.parse_args()
    configs=vars(args)
    gin.parse_config_file(configs["gin_config"])
    configs=parse_gin_args(configs)

    configs["BATCH_IDXS"]=list(range(configs["max_batch_idx"]))

    execute(configs)
    
