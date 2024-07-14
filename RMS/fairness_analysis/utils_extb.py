def get_group_threshold(target_value, predictions, map_group_pid, cats):
    labels_group, preds_group = map_predictions_group(predictions, map_group_pid)
    thres_group = {}
    for cat in cats:
        thres_group[cat] = find_threshold_precision_recall(target_value, 'recall', labels_group[cat], 
                                                           preds_group[cat])
    return thres_group

def run_stat_test_binary(df_metrics, group_name, cats, alpha):
    with_stars = {}
    for metric in ['precision_0.8', 'precision_0.2']:
        with_stars[metric] = []
        pvalue_1 = run_mann_whitney_u(
            df_metrics, metric, 'cat', cats[0], cats[1], hyp="greater"
            ).pvalue
        pvalue_2 = run_mann_whitney_u(
            df_metrics, metric, 'cat', cats[1], cats[0], hyp="greater"
            ).pvalue
        if pvalue_1 < alpha:
            with_stars[metric].append(cats[1])
        elif pvalue_2 < alpha:
            with_stars[metric].append(cats[0])
    return with_stars

def run_stat_test_multi(df_metrics, group_name, cats, alpha):
    with_stars = {}
    for metric in ['precision_0.8', 'precision_0.2']:
        with_stars[metric] = []
        for cat in cats:
            pvalue_less = run_mann_whitney_u(
                df_metrics, metric, 'cat', cat, f"Not {cat}", hyp="less"
            ).pvalue
            if pvalue_less < alpha:
                with_stars[metric].append(cat)
    return with_stars

def run_analysis(group_name, cats, alpha):
    analysis_res = []
    map_group_pid = get_map_group_pid(df_patients, group_name, cats)
    thres_08 = get_group_threshold(0.8, predictions, map_group_pid, cats)
    thres_02 = get_group_threshold(0.2, predictions, map_group_pid, cats)
    n_analysed_samples = 0
    for it, bootstrap_pids in enumerate(bootstrap_sample_pids):
        if n_analysed_samples >= 100:
            break
        patients = df_patients.loc[bootstrap_pids]
        if not check_group_has_event(
                    patients,
                    {group_name: cats},
                    event_bounds,
                    8064,
                    0,
                ):
            continue
        n_analysed_samples += 1
        map_group_pid = get_map_group_pid(patients, group_name, cats)
        labels_group, preds_group = map_predictions_group(predictions, map_group_pid)
        
        for cat in cats:
            precision_08 = get_precision(labels_group[cat], (preds_group[cat] >= thres_08[cat]).astype(int))
            precision_02 = get_precision(labels_group[cat], (preds_group[cat] >= thres_02[cat]).astype(int))
            analysis_res.append({'precision_0.8': precision_08, 'precision_0.2': precision_02, 'group': group_name,
                                 'cat': cat, 'run': it})
        if len(cats) > 2:
            map_group_pid = get_map_not_group_pid(patients, group_name, cats)
            labels_group, preds_group = map_predictions_group(predictions, map_group_pid)
            prev_group, prev_precision, _ = compute_prevalence_groups(labels_group)
            for cat in map_group_pid.keys():
                precision_08 = get_precision(labels_group[cat], (preds_group[cat] >= global_thres_08).astype(int))
                precision_02 = get_precision(labels_group[cat], (preds_group[cat] >= global_thres_02).astype(int))
                analysis_res.append({'precision_0.8': precision_08, 'precision_0.2': precision_02, 'group': f"{group_name}_bool",
                                 'cat': cat, 'run': it})
    df_metrics = pd.DataFrame(analysis_res)
    if len(cats) == 2:
        with_stars = run_stat_test_binary(df_metrics, group_name, cats, alpha)
    else:
        with_stars = run_stat_test_multi(df_metrics, group_name, cats, alpha)
    return df_metrics, with_stars 