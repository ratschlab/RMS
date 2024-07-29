import pickle
from pathlib import Path
import pandas as pd
from famews.fairness_check.utils.helper_groups import build_age_group
from famews.fairness_check.utils.helper_endpoints import get_event_bounds
from famews.data.utils import MarkedDataset
import re

def preprocess_preds(path_split: Path, path_chunk: Path, root_pred_dir:Path, task: str, temp_id: int=1) -> dict: 
    """Get predictions in expected format for the fairness analysis

    Parameters
    ----------
    path_split : Path
        _description_
    path_chunk : Path
        _description_
    pred_dir : Path
        _description_
    temp_id : int, optional
        _description_, by default 1

    Returns
    -------
    dict
        Predictions for each patient
    """
    with open(path_split, 'rb') as f:
        split_info = pickle.load(f)
    test_pids = split_info[f'temporal_{temp_id}']['test']
    with open(path_chunk, 'rb') as f:
        chunk_info = pickle.load(f)
    test_chunks = []
    for pid in test_pids:
        chunk = chunk_info['pid_to_chunk'][pid]
        if chunk not in test_chunks:
            test_chunks.append(chunk)
    predictions = {}
    pred_dir = root_pred_dir / f"temporal_{temp_id}" / task
    for chunk_id in test_chunks:
        part = pred_dir / f"batch_{chunk_id}.h5"
        table_preds = tables.open_file(part, "r").root
        for key in table_preds._v_children:
            if int(key[1:]) not in test_pids:
                continue
            df_pred = pd.read_hdf(part, key=key).set_index('RelDatetime').sort_index()
            pid = int(df_pred.iloc[0]['PatientID'])
            predictions[pid] = (df_pred['PredScore'].values, df_pred['TrueLabel'].values)
    return predictions

def remove_only_nan_predictions(predictions_w_nan: dict) -> dict:
    predictions = {}
    for pid, (pred, label) in predictions_w_nan.items():
        if sum(~np.isnan(label)):
            predictions[pid] = (pred, label)
    return predictions

def get_mean_rf_preds(path_split: Path, path_chunk: Path, root_pred_dir:Path)-> dict:
    list_predictions = []
    for i in range(1,6):
        list_predictions.append(preprocess_preds(path_split, path_chunk, root_pred_dir, "Label_WorseStateFromZeroOrOne0To24Hours_internal_rmsRF_lightgbm", i))
    return {
            pid: (
                np.mean([pred[pid][0] for pred in list_predictions], axis=0),
                list_predictions[0][pid][1],
            )
            for pid in list_predictions[0].keys()
        }

def get_mean_extf_preds(path_split: Path, path_chunk: Path, root_pred_dir:Path)-> dict:
    list_predictions = []
    for i in range(1,6):
        list_predictions.append(remove_only_nan_predictions(preprocess_preds(path_split, path_chunk, root_pred_dir, "Label_ExtubationFailureSimple_internal_rmsEF_lightgbm", i)))
    return {
            pid: (
                np.mean([pred[pid][0] for pred in list_predictions], axis=0),
                list_predictions[0][pid][1],
            )
            for pid in list_predictions[0].keys()
        }

BATCH_HDF_PATTERN = re.compile("batch_([0-9]+).h5")
def get_event_bounds_resp(endpoint_dir:Path)-> dict:
    endpt_hdf_file = next(endpoint_dir.rglob("*.h5"))
    endpt_hdf_directory = endpt_hdf_file.parent
    endpoint_dataset = MarkedDataset(endpt_hdf_directory, part_re=BATCH_HDF_PATTERN, force=False)
    parts_list = endpoint_dataset.list_parts()
    keep_columns = ['endpoint_status', 'RelDatetime', 'PatientID']
    event_bounds = {}
    for part in parts_list:
        df_endpt = pd.read_hdf(part, columns=keep_columns)
        df_endpt['InEvent'] = df_endpt['endpoint_status'].isin(['event_2', 'event_3'])
        df_endpt = df_endpt.drop(columns=['endpoint_status'])
        grouped_df_endpt = df_endpt.groupby('PatientID')
        for pid, df in grouped_df_endpt:
            event_bounds[pid] = get_event_bounds(df, 'RelDatetime')
    return event_bounds

def get_event_bounds_ext(predictions: dict)->dict:
    event_bounds = {pid: [] for pid in predictions.keys()}
    for pid, (pred, label) in predictions.items():
        start = None
        in_event = False
        for i, t in enumerate(label):
            if t==1 and not in_event:
                start = i
                in_event = True
            elif t!=1 and in_event:
                event_bounds[pid].append((start, i))
                in_event = False
        if in_event:
            event_bounds[pid].append((start, len(label)))
    return event_bounds

MAP_APACHE_PAT = {1: 'Cardiovascular', 2: 'Respiratory', 3: 'Gastrointestinal', 4: 'Neurological', 6: 'Other',
                  7: 'Trauma', 8: 'Other', 9: 'Other', 11: 'Cardiovascular', 12: 'Respiratory', 
                  13: 'Gastrointestinal', 14: 'Neurological', 15: 'Trauma', 16: 'Other', 17: 'Other'}

def build_patient_groups(path_static_patients: Path)-> pd.DataFrame
    df_patients = pd.read_hdf(PATH_STATIC_PATIENTS).set_index('PatientID')
    df_patients = df_patients.rename(columns={'Sex': 'sex'})
    df_patients["age_group"] = df_patients["Age"].apply(build_age_group)
    df_patients['APACHE_group'] = df_patients['APACHEPatGroup'].dropna().apply(lambda g: MAP_APACHE_PAT[g])
    return df_patients