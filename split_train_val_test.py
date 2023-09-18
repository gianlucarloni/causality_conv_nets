# -*- coding: utf-8 -*-

import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit

def get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE, csv_path, testset_size=0.2, validset_size=0.15):
    path_tr = os.path.join(os.getcwd(),"dataset_PICAI", "csv_files",f"d_train_{CONDITIONING_FEATURE}_unfolded.csv")
    path_va = os.path.join(os.getcwd(),"dataset_PICAI","csv_files",f"d_val_{CONDITIONING_FEATURE}_unfolded.csv")
    path_te = os.path.join(os.getcwd(),"dataset_PICAI","csv_files",f"d_test_{CONDITIONING_FEATURE}_unfolded.csv")

    if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)):

        df = pd.read_csv(csv_path)
        #%% Split into training+validation and test
        study1 = df.study_id
        #labels1 = df.label
        patients1 = df.patient_id
        gs = GroupShuffleSplit(n_splits=2, test_size=testset_size, random_state=0)
        trainval_idx, test_idx = next(gs.split(study1, groups=patients1))
        trainvalset = df.loc[trainval_idx]
        testset = df.loc[test_idx]
        #%% Split into training and validation
        study2 = trainvalset.study_id
        #labels2 = trainvalset.label
        patients2 = trainvalset.patient_id
        gs2 = GroupShuffleSplit(n_splits=2, test_size=validset_size, random_state=0)
        train_idx, val_idx = next(gs2.split(study2, groups=patients2))
        trainset = trainvalset.reset_index().loc[train_idx]
        valset = trainvalset.reset_index().loc[val_idx]
        #%% Save        
        trainset.to_csv(path_tr,index=False)
        valset.to_csv(path_va,index=False)
        testset.to_csv(path_te,index=False)
        print("get_or_create_datasetsCSVpaths(): created the three CSV files")
        
    else:
        print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before")
    
    return path_tr, path_va, path_te
