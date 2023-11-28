# -*- coding: utf-8 -*-

import pandas as pd
import os
import sklearn
import numpy as np 
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

seed=42

def get_or_create_datasetsCSVpaths(EXPERIMENT, CONDITIONING_FEATURE, csv_path, testset_size=0.2, validset_size=0.15):

    if EXPERIMENT=="prostate": #PICAI dataset
        path_tr = os.path.join(os.getcwd(),"dataset_PICAI", "csv_files",f"d_train_{CONDITIONING_FEATURE}_unfolded.csv")
        path_va = os.path.join(os.getcwd(),"dataset_PICAI","csv_files",f"d_val_{CONDITIONING_FEATURE}_unfolded.csv")
        path_te = os.path.join(os.getcwd(),"dataset_PICAI","csv_files",f"d_test_{CONDITIONING_FEATURE}_unfolded.csv")
        if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)):

            df = pd.read_csv(csv_path)
            #%% Split into training+validation and test
            study1 = df.study_id
            #labels1 = df.label
            patients1 = df.patient_id
            gs = GroupShuffleSplit(n_splits=2, test_size=testset_size, random_state=seed)
            trainval_idx, test_idx = next(gs.split(study1, groups=patients1))
            trainvalset = df.loc[trainval_idx]
            testset = df.loc[test_idx]
            #%% Split into training and validation
            study2 = trainvalset.study_id
            #labels2 = trainvalset.label
            patients2 = trainvalset.patient_id
            gs2 = GroupShuffleSplit(n_splits=2, test_size=validset_size, random_state=seed)
            train_idx, val_idx = next(gs2.split(study2, groups=patients2))
            trainset = trainvalset.reset_index().loc[train_idx]
            valset = trainvalset.reset_index().loc[val_idx]
            #%%    
            # # shuffle the DataFrame rows
            trainset = trainset.sample(frac = 1, random_state=seed)
            valset = valset.sample(frac = 1, random_state=seed)
            testset = testset.sample(frac = 1, random_state=seed)
            # Save
            trainset.to_csv(path_tr,index=False)
            valset.to_csv(path_va,index=False)
            testset.to_csv(path_te,index=False)
            print("get_or_create_datasetsCSVpaths(): created the three CSV files")
            
        else:
            print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before")
        
    elif EXPERIMENT == "procancer":
        
        desired_test_ratio=0.20
        desired_val_ratio=0.15

        path_tr = os.path.join(os.getcwd(),"dataset_procancer", "csv_files",f"d_tr_{EXPERIMENT}_{CONDITIONING_FEATURE}.csv")
        path_va = os.path.join(os.getcwd(),"dataset_procancer","csv_files",f"d_va_{EXPERIMENT}_{CONDITIONING_FEATURE}.csv")
        path_te = os.path.join(os.getcwd(),"dataset_procancer","csv_files",f"d_te_{EXPERIMENT}_{CONDITIONING_FEATURE}.csv")

        if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)):

            df = pd.read_csv(csv_path)
            #%% Split into training+validation and test
            data1 = df.data_index
            labels1 = df.groundtruth
            patients1 = df.patient_id

            # gs = GroupShuffleSplit(n_splits=2, test_size=testset_size, random_state=seed)
            # trainval_idx, test_idx = next(gs.split(series1, groups=patients1))            
            cv = StratifiedGroupKFold(n_splits=int(1/desired_test_ratio), shuffle=True)
            trainval_idx, test_idx = next(cv.split(data1, labels1, patients1))
            
            trainvalset = df.loc[trainval_idx]
            testset = df.loc[test_idx]
            #%% Split into training and validation
            data2 = trainvalset.data_index
            labels2 = trainvalset.groundtruth
            patients2 = trainvalset.patient_id

            # gs2 = GroupShuffleSplit(n_splits=2, test_size=validset_size, random_state=seed)
            # train_idx, val_idx = next(gs2.split(series2, groups=patients2))
            cv = StratifiedGroupKFold(n_splits=int(1/desired_val_ratio), shuffle=True)
            train_idx, val_idx = next(cv.split(data2, labels2, patients2))

            trainset = trainvalset.reset_index().loc[train_idx]
            valset = trainvalset.reset_index().loc[val_idx]
            #%%    
            # # shuffle the DataFrame rows
            trainset = trainset.sample(frac = 1, random_state=seed)
            valset = valset.sample(frac = 1, random_state=seed)
            testset = testset.sample(frac = 1, random_state=seed)
            # Save
            trainset.to_csv(path_tr,index=False)
            print(f"Saved TRAINSET csv file, with proportion of events: {labels2.reset_index().loc[train_idx].mean()}")

            valset.to_csv(path_va,index=False)
            print(f"Saved VALSET csv file, with proportion of events: {labels2.reset_index().loc[val_idx].mean()}")

            testset.to_csv(path_te,index=False)
            print(f"Saved TESTSET csv file, with proportion of events: {labels1.reset_index().loc[test_idx].mean()}")            
        else:
            print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before")
    



    if EXPERIMENT=="breakhis": #breakhistopathology dataset
        desired_test_ratio=0.20
        desired_val_ratio=0.15

        path_tr = os.path.join(os.getcwd(),"dataset_breakhis", "csv_files",f"d_train_{CONDITIONING_FEATURE}_unfolded.csv")
        path_va = os.path.join(os.getcwd(),"dataset_breakhis","csv_files",f"d_val_{CONDITIONING_FEATURE}_unfolded.csv")
        path_te = os.path.join(os.getcwd(),"dataset_breakhis","csv_files",f"d_test_{CONDITIONING_FEATURE}_unfolded.csv")
        if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)):

            df = pd.read_csv(csv_path)
            
            #%% Split into training+validation and test
            # xs = df.image
            xs = df.index

            ys = df.binary_target

            trainval_idx, test_idx, _, _,= sklearn.model_selection.train_test_split(xs, ys,
                                                    test_size=desired_test_ratio,
                                                    random_state=seed,
                                                    stratify=ys)

            trainvalset = df.loc[trainval_idx] #temprorary

            testset = df.loc[test_idx] #final

            #%% Split into training and validation
            # xs = trainvalset.image
            xs = trainvalset.index

            ys = trainvalset.binary_target
            train_idx, val_idx, _, _ = sklearn.model_selection.train_test_split(xs, ys,
                                                    test_size=desired_val_ratio, #real validation
                                                    random_state=seed,
                                                    stratify=ys)

            trainset = trainvalset.loc[train_idx] #final
            valset = trainvalset.loc[val_idx] #final
            # trainset = trainvalset.reset_index().loc[train_idx] #final
            # valset = trainvalset.reset_index().loc[val_idx] #final
            #%%    
            # # shuffle the DataFrame rows
            trainset = trainset.sample(frac = 1, random_state=seed)
            valset = valset.sample(frac = 1, random_state=seed)
            testset = testset.sample(frac = 1, random_state=seed)
            # Save
            trainset.to_csv(path_tr,index=False)
            valset.to_csv(path_va,index=False)
            testset.to_csv(path_te,index=False)
            print("get_or_create_datasetsCSVpaths(): created the three CSV files")
            
        else:
            print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before")
        


    return path_tr, path_va, path_te
