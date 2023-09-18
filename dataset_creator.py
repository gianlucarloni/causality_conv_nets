# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
from torch.utils import data
from PIL import Image
import os
import logging

#from settings import CONDITIONING_FEATURE

logger = logging.getLogger(__name__)



class Dataset2DSL(data.Dataset): 
    def __init__(self, csv_path, dataset_name, CONDITIONING_FEATURE, transform=None, use_label=False):
        
        """
        Parameters:
            
            - csv_path (string): percorso al file csv con le annotazioni
            - dataset_name (string): nome del dataset con cui allenare e anche folder da cui prendere i dati
            - transform (torchvision.transforms.Compose): da applicare alle immagini, eg, resize, flip, totensor, etc..
            - use_label (boolean): consider or discard y-label information    
        """
        
        self.info = pd.read_csv(csv_path)
        self.dir_path = os.path.join(os.getcwd(),"dataset_PICAI","cropped_images",dataset_name)
        self.CONDITIONING_FEATURE = CONDITIONING_FEATURE
        self.transform = transform
        self.use_label= use_label
       
    def __len__ (self):
            return len(self.info)
        
    def __getitem__(self, idx): 
            if torch.is_tensor(idx): # se idx è un tensore
                idx = idx.tolist() # lo converto in una lista
            patient = str(self.info.iloc[idx]['patient_id'])
            study = str(self.info.iloc[idx]['study_id'])
            slice_number = str(self.info.iloc[idx]['slice'])

            image_path = os.path.join(self.dir_path, f"{patient}_{study}_{slice_number}.png")
            image = Image.open(image_path)
            ##

            if self.use_label:
                
                if self.CONDITIONING_FEATURE == "aggressiveness":
                    label = str(self.info.iloc[idx]['label'])
                    if label == 'LG':
                        label = np.array(0)
                    else:
                        label = np.array(1)
                elif self.CONDITIONING_FEATURE == "no_tumour": # 1 may 2023
                    histopath_type=str(self.info.iloc[idx]['histopath_type'])
                    label = np.array(0) if (histopath_type=='' or histopath_type==None) else np.array(1)
                elif self.CONDITIONING_FEATURE == "scanner_vendor": # 3 may 2023
                    scanner_manufacturer=str(self.info.iloc[idx]['manufacturer'])
                    if scanner_manufacturer == "None":
                        print("MY ERROR: raise Stopiteration called in the dataloader")
                        raise StopIteration
                    elif scanner_manufacturer == "Philips Medical Systems":
                        label = np.array(0)
                    elif scanner_manufacturer == "SIEMENS":
                        label = np.array(1)
                    else:
                        print(f"MY ERROR: Unrecognised scanner manufacturer: {scanner_manufacturer}")
                        raise StopIteration
                elif self.CONDITIONING_FEATURE == "disease_yes_no":
                    label = str(self.info.iloc[idx]['label'])
                    if (label == 'LG' or label == 'HG'):
                        label = np.array(1)
                    else:
                        label = np.array(0)
            
            ## Applico qui eventuali trasformazioni alla immagine prima di ritornarla col getitem
            if self.transform:
                image = self.transform(image.convert("L"))

            if self.use_label:
                return image, label
            else:
                return image


        

""" class ToTensorDataset2DSL(torch.utils.data.Subset):
    
    #Dato un dataset crea un altro dataset applicando una funzione data
    #ad ognuno dei suoi valori. In questo caso converte i volumi e le label
    #in tensori da fornire in ingresso ad una rete neurale.

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        
    
        image = torch.from_numpy(np.array(image)).float().unsqueeze(dim=0)
        #image = image.permute(2,0,1) #switch channels in the correct order

        label = torch.from_numpy(label).float()
        
        return image,label

    def __len__(self):
        return len(self.dataset) """