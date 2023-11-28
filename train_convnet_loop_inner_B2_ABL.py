#%% IMPORT
import os
import argparse
from ast import literal_eval

#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-6772vh08 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
from torchvision.transforms import Compose
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.optim.lr_scheduler as lr_scheduler
import glob


## When running python jobs using Docker in a cluster system (docker run ...), we typically need to specify the --user tag in the command.
## For instance, --user $(id -u):$(id -g) will associate the right user group, so that our container is always identified as ours.
## Unfortunately, by doing so, we'd loose the 'root' priviledge that is typically associated with a default docker container.
## As a result, we'd loose the ability to import pretrained weights of the popular computer vision models, since we can no longer access the so called ./cache directory.
# To overcome this issue, we can create a custom folder - say 'pretrained_models' in our working dir, and set that as the Torch Hub and Torch Home (see below).
# The first step would be to run everything without the --user tag on the container, thus using default root priviledge, and make torch download the
# weights into our created 'pretrained_models' dir. Then, for every other experiment we'd need to run, we can set the --user tag in the container
# as requested by some clsuter system admins, and the script will load the pretrained weights that we saved at the previous step.
mydir=os.path.join(os.getcwd(), 'pretrained_models') ## these are the regular ImageNet-trained CV models, like resnets weights.
torch.hub.set_dir(mydir)
os.environ['TORCH_HOME']=mydir

## If you need to debug NaN and None values throughout your script, forward(), and back-prop.. setting this flag to True can be useful
# torch.autograd.set_detect_anomaly(True)
##

print()
print("###### ABLATION STUDIES ######")
print()

parser = argparse.ArgumentParser(description="Training script for a convnetm with ABLATION network.")
parser.add_argument("--number_of_gpus",type=int,default=1,help="The number of GPUs you intend to use")
parser.add_argument("--gpus_ids",type=str,default="0",help="The comma separated list of integers representing the id of requested GPUs - such as '0,1'")
#
parser.add_argument("--SEED",type=int,default=42,help="fix seed to set reproducibility, eg, seed=42")
parser.add_argument("--EXPERIMENT",type=str,default="breakhis",help="prostate, breakhis")
parser.add_argument("--CONDITIONING_FEATURE",type=str,default="aggressiveness",help="for imagenette is imagenette, for prostate can be (aggressiveness, no_tumour, scanner_vendor, disease_yes_no)")
parser.add_argument("--IMAGE_SIZE",type=int,default=128,help="imagenette 64, prostate 128")
parser.add_argument("--BATCH_SIZE_TRAIN",type=int,default=16,help="eg, 200 images")
parser.add_argument("--BATCH_SIZE_VALID",type=int,default=1,help=" 1 image")
parser.add_argument("--BATCH_SIZE_TEST",type=int,default=1,help=" 1 image")
parser.add_argument("--NUMBER_OF_EPOCHS",type=int,default=200,help="eg, 100, eventually early stopped")
parser.add_argument("--MODEL_TYPE",type=str,default="resnet18",help="eg, EqualCNN, alexnet, SimpleCNN, EqualCNN, BaseCNN, LightCNN, resnet18, resnet34, resnet50, resnet101, ")
parser.add_argument("--LR",type=str,default="[0.01,0.001]",help="list of str of floats, eg, [0.0003, 0.001]")
parser.add_argument("--WD",type=str,default="[0.01,0.01,0.001]",help="list of str of floats, eg, [0.0003, 0.001]")
parser.add_argument("--CAUSALITY_AWARENESS_METHOD", type=str, default="[None, 'max', 'lehmer']", help="[None, 'max', 'lehmer']")
parser.add_argument("--LEHMER_PARAM", type=str, default="[-2,-1,0,1,2]",help="if using Lehmer mean, which power utilize among: [-100,-1,0,1,2,100]")
parser.add_argument("--CAUSALITY_SETTING", type=str, default="['mulcat','mulcatbool']",help="if CA, which setting to use, eg: ['cat','mulcat']")
parser.add_argument("--MULCAT_CAUSES_OR_EFFECTS", type=str, default="['causes','effects']", help="if CA, which one to use for causality factors computation: ['causes','effects']")
parser.add_argument("--which_resnext_to_use",type=str, choices=["tiny","base"])
parser.add_argument("--is_pretrained",type=str, choices=["True","False"], default="False")

args = parser.parse_args()

###repoducibility:#############
SEED = args.SEED
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED) 

#model
model_type = args.MODEL_TYPE
which_resnext_to_use=""
if model_type=="resnext":
    which_resnext_to_use=args.which_resnext_to_use

#causality awareness and related settings
causality_awareness_method = literal_eval(args.CAUSALITY_AWARENESS_METHOD) #[None, 'max', 'lehmer']
LEHMER_PARAM = literal_eval(args.LEHMER_PARAM)  #"[-100,-2,-1,0,1,100]"
CAUSALITY_SETTING = literal_eval(args.CAUSALITY_SETTING) #['cat','mulcat','mulcatbool']
MULCAT_CAUSES_OR_EFFECTS = literal_eval(args.MULCAT_CAUSES_OR_EFFECTS) #['causes','effects']

#define some settings about data, paths, training params, etc.
image_size = args.IMAGE_SIZE
batch_size_train = args.BATCH_SIZE_TRAIN
batch_size_valid = args.BATCH_SIZE_VALID
batch_size_test = args.BATCH_SIZE_TEST
epochs = args.NUMBER_OF_EPOCHS
LR = literal_eval(args.LR) #list of floats, "[0.001,0.0003]"
wd = literal_eval(args.WD)

#some other settings
loss_type="CrossEntropyLoss"
is_pretrained = args.is_pretrained
if is_pretrained=="False":
    is_pretrained=False
elif is_pretrained=="True":
    is_pretrained=True
print(f"is_pretrained: {is_pretrained}")
is_feature_extractor = False


csv_path=""
train_root_path = val_root_path = test_root_path = ""

if args.EXPERIMENT == "prostate":
    dataset_name=""
    CONDITIONING_FEATURE = args.CONDITIONING_FEATURE
    channels = 1
    num_classes = 2
    if CONDITIONING_FEATURE == "aggressiveness": # lesion aggressiveness labels: LG and HG
        dataset_name = "UNFOLDED_DATASET_5_LOW_RESOLUTION_NORMALIZED_GUIDED_CROP_GUIDED_SLICE_SELECTION"
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","cs_les_unfolded.csv") ## versione nuova, con train valid e test tutti uniti
    elif CONDITIONING_FEATURE == "disease_yes_no": 
        dataset_name = "UNFOLDED_DATASET_DISEASE_YES_NO"
        #csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","unfolded_disease_YesNo.csv") #TODO: fonde i csv del caso "aggressiveness" e "no_tumour", ma da questo ultimo elimino tutte le righe di pazienti che erano stati mandati a biopsia ma che poi si sono rivelati non essere tumorali, cosi da avere le due classi ben distinte: i pazienti che non hanno fatto manco la biopsia e i pazienti che hanno biopsia clinically significant
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","unfolded_disease_YesNo_balanced.csv") #TODO: la versione bilanciata del precedente: per farlo, ho eliminato (frai pazienti senza tumore) quelli che aveano PSA piu alto (e in seconda chiave di ordinamento quelli che avevano il Volume piu alto), e poi ho riordinato le righe questa volta per pazienteID, quindi non sono stratificate come prima (dove aveo prima i tumorali e poi tutti i non tumorali)

elif args.EXPERIMENT == "breakhis": #TODO 31 oct 2023
    dataset_name=""
    CONDITIONING_FEATURE = args.CONDITIONING_FEATURE
    channels = 3
    num_classes = 2
    if CONDITIONING_FEATURE == "aggressiveness": # 
        csv_path = os.path.join(os.getcwd(),"dataset_breakhis","csv_files","breakhis_metadata_400X.csv") ##
else:
    raise ValueError
print(f"Dataset_name: {dataset_name}\n  csv_path: {csv_path}")

###############################

NUM_GPUS = args.number_of_gpus #1 #2 TODO
list_of_GPU_ids = list(args.gpus_ids)
list_of_GPU_ids = list(filter((",").__ne__, list_of_GPU_ids))


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False    
    def get_patience_and_minDelta(self):
        return (self.patience, self.min_delta)



def main(rank, world_size, causality_awareness, learning_rate, weight_decay, causality_method=None, lehmer_param=None, causality_setting="cat",mulcat_causes_or_effects="causes"):

    print(torch.cuda.is_available()) 
    os.environ['CUDA_VISIBLE_DEVICES'] = list_of_GPU_ids[rank]
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torchvision.datasets import ImageFolder
    from torchvision import transforms    
    from split_train_val_test import get_or_create_datasetsCSVpaths
    from pathlib import Path

    results_folder = Path("./results_YOUR_EXPERIMENT") #TODO <------ set your desired output folder based on your experiments
    results_folder.mkdir(exist_ok = True)

    ## Set some stuff for the torch DDP setting, that is valid also for the single GPU setting anyway 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    ### Regarding torch DDP stuff, we need to specify which backend to use:
    ## if linux OS, then use "nccl";        
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    ## if Windows, use "gloo" instead: #TODO
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    ###

    if args.EXPERIMENT == "prostate": # prostate
        from dataset_creator import Dataset2DSL

        my_transform = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1) #TODO 14 settembre 2023: Provo a togliere la normalizzazione, perche tanto sono già valori fra [0,1] essendo Tensor
        ])

        my_transform_valid_and_test = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1) #TODO 14 settembre 2023: Provo a togliere la normalizzazione, perche tanto sono già valori fra [0,1] essendo tensori
        ])

        path_to_train_csv, path_to_val_csv, _ = get_or_create_datasetsCSVpaths(EXPERIMENT=args.EXPERIMENT, CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
        dataset_train = Dataset2DSL(csv_path=path_to_train_csv, dataset_name=dataset_name, CONDITIONING_FEATURE=CONDITIONING_FEATURE, transform=my_transform, use_label=True)
        dataset_val = Dataset2DSL(csv_path=path_to_val_csv, dataset_name=dataset_name, CONDITIONING_FEATURE=CONDITIONING_FEATURE, transform=my_transform_valid_and_test, use_label=True)
        
    elif args.EXPERIMENT == "breakhis": #  
        from dataset_creator import BREAKHISDataset2D

        my_transform = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1) #TODO 14 settembre 2023: Provo a togliere la normalizzazione, perche tanto sono già valori fra [0,1] essendo Tensor
        ])

        my_transform_valid_and_test = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1) #TODO 14 settembre 2023: Provo a togliere la normalizzazione, perche tanto sono già valori fra [0,1] essendo tensori
        ])

        path_to_train_csv, path_to_val_csv, _ = get_or_create_datasetsCSVpaths(EXPERIMENT=args.EXPERIMENT, CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path)
        dataset_train = BREAKHISDataset2D(csv_path=path_to_train_csv, cls_type="binary", transform=my_transform)
        dataset_val = BREAKHISDataset2D(csv_path=path_to_val_csv, cls_type="binary", transform=my_transform_valid_and_test)
        # dataset_test = BREAKHISDataset2D(csv_path=TEST, cls_type="binary", transform=my_transform,)   
    
    #### prepare the dataloaders
    from torch.utils.data.distributed import DistributedSampler

    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler_train)
    print(f"dataloader_train of size {len(dataloader_train)} batches, each of {batch_size_train}")

    sampler_valid = DistributedSampler(dataset_val, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader_valid = DataLoader(dataset_val, batch_size=batch_size_valid, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler_valid)
    print(f"dataloader_valid of size {len(dataloader_valid)} batches, each of {batch_size_valid}")
        

    ## when using torch DDP settings (multiple GPUs), each process has its own rank, e.g., 0,1,2: we set that value to the device variable governing the ordinal of the GPU device to use 
    device=rank

    ###### utility function to show some of the images; actually we do not use that..    
    # def imshow(img):
    #     size = img.size()[0]
    #     if size==3:
    #         if (dataset_name == "cifar10") or (dataset_name == "cifar10_user") or (dataset_name=="imagenette_user") or (dataset_name=="imagenette"):
    #             denormalize = transforms.Compose([
    #                 transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    #                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ])])
                
    #             img = denormalize(img)
    #         else:
    #             img = img / 2 + 0.5     # unnormalize, since they underwent the transforms.Lambda(lambda t: (t * 2) - 1)
            
    #         if img.get_device()==-1 : #cpu
    #             npimg = img.numpy()
    #         else: #gpu
    #             npimg = img.cpu().numpy()            
    #         plt.imshow(np.transpose(npimg, (1, 2, 0)).contiguous())  
    #         plt.show()

    #     else:           
    #         if img.get_device()==-1 : #cpu
    #             npimg = img.numpy()
    #         else: #gpu
    #             npimg = img.cpu().numpy()            
    #         npimg = np.transpose(npimg, (1, 2, 0)).contiguous()
    #         plt.figure(figsize=(14,14))
    #         for i in range(size):      
    #             plt.subplot(int(np.ceil(np.sqrt(size))), int(np.ceil(np.sqrt(size))), i+1)
    #             plt.imshow(npimg[:,:,i], cmap="magma")
    #             plt.axis('off')
    #             plt.title(f"ftrmp {i}",fontsize = 8)
    #         plt.show() 



    ###############
    number_of_feature_maps = None

    ## Below, we define the model, and move it to the GPU. We also define a standard optimizer (Adam).
    if model_type=="resnet18":

        # from networks_attn_learnLM_clean import Resnet18CA_clean 
        from networks_attn_learnLM_clean_ABL import Resnet18CA_clean # TODO 

        model = Resnet18CA_clean(
            dim=image_size,
            channels=channels,
            num_classes=num_classes,            
            is_pretrained=is_pretrained,
            is_feature_extractor=False,            
            causality_aware=causality_awareness,
            causality_method=causality_method,
            LEHMER_PARAM=lehmer_param,
            causality_setting=causality_setting,
            visual_attention=False, 
            MULCAT_CAUSES_OR_EFFECTS=mulcat_causes_or_effects
            )
        print("-#-#-#: intialized a Resnet18CA model from networks_attn_learnLM_clean_ABL")#: pretrainedweights ({is_pretrained}), usedAsFeatureExtractor ({is_feature_extractor}), causality-aware ({causality_awareness})") 

    # elif model_type=="resnext":
    #     from networks_attn import ResNextCA #TODO 6 sett 2023
    #     model = ResNextCA(
    #         which_resnext=which_resnext_to_use,
    #         dim=image_size,
    #         channels=channels,
    #         num_classes=num_classes,
    #         is_pretrained=is_pretrained, #False,
    #         is_feature_extractor=False,            
    #         causality_aware=causality_awareness,
    #         causality_method=causality_method,
    #         LEHMER_PARAM=lehmer_param,
    #         causality_setting=causality_setting,
    #         visual_attention=False, #TODO 12 july 2023 Just for now, disable attention computation
    #         MULCAT_CAUSES_OR_EFFECTS=mulcat_causes_or_effects #TODO 21 july 2023
    #         )
    #     print(f"-#-#-#: intialized a ResNext_{which_resnext_to_use}_CA model from networks_attn") 
    

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    #define loss function criterion and optimizer
    if loss_type == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()        
    else:
        print("Please, specify a valid loss function type, such as CrossEntropyLoss")
        raise NotImplementedError
   
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    

    # for name, param in model.named_parameters():
    #     if "causality_map_extractor.lehmer_seed" in name:
    #         print(f"{name}\t {param.data}")
    # print()


    ## choose your desired scheduling regime...
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=epochs) #end_factor=start_factor means no effect.
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=round(epochs/2))
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.20*epochs), round(0.50*epochs)], gamma=0.5)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.25*epochs), round(0.50*epochs), round(0.75*epochs),], gamma=0.5)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    

    min_valid_loss = float("inf")
    path_to_model_dir = ""
    
    dateTimeObj = datetime.now() # current date and time
    date_time = dateTimeObj.strftime("%Y%m%d%H%M%S")  

    list_of_epochLosses = []
    list_of_validLosses = []
    list_of_validAccs = []

    def print_number_of_model_parameters_and_MB(model):
        total_params = sum(
            param.numel() for param in model.parameters()
        )
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return total_params, trainable_params, size_all_mb

    ## Let's define a custom save_stamp for this experiment, that will be used to define the corresponding output directory
    save_stamp = date_time + f"_{epochs}e_{image_size}i_{batch_size_train}b_{learning_rate}L_{weight_decay}w"
    
    causality_setting_TMP = causality_setting + "_" + mulcat_causes_or_effects + "_ablation" #TODO 13 nov
    model_type_TMP = model_type+which_resnext_to_use

    if is_pretrained:
        causality_awareness_TMP = str(causality_awareness)+"_pretrained"
    else:
        causality_awareness_TMP = causality_awareness

    if causality_awareness:
        if causality_method=="lehmer":            
            path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type_TMP, str(SEED), causality_setting_TMP, f"CA_{causality_awareness_TMP}", causality_method, str(lehmer_param), f"{save_stamp}")
            # path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type_TMP, causality_setting_TMP, f"CA_{causality_awareness_TMP}", causality_method, f"{save_stamp}")
        else:
            path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type_TMP, str(SEED), causality_setting_TMP, f"CA_{causality_awareness_TMP}", causality_method, f"{save_stamp}")
    else:
        path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type_TMP, str(SEED), f"CA_{causality_awareness_TMP}",f"{save_stamp}")


    if not os.path.exists(path_to_model_dir):
        os.makedirs(path_to_model_dir,exist_ok=True)
        with open(os.path.join(path_to_model_dir,"settings_of_this_experiment.txt"),"w") as fout:
            fout.write(f" csv_path: {csv_path}\n\
            dataset_name: {dataset_name}\n\
            SEED: {SEED}\n \
            GPU used: {world_size}\n \
                ---list_of_GPU_ids: {list_of_GPU_ids}\n \
            dataset_name: {dataset_name}\n \
            number of image classes: {num_classes}\n \
            channels: {channels}\n \
            image_size: {image_size}\n \
            batch_size_train: {batch_size_train}\n \
            batch_size_valid: {batch_size_valid}\n \
            batch_size_test: {batch_size_test}\n \
            Dataloader_train of size: {len(dataloader_train)} batches\n \
            epochs: {epochs}\n \
            initial_LR: {learning_rate}\n \
            LR Scheduler: {scheduler}\n \
            wd: {wd}\n \
            loss_type: {loss_type}\n \
            model_type: {model_type}\n \
                --- if EqualCNN, No. f maps: {number_of_feature_maps}\n \
                --- if ResNext, which type: {which_resnext_to_use}\n \
            is_pretrained: {is_pretrained} \n \
            is_feature_extractor: {is_feature_extractor} \n \
            causality_aware: {causality_awareness} \n \
                ---causality_method: {causality_method} \n \
                ---LEHMER PARAM (alpha, or p): {lehmer_param} \n \
                ---causality_setting: {causality_setting_TMP} \n \
            {model}")

    
    #### DEFINE validation loop
    def validation_loop(model, dataloader_valid, IS_CAUSALITY_AWARE, loss_function):
        
        model.eval()
        accuracy_validation = 0.0
        total_validation = 0.0
        valid_loss = 0.0

        ytrue_validation = []
        yscore_validation = []       

        with torch.no_grad():
            need_to_savefig = False
            for images_v, labels_v in dataloader_valid:                

                images_v = images_v.to(device)
                # labels_v = labels_v.to(device)
                ytrue_validation.append(labels_v.item())
                
                if IS_CAUSALITY_AWARE:                    
                    outputs_v, batch_causality_maps = model(images_v)
                    #
                    yscore_validation.append(outputs_v.detach().cpu().numpy()[:,1])
                    #

                    if need_to_savefig and dist.get_rank()==0: #save the figure only if it is the MASTER process (GPU with rank 0)                        
                        path_to_causality_maps = os.path.join(path_to_model_dir, "caumps")
                        if not os.path.exists(path_to_causality_maps):
                                os.makedirs(path_to_causality_maps,exist_ok=True)
                        path_to_original_images = os.path.join(path_to_model_dir, "orgimg")
                        if not os.path.exists(path_to_original_images):
                                os.makedirs(path_to_original_images,exist_ok=True)
                        need_to_savefig=False                            
                        if model_type=="resnet18":                           
                            for b_i in range(batch_causality_maps.size()[0]):                             
                                if epoch>15:
                                    if np.random.random()<0.25:
                                        c_map = batch_causality_maps[b_i,:,:]
                                        c_map *= 100 #since they are probability values (0---1), multiply them for 100 to get % (percentage)
                                        c_map = c_map.cpu().numpy()                                
                                        np.save(os.path.join(path_to_causality_maps,f"e{epoch}_c{b_i}.npy"), c_map) 
                                else:
                                    c_map = batch_causality_maps[b_i,:,:]
                                    c_map *= 100 #since they are probability values (0---1), multiply them for 100 to get % (percentage)
                                    c_map = c_map.cpu().numpy()
                                    np.save(os.path.join(path_to_causality_maps,f"e{epoch}_c{b_i}.npy"), c_map)
                else:
                    outputs_v, _ = model(images_v)
                    #
                    yscore_validation.append(outputs_v.detach().cpu().numpy()[:,1])
                    #
                    if need_to_savefig and dist.get_rank()==0:                    
                        path_to_original_images = os.path.join(path_to_model_dir, "orgimg")
                        if not os.path.exists(path_to_original_images):
                                os.makedirs(path_to_original_images,exist_ok=True)
                        need_to_savefig=False
                        if model_type=="resnet18":                          
                            if epoch == 3 :                            
                                # print("Validation: saving org img figure")
                                plt.figure()
                                tmpimage=images_v[0,:,:,:].cpu().numpy()
                                tmpimage=np.transpose(tmpimage, (1, 2, 0))
                                tmpimage=255*(tmpimage-tmpimage.min())/(tmpimage.max()-tmpimage.min())
                                tmpimage = tmpimage.astype(np.uint8)
                                plt.imshow(tmpimage)
                                plt.savefig(os.path.join(path_to_original_images,f"ep{epoch}_i0.png"))
                                plt.close()                         

                

                #print(f"Model output validation: {outputs_v}")
                labels_v=labels_v.to(device)
                loss_val = loss_function(outputs_v,labels_v)
                valid_loss += loss_val.item() * images_v.size(0) / len(dataloader_valid.dataset) #TODO#####
                # the class with the highest energy is what we choose as prediction
                predicted = torch.argmax(outputs_v, 1)               
                total_validation += labels_v.size(0)
                count_correct_guess = (torch.eq(predicted,labels_v)).sum().item()                
                accuracy_validation += count_correct_guess
        
        #list_of_validLosses.append(np.round(valid_loss,decimals=6))
        accuracy_validation = 100 * (accuracy_validation / total_validation)
        auroc_softmax = roc_auc_score(ytrue_validation, yscore_validation)

        return valid_loss, accuracy_validation, auroc_softmax
    ## END of the validation loop ######

    # If you need it, define the early stopping by declaring patience and minimun delta to be used in the validation loss tracking
    early_stopper = EarlyStopper(patience=5, min_delta=0.005) #If you trigger validation loop every three epochs, then setting a patience of 5 means you'll wait for 5*3=15 epochs in absolute sense.
 

    ## Define some stuff prior to starting training
    tmp_val_acc_value = 0
    tmp_val_loss_value = 0
    with open(os.path.join(path_to_model_dir,"results.txt"),"w") as fout:
            fout.write("Results\n")
    
    if dist.get_rank()==0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=path_to_model_dir)

    count_model_params, count_model_params_trainable, count_model_MB = print_number_of_model_parameters_and_MB(model=model)
    if dist.get_rank()==0:
        writer.add_scalar("Model/count_model_params", count_model_params)
        writer.add_scalar("Model/count_model_params_trainable", count_model_params_trainable)
        writer.add_scalar("Model/count_model_MB", count_model_MB)


    ## Let's start training!
    for epoch in range(epochs):
        if dist.get_rank()==0:
            print(f"EPOCH {epoch}---------")
        dataloader_train.sampler.set_epoch(epoch)    ## we are using DistributedSampler, thus we have to tell it which epoch this is       
       
        epoch_loss = 0.0 # the running loss

        model.train()        
        
        for batch_images,batch_labels in tqdm(dataloader_train):           

            optimizer.zero_grad(set_to_none=True) 
            step_batch_size = batch_images.size()[0]
            images = batch_images.to(device)
            labels = batch_labels.to(device)   

            ##TODO when running in Windows with  'gloo' backend.
            # labels = labels.type(torch.LongTensor).to(device) #TODO when running in Windows with  'gloo' backend.
    
            outputs, _ = model(images)
            loss = loss_function(outputs,labels)
            if not torch.isnan(loss): ##TODO 30 Oct
                loss.backward()        
                #Step with the optimzer
                optimizer.step()  
                #Keep track of the loss during epochs
                epoch_loss += loss.item() * step_batch_size / len(dataloader_train.dataset)
            else:
                print("LOSS WAS Nan IN THIS EPOCH, JUST SKIPPING THE .BACKWARD() AND .STEP()...")
            

        if dist.get_rank()==0:      
            with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                fout.write(f"epoch:  {epoch},    training loss: {epoch_loss}\n")
                list_of_epochLosses.append(epoch_loss) # training loss collection

        if dist.get_rank()==0:
            writer.add_scalar("Loss/train", epoch_loss, epoch)

        if ((epoch>0) and (epoch%3==0)): #Trigger validation loop every three epochs... TODO customize it if needed
            validation_loss, validation_accuracy, auroc_softmax = validation_loop(model, dataloader_valid, causality_awareness, loss_function)

            if dist.get_rank()==0:
                writer.add_scalar("Loss/valid", validation_loss, epoch)
                writer.add_scalar("Acc/valid", validation_accuracy, epoch)
                writer.add_scalar("AUROC/valid", auroc_softmax, epoch)

                with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                    fout.write(f"    val loss: {validation_loss}, val acc: {validation_accuracy}, val auroc: {auroc_softmax}\n")                

            if min_valid_loss > validation_loss: 
                min_valid_loss = validation_loss
                if dist.get_rank()==0:
                    writer.add_scalar("Best/valid_loss", validation_loss, epoch)
                    writer.add_scalar("Best/acc_at_best_valid_loss", validation_accuracy, epoch)
                    ##
                    path_to_model_epoch = os.path.join(path_to_model_dir,f"ep{epoch}_betterValid")                
                    torch.save(model.state_dict(), path_to_model_epoch)  
                    # get a list of all .pth files in the models directory
                    file_list = glob.glob(os.path.join(path_to_model_dir,"*_betterVal*"))
                    # sort the list by the epoch number in ascending order
                    file_list.sort(key=lambda x: int(x.split("ep")[1].split("_")[0]))
                    # delete all files except the last 3 ones
                    for file in file_list[:-3]:
                        os.remove(file)

            list_of_validLosses.append(validation_loss)
            list_of_validAccs.append(validation_accuracy)
            tmp_val_loss_value = validation_loss
            tmp_val_acc_value = validation_accuracy

            # check for early stopping during training:
            flag_tensor = torch.zeros(1).to(device)
            if dist.get_rank()==0:
                if early_stopper.early_stop(validation_loss, epoch):
                    flag_tensor += 1
            dist.all_reduce(flag_tensor)
            if flag_tensor == 1:
                with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                    fout.write(f"Exit condition from early stop on validation loss (earlyStopper patience and minDelta: {early_stopper.get_patience_and_minDelta()})")             
                break
                
        else:
            list_of_validLosses.append(tmp_val_loss_value)
            list_of_validAccs.append(tmp_val_acc_value)
        
        ## Create, or update, the figure during training every so often (e.g., once in 9 epochs) since the graphics might be a bottleneck:
        if (((epoch > 0) and epoch % 9 == 0) or (epoch == epochs-1)):
            plt.figure()
            plt.plot(list_of_epochLosses,'k-') # training
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Training loop updated at epoch {epoch}")
            plt.plot(list_of_validLosses,'b-') # validation
            plt.show()
            plt.savefig(os.path.join(path_to_model_dir,"training_and_validation_loss_curve.pdf"))
            plt.close()

            plt.figure()
            plt.plot(list_of_validAccs,'b-')
            plt.xlabel("Epochs")
            plt.ylabel("Validation Accuracy")
            plt.title(f"Training loop updated at epoch {epoch}")
            plt.show()
            plt.savefig(os.path.join(path_to_model_dir,"validation_acc_curve.pdf"))
            plt.close()
        
        ## if some kind of learning rate scheduler is used, then you can see how its value changes during training
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        if dist.get_rank()==0:
            print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        ##
    
    if dist.get_rank()==0:
        writer.flush() #to make sure that all pending events have been written to disk.
        writer.close()


    with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                fout.write("---End of this training---")

    # Clean up the process groups:
    dist.destroy_process_group()
    pass

#%% MAIN

import torch.multiprocessing as mp
if __name__ == '__main__': #main(rank, world_size, causality_awareness, learning_rate, causality_method=None, lehmer_param=None):
    world_size=args.number_of_gpus
    
    for CA_method in causality_awareness_method: #none,max,lehmer
       
        if CA_method is None:
            for lr in LR:
                for we_de in wd:
                    print(f"Sto lanciando CA None e LR={lr}, wd={we_de}")
                    mp.spawn(
                        main,
                        args=(world_size, False, lr, we_de, None, None),
                        nprocs=world_size
                    )
        elif CA_method=="max":
            for causality_setting in CAUSALITY_SETTING: #cat,mulcat,mulcatbool
                for mulcat_causes_or_effects in MULCAT_CAUSES_OR_EFFECTS: #TODO 21 luglio                    
                    for lr in LR:                        
                        for we_de in wd:
                            print(f"Sto lanciando CA max e LR={lr}, wd={we_de}, causality_setting {causality_setting}, con mulcat_causes_or_effects {mulcat_causes_or_effects}")
                            mp.spawn(
                                main,
                                args=(world_size, True, lr, we_de, "max", 0, causality_setting, mulcat_causes_or_effects),
                                nprocs=world_size
                            )
        elif CA_method=="lehmer":
            for causality_setting in CAUSALITY_SETTING: #cat,mulcat,mulcatbool
                for mulcat_causes_or_effects in MULCAT_CAUSES_OR_EFFECTS: #TODO 21 luglio
                    for alpha in LEHMER_PARAM:
                        for lr in LR:
                            for we_de in wd:
                                print(f"Sto lanciando CA lehmer con alpha {alpha} e LR={lr}, wd={we_de}, causality_setting {causality_setting}, con mulcat_causes_or_effects {mulcat_causes_or_effects}")
                                mp.spawn(
                                    main,
                                    args=(world_size, True, lr, we_de, "lehmer", alpha, causality_setting, mulcat_causes_or_effects),
                                    nprocs=world_size
                                )
        else:
            print("errore nel ciclo for per CA_method")