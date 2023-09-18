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
from torchvision.utils import make_grid
import random
from textwrap import wrap
#from itertools import compress



import seaborn as sns
#import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
 

parser = argparse.ArgumentParser(description="Training script for a convnet.")
parser.add_argument("number_of_gpus",type=int,help="The number of GPUs you intend to use")
parser.add_argument("gpus_ids",type=str,help="The comma separated list of integers representing the id of requested GPUs - such as '0,1'")
#
parser.add_argument("SEED",type=int,help="fix seed to set reproducibility, eg, seed=42")
parser.add_argument("EXPERIMENT",type=str,help="cifar10, prostate, imagenette")
parser.add_argument("CONDITIONING_FEATURE",type=str,help="for imagenette is imagenette, for prostate can be (aggressiveness, no_tumour, scanner_vendor, disease_yes_no)")
parser.add_argument("IMAGE_SIZE",type=int,help="imagenette 64, prostate 128")
parser.add_argument("BATCH_SIZE_TRAIN",type=int,help="eg, 200 images")
parser.add_argument("BATCH_SIZE_VALID",type=int,help=" 1 image")
parser.add_argument("BATCH_SIZE_TEST",type=int,help=" 1 image")
parser.add_argument("NUMBER_OF_EPOCHS",type=int,help="eg, 100, eventually early stopped")
parser.add_argument("MODEL_TYPE",type=str,help="eg, EqualCNN, alexnet, SimpleCNN, EqualCNN, BaseCNN, LightCNN, resnet18, resnet34, resnet50, resnet101, ")

parser.add_argument("ADAM_LR",type=str,help="list of str of floats, eg, [0.0003, 0.001]")
parser.add_argument("WD",type=str,help="list of str of floats, eg, [0.0003, 0.001]")
parser.add_argument("CAUSALITY_AWARENESS_METHOD", type=str, default= None, help="[None, 'max', 'lehmer']")
parser.add_argument("LEHMER_PARAM", type=str, help="if using Lehmer mean, which power utilize among: [-100,-1,0,1,2,100]")
#15 giugno 2023:
parser.add_argument("CAUSALITY_SETTING", type=str, help="if CA, which setting to use among: ['cat','mulcat']")
#21 luglio 2023:
parser.add_argument("MULCAT_CAUSES_OR_EFFECTS", type=str, default="['causes']", help="if CA, which one to use for causality factors computation: ['causes','effects']")

args = parser.parse_args()

print()
print()
print("###### ABLATION STUDIES #####")
print()
print()

###repoducibility:#############
SEED = args.SEED
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED) 
## newly added
#torch.backends.cudnn.benchmark = False 
#torch.use_deterministic_algorithms(True)
#g_cuda = torch.Generator(device="cuda").manual_seed(SEED)
#g_cpu = torch.Generator(device="cpu").manual_seed(SEED)




""" def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) """



#model
model_type = args.MODEL_TYPE




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

adam_LR = literal_eval(args.ADAM_LR) #list of floats, "[0.001,0.0003]"
wd = literal_eval(args.WD)

print(f"LITERAL EVALUATIONS: causality_awareness_method {causality_awareness_method}, LEHMER_PARAM {LEHMER_PARAM}, adam_LR {adam_LR}, wd {wd}, causality_setting {CAUSALITY_SETTING}")


#some other fixed settings
loss_type="CrossEntropyLoss"
is_pretrained = False
is_feature_extractor = False # (SOLO SE USI 3 CANALI HA SENSO, ALTRIMENTI PER FORZA LA RIALLENI TUTTA)
csv_path=""
train_root_path = val_root_path = test_root_path = ""



if args.EXPERIMENT == "imagenette":
    dataset_name = "imagenette"
    CONDITIONING_FEATURE = "imagenette"
    channels = 3
    num_classes = 10
    train_root_path=os.path.join(os.getcwd(),"imagenette2","train")
    val_root_path=os.path.join(os.getcwd(),"imagenette2","val")
    test_root_path=os.path.join(os.getcwd(),"imagenette2","test")
    
elif args.EXPERIMENT == "prostate":
    dataset_name=""
    CONDITIONING_FEATURE = args.CONDITIONING_FEATURE
    channels = 1
    num_classes = 2
    if CONDITIONING_FEATURE == "aggressiveness": # lesion aggressiveness labels: LG and HG
        #dataset_name = "DATASET_4_MEDIUM_RESOLUTION_NORMALIZED_GUIDED_CROP"
        dataset_name = "UNFOLDED_DATASET_5_LOW_RESOLUTION_NORMALIZED_GUIDED_CROP_GUIDED_SLICE_SELECTION"
        #csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","trainingset_clinically_significant.csv") ## versione di csv di prima (before 3 april) col dataset "sbagliato" (384imgsize, fetta centrale delle 12 di ogni paziente)...
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","cs_les_unfolded.csv") ## versione nuova, con train valid e test tutti uniti
    elif CONDITIONING_FEATURE == "no_tumour": # 1 may 2023: lets try to use a different csv, the one for 861 no-tumour patients: in here, 363 patients were sent to biopsy but ultimately did not tumour, this means that the appearance of their image has warned the radiologist
        dataset_name = "UNFOLDED_DATASET_NOTUMOUR"
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","metadata_no_tumour_unfolded.csv")
    elif CONDITIONING_FEATURE == "scanner_vendor": # 3-4 may 2023: manufacturer, over the dataset with 2079 clinically sign lesions' images
        dataset_name = "UNFOLDED_DATASET_5_LOW_RESOLUTION_NORMALIZED_GUIDED_CROP_GUIDED_SLICE_SELECTION"
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","cs_les_unfolded_manufacturer_noNone.csv") #TODO: eliminated the 30 rows that do not have scanner information, since they compromised the dataloader functioning
    elif CONDITIONING_FEATURE == "disease_yes_no": 
        dataset_name = "UNFOLDED_DATASET_DISEASE_YES_NO"
        #csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","unfolded_disease_YesNo.csv") #TODO: fonde i csv del caso "aggressiveness" e "no_tumour", ma da questo ultimo elimino tutte le righe di pazienti che erano stati mandati a biopsia ma che poi si sono rivelati non essere tumorali, cosi da avere le due classi ben distinte: i pazienti che non hanno fatto manco la biopsia e i pazienti che hanno biopsia clinically significant
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","unfolded_disease_YesNo_balanced.csv") #TODO: la versione bilanciata del precedente: per farlo, ho eliminato (frai pazienti senza tumore) quelli che aveano PSA piu alto (e in seconda chiave di ordinamento quelli che avevano il Volume piu alto), e poi ho riordinato le righe questa volta per pazienteID, quindi non sono stratificate come prima (dove aveo prima i tumorali e poi tutti i non tumorali)
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

    def early_stop(self, validation_loss):
        #return validation_loss < self.min_validation_loss # DEBUGGING PURPOSE 2 giugno june
    
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def get_patience_and_minDelta(self):
        return (self.patience, self.min_delta)









def main(rank, world_size, causality_awareness, learning_rate, weight_decay, causality_method=None, lehmer_param=None, causality_setting="cat",mulcat_causes_or_effects="causes"):
    torch.cuda.is_available() 
    os.environ['CUDA_VISIBLE_DEVICES'] = list_of_GPU_ids[rank]
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    
    from split_train_val_test import get_or_create_datasetsCSVpaths
    from networks import LightCNN,EqualCNN, BaseCNN, SimpleCNN

    from pathlib import Path
    #results_folder = Path("./results") #1 giugno
    results_folder = Path("./results_june") #esperimenti sistematici giugno luglio agosto 2023
    results_folder.mkdir(exist_ok = True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
   

    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



    

    ###

    if (dataset_name == "cifar10") or (dataset_name == "cifar10_user"):

        from torchvision.datasets import CIFAR10
        
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        my_transform = Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        my_transform_valid = Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset_train = CIFAR10(root='./data', train=True, download=True, transform=my_transform)
        dataset_val = CIFAR10(root='./data', train=False, download=True, transform=my_transform_valid)
        #print(f"Using {dataset_name}, train dataset: len={len(dataset_train)}")
        #print(f"Using {dataset_name}, valid dataset: len={len(dataset_val)}")
        
        
    elif (dataset_name=="imagenette_user" or dataset_name=="imagenette"):
        
        
        classes = ("carpFish",  #n01440764
                   "dog", #n02102040
                   "radioStereo", #n02979186
                   "chainsaw", #n03000684
                   "churchCathedral", #n03028079
                   "hornTrumpet", #n03394916
                   "garbageTruck", #n03417042
                   "fuelStation", #n03425413
                   "golfBall", #n03445777,
                   "parachute", #n03888257
                   )
        
        my_transform = Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        my_transform_valid_and_test = Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset_train = ImageFolder(root=train_root_path, transform=my_transform)
        dataset_val = ImageFolder(root=val_root_path, transform=my_transform_valid_and_test)
        # dataset_test = ImageFolder(root=test_root_path, transform=my_transform_valid_and_test)

      
    elif args.EXPERIMENT == "prostate": # prostate
        from dataset_creator import Dataset2DSL

        my_transform = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        my_transform_valid_and_test = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        #path_to_train_csv, path_to_val_csv, path_to_test_csv = get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
        path_to_train_csv, path_to_val_csv, _ = get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
        dataset_train = Dataset2DSL(csv_path=path_to_train_csv, dataset_name=dataset_name, CONDITIONING_FEATURE=CONDITIONING_FEATURE, transform=my_transform, use_label=True)
        dataset_val = Dataset2DSL(csv_path=path_to_val_csv, dataset_name=dataset_name, CONDITIONING_FEATURE=CONDITIONING_FEATURE, transform=my_transform_valid_and_test, use_label=True)
        # dataset_test = Dataset2DSL(csv_path=path_to_test_csv, dataset_name=dataset_name, transform=my_transform, use_label=True)   
        

 


    # prepare the dataloaders
    from torch.utils.data.distributed import DistributedSampler

    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler_train)
    #dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, pin_memory=False, num_workers=16, worker_init_fn=seed_worker, drop_last=False, shuffle=False, sampler=sampler_train)
    print(f"dataloader_train of size {len(dataloader_train)} batches, each of {batch_size_train}")

    sampler_valid = DistributedSampler(dataset_val, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader_valid = DataLoader(dataset_val, batch_size=batch_size_valid, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler_valid)
    #dataloader_valid = DataLoader(dataset_val, batch_size=batch_size_valid, pin_memory=False, num_workers=16, worker_init_fn=seed_worker, drop_last=False, shuffle=False, sampler=sampler_valid)
    print(f"dataloader_valid of size {len(dataloader_valid)} batches, each of {batch_size_valid}")

    """  sampler_test = DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler_test)
    print(f"dataloader_test of size {len(dataloader_test)} batches, each of {batch_size_test}")
    """
    

    ## 31 march: provo a mettere device=rank, non so se ci va proprio il device...bho
    device=rank

    #########################
    # get some random training images
    
    def imshow(img):
        size = img.size()[0]
        if size==3:
            if (dataset_name == "cifar10") or (dataset_name == "cifar10_user") or (dataset_name=="imagenette_user") or (dataset_name=="imagenette"):
                denormalize = transforms.Compose([
                    transforms.Normalize(mean = [ 0., 0., 0. ],std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ])])
                
                img = denormalize(img)
            else:
                img = img / 2 + 0.5     # unnormalize, since they underwent the transforms.Lambda(lambda t: (t * 2) - 1)
            
            if img.get_device()==-1 : #cpu
                npimg = img.numpy()
            else: #gpu
                npimg = img.cpu().numpy()
            
            plt.imshow(np.transpose(npimg, (1, 2, 0)))  
            plt.show()

        else:
            

            if img.get_device()==-1 : #cpu
                npimg = img.numpy()
            else: #gpu
                npimg = img.cpu().numpy()
            
            npimg = np.transpose(npimg, (1, 2, 0))

            plt.figure(figsize=(15,15))
            for i in range(size):      
                plt.subplot(int(np.ceil(np.sqrt(size))), int(np.ceil(np.sqrt(size))), i+1)
                plt.imshow(npimg[:,:,i])
                plt.axis('off')
                plt.title(f"ftrmp {i}",fontsize = 9)
            plt.show()
            #plt.savefig(os.path.join(results_folder,f"im__featuremaps.pdf"))
            #plt.close()
        
    
    

    # dataiter = iter(dataloader_valid)
    # images, labels = next(dataiter)
    # images = images[2:4]
    # labels = labels[2:4]
    # # show images
    # plt.figure()
    # imshow(make_grid(images))
    # # print labels
    # labels = labels.tolist()
    # batch_labels = [classes[labels[j]] for j in range(images.size(0))]
    # print(batch_labels)
    # plt.title(batch_labels)
    # plt.savefig(os.path.join(os.getcwd(),f"some_{dataset_name}_images1.pdf"))
    # plt.close()
    

    ########################
    number_of_feature_maps = None

    ## Below, we define the model, and move it to the GPU. We also define a standard optimizer (Adam).
    if model_type=="LightCNN":
        model = LightCNN(
            dim=image_size,
            channels=channels,
            num_classes=num_classes
        )
    elif model_type=="SimpleCNN":
        from settings import INFERENCE_MODE
        model = SimpleCNN(
            dim=image_size,
            channels=channels,
            num_classes=num_classes,
            inference_mode= INFERENCE_MODE, #TODO
            causality_aware=causality_awareness ##TODO
        )
    elif model_type=="EqualCNN":

        ## DEFINE
        number_of_feature_maps = 256 #32 #64 #TODO #(32), 64, 128, 256 #################################################################

        model = EqualCNN( # number of channels (ftmaps) 3>16>32>32>32>32>32>32... remains equally 32, straight
            NUMB_FTRS_MPS=number_of_feature_maps, 
            dim=image_size,
            channels=channels,
            num_classes=num_classes,
            causality_aware=causality_awareness,
            causality_method=causality_method,
            LEHMER_PARAM=lehmer_param
        )
    elif model_type=="BaseCNN":
        model = BaseCNN(
            dim=image_size,
            channels=channels,
            num_classes=num_classes,
            causality_aware=causality_awareness
        )

    elif model_type=="resnet18":
        # from networks import Resnet18CA
        from networks_attn_ablation import Resnet18CA #TODO 27 july 2023
        model = Resnet18CA(
            dim=image_size,
            channels=channels,
            num_classes=num_classes,
            is_pretrained=False,
            is_feature_extractor=False,            
            causality_aware=causality_awareness,
            causality_method=causality_method,
            LEHMER_PARAM=lehmer_param,
            causality_setting=causality_setting,
            visual_attention=False, #TODO 12 july 2023
            MULCAT_CAUSES_OR_EFFECTS=mulcat_causes_or_effects #TODO 21 july 2023
            )
        print("-#-#-#: intialized a Resnet18CA model from networks_attn_ablation")#: pretrainedweights ({is_pretrained}), usedAsFeatureExtractor ({is_feature_extractor}), causality-aware ({causality_awareness})") 

        
    """ elif model_type=="resnet34":
        from torchvision.models import resnet34
        model = resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(channels,64,kernel_size=7, stride=2, padding=3, bias=False) #change the first layer to handle gray-level images
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) 
    elif model_type=="resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(channels,64,kernel_size=7, stride=2, padding=3, bias=False) #change the first layer to handle gray-level images
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) 
    elif model_type=="resnet101":
        from torchvision.models import resnet101
        model = resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(channels,64,kernel_size=7, stride=2, padding=3, bias=False) #change the first layer to handle gray-level images
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) 
    elif model_type == 'VGG16':
        from torchvision.models import vgg16
        model = vgg16(pretrained=True)
        model.features[0] = nn.Conv2d(channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        model.classifier[6] = nn.Linear(4096, args.num_classes)
    
    elif model_type == "alexnet":
        from networks import AlexnetCA
        model = AlexnetCA(img_size=image_size, num_classes=num_classes, causality_aware=causality_awareness)
        print(f"-#-#-#: intialized a AlexnetCA model with causality-aware: {causality_awareness}")  """
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)




    #define loss function criterion and optimizer
    if loss_type == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
        
    else:
        print("Please, specify loss function type in settings.py and train_convet.py, such as CrossEntropyLoss")
        raise NotImplementedError

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #TODO 12 luglio
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs)

    # 13 giugno june 2023
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=round(epochs/2))
    # 14 giugno 2023# Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.20*epochs), round(0.50*epochs)], gamma=0.1)
    #
    
    min_valid_loss = float("inf")
    path_to_model_dir = ""
    
    """ ## MAY 4 2023: created subfolder for dividing experiments based on the conditioning feature
    if not os.path.exists(os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE,f"CA_{causality_awareness}")):
        os.makedirs(os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE,f"CA_{causality_awareness}"),exist_ok=True)
    ## """

    dateTimeObj = datetime.now() # current date and time
    date_time = dateTimeObj.strftime("%Y%m%d%H%M%S")  

    list_of_epochLosses = []
    list_of_validLosses = []
    list_of_validAccs = []

    def print_number_of_model_parameters(model):
        total_params = sum(
            param.numel() for param in model.parameters()
        )

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        return f"({total_params}, {trainable_params})"



    #print(f"STARTING TRAINING OVER EPOCHS-----") #, upon {epochs_of_warmup} warmup epochs...")

    save_stamp = date_time + f"_{epochs}e_{image_size}i_{batch_size_train}b_{learning_rate}L_{weight_decay}w"
    




    # if CAUSALITY_AWARE:
    #     if causality_method=="lehmer":
    #         path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE,f"CA_{CAUSALITY_AWARE}",causality_method,str(LEHMER_PARAM),model_type,f"{save_stamp}")
    #     else:
    #         path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE,f"CA_{CAUSALITY_AWARE}",causality_method,model_type,f"{save_stamp}")

    # else:
    #     path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE,f"CA_{CAUSALITY_AWARE}",model_type,f"{save_stamp}")
    
    



    # ## 5 giugno, spostato indentazione modeltype prima delle altre
    # if causality_awareness:
    #     if causality_method=="lehmer":
    #         if model_type=="EqualCNN":
    #             path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting, str(number_of_feature_maps), f"CA_{causality_awareness}", causality_method, str(lehmer_param), f"{save_stamp}")
    #         else:
    #             path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting, f"CA_{causality_awareness}", causality_method, str(lehmer_param), f"{save_stamp}")

    #     else:
    #         if model_type=="EqualCNN":
    #             path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting, str(number_of_feature_maps), f"CA_{causality_awareness}", causality_method, f"{save_stamp}")
    #         else:
    #             path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting, f"CA_{causality_awareness}", causality_method, f"{save_stamp}")

    # else:
    #     if model_type=="EqualCNN":
    #         path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, str(number_of_feature_maps), f"CA_{causality_awareness}",f"{save_stamp}")
    #     else:
    #         path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, f"CA_{causality_awareness}",f"{save_stamp}")




    ## TODO 21 luglio: creata variabile temporanea per il nome causality_setting per creare la cartella corretta
    causality_setting_TMP = causality_setting + "_" + mulcat_causes_or_effects + "_ablation"
    if causality_awareness:
        if causality_method=="lehmer":
            if model_type=="EqualCNN":
                path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting_TMP, str(number_of_feature_maps), f"CA_{causality_awareness}", causality_method, str(lehmer_param), f"{save_stamp}")
            else:
                path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting_TMP, f"CA_{causality_awareness}", causality_method, str(lehmer_param), f"{save_stamp}")

        else:
            if model_type=="EqualCNN":
                path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting_TMP, str(number_of_feature_maps), f"CA_{causality_awareness}", causality_method, f"{save_stamp}")
            else:
                path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, causality_setting_TMP, f"CA_{causality_awareness}", causality_method, f"{save_stamp}")

    else:
        if model_type=="EqualCNN":
            path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, str(number_of_feature_maps), f"CA_{causality_awareness}",f"{save_stamp}")
        else:
            path_to_model_dir = os.path.join(results_folder,"saved_models",CONDITIONING_FEATURE, model_type, f"CA_{causality_awareness}",f"{save_stamp}")









    # se è la primissima volta, allora crea anche la cartella, altrimenti salva i modelli delle varie epoche nella cartella che esiste già
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
            adam_LR: {learning_rate}\n \
            wd: {wd}\n \
            loss_type: {loss_type}\n \
            model_type: {model_type}\n \
                --- if EqualCNN, No. f maps: {number_of_feature_maps}\n \
            is_feature_extractor: {is_feature_extractor} \n \
            causality_aware: {causality_awareness} \n \
                ---causality_method: {causality_method} \n \
                ---LEHMER PARAM (alpha, or p): {lehmer_param} \n \
                ---causality_setting: {causality_setting_TMP} \n \
            Number of parameters of the model (TOT, Trainable): {print_number_of_model_parameters(model=model)} \n\
            {model}")
        
        with open(os.path.join(results_folder,"saved_models","saved_model_names.txt"),"a") as fout:
            fout.write(f"\nmodel_{save_stamp}")


    feature_maps_hooked = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_maps_hooked[name] = output.detach()
        return hook

    ## validation loop
    def validation_loop(model, dataloader_valid, IS_CAUSALITY_AWARE, loss_function):
        
        ## 26 maggio 2023
        #handle_a = model.module.conv1_1.register_forward_hook(get_activation("conv1_1"))
        ### 29 maggio
        if dist.get_rank()==0:
            if model_type=="EqualCNN":
                handle_b = model.module.maxpool2D_1.register_forward_hook(get_activation("maxpool2D_1"))
                handle_c = model.module.maxpool2D_2.register_forward_hook(get_activation("maxpool2D_2"))
                handle_d = model.module.maxpool2D_3.register_forward_hook(get_activation("maxpool2D_3"))
            elif model_type=="resnet18":
                # handle_b = model.module.features.conv1.register_forward_hook(get_activation("conv1"))
                handle_b = model.module.starting_block.register_forward_hook(get_activation("starting_block")) #TODO 21 luglio

                # handle_c = model.module.features.layer1.register_forward_hook(get_activation("layer1"))
                # handle_d = model.module.features.layer2.register_forward_hook(get_activation("layer2"))
                # handle_e = model.module.features.layer3.register_forward_hook(get_activation("layer3"))
                # handle_f = model.module.features.layer4.register_forward_hook(get_activation("layer4"))
                

        ##
        
        model.eval()
        accuracy_validation = 0.0
        total_validation = 0.0

        #correct_v=0
        #total_v=0
        valid_loss = 0.0

        c_map_threshold = 50 #% percentage

        with torch.no_grad():
            need_to_savefig = False
            for count, (images_v, labels_v) in enumerate(dataloader_valid):
                
                if count == 1: #considera ad esempio le immagini del primo (0), o secondo (1) etc. batch, per esempio di visualizzazione durante validation
                    need_to_savefig = True

                images_v = images_v.to(device)
                labels_v = labels_v.to(device)

                
                if IS_CAUSALITY_AWARE:
                    
                    outputs_v, batch_causality_maps = model(images_v)

                    if need_to_savefig and dist.get_rank()==0: #save the figure only if it is the MASTER process (GPU with rank 0)
                        

                        path_to_feature_maps = os.path.join(path_to_model_dir, "ftrmps")
                        if not os.path.exists(path_to_feature_maps):
                                os.makedirs(path_to_feature_maps,exist_ok=True)
                        path_to_causality_maps = os.path.join(path_to_model_dir, "caumps")
                        if not os.path.exists(path_to_causality_maps):
                                os.makedirs(path_to_causality_maps,exist_ok=True)
                        path_to_original_images = os.path.join(path_to_model_dir, "orgimg")
                        if not os.path.exists(path_to_original_images):
                                os.makedirs(path_to_original_images,exist_ok=True)

                        need_to_savefig=False

                        #### 26 maggio 2023
                        #feature_maps_hooked will now contain the hooked activations of the forward passing of the batch through the model
                        #plt.figure()
                        #imshow(make_grid(feature_maps_hooked['conv1_1'], nrow = int(np.sqrt(feature_maps_hooked['conv1_1'].size()[2]))))



                        ## 29 maggio
                        """ imshow(make_grid(feature_maps_hooked['conv1_1']))
                        plt.savefig(os.path.join(path_to_model_dir,f"ep{epoch}_conv1_1.pdf"))
                        plt.close()
                        handle_a.remove() """
                        ##
                        if model_type=="EqualCNN":
                            imshow(make_grid(feature_maps_hooked['maxpool2D_1']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_maxpool2D_1.pdf"))
                            plt.close()
                            handle_b.remove()
                            ##
                            imshow(make_grid(feature_maps_hooked['maxpool2D_2']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_maxpool2D_2.pdf"))
                            plt.close()
                            handle_c.remove()
                            ##
                            imshow(make_grid(feature_maps_hooked['maxpool2D_3']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_maxpool2D_3.pdf"))
                            plt.close()
                            handle_d.remove()

                            #####
                            for b_i in range(batch_causality_maps.size()[0]):
                                ##
                                
                                tmpimage = images_v[b_i,:,:,:]
                                plt.figure()
                                imshow(tmpimage.cpu())
                                plt.savefig(os.path.join(path_to_original_images,f"ep{epoch}_i{b_i}.png"))
                                plt.close()
                                ##

                                c_map = batch_causality_maps[b_i,:,:]
                                c_map *= 100 #since they are probability values (0---1), multiply them for 100 to get % (percentage)
                                c_map = c_map.cpu().numpy()

                                c_map_min = np.around(c_map.min(), decimals=1)
                                c_map_max = np.around(c_map.max(),decimals=1)
                                #c_map_maxidx = np.argmax(c_map, keepdims=True)

                                c_map_overThresh = np.argwhere(c_map > c_map_threshold)  # ad esempio [[ 6 23], [17 23], [19 23]]
                                
                                indices = [(c_map_overThresh[n][0], c_map_overThresh[n][1]) for n in range(c_map_overThresh.shape[0])]
                                c_map= np.round(c_map, decimals=1) #for visualization purposes

                                fig=plt.figure(figsize=(20,20)) #2000px X 2000px = 1.6-2MB PNG
                                ax = fig.add_subplot(111)
                                sns.heatmap(c_map, annot=True, annot_kws={"fontsize":6}, linewidths=.5, vmax=100, vmin=0, square=True)                            
                                ##
                                title = ax.set_title("\n".join(wrap(f"Ep{epoch} causality map of im{b_i} of valid batch. Values from {c_map_min} to {c_map_max}. Indices of elements over {c_map_threshold}%: {indices}", 200)))
                                fig.tight_layout()
                                title.set_y(1.05)
                                fig.subplots_adjust(top=0.8)
                                ##
                                #plt.title(f"Ep{epoch} causality map of im{b_i} of valid batch\nvalues from {np.round(c_map.min(),decimals=1)} to {np.round(c_map.max(),decimals=1)} ({c_map_maxidx})\nindices elems over {c_map_threshold}: {indices}")
                                plt.show()
                                plt.savefig(os.path.join(path_to_causality_maps,f"ep{epoch}_cmap{b_i}.pdf"))
                                plt.close()

                        elif model_type=="resnet18":
                            # imshow(make_grid(feature_maps_hooked['conv1']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_conv1.pdf"))
                            # plt.close()
                            # handle_b.remove()
                            #TODO 21 luglio
                            imshow(make_grid(feature_maps_hooked['starting_block']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_strtngBlck.pdf"))
                            plt.close()
                            handle_b.remove()



                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer1']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer1.pdf"))
                            # plt.close()
                            # handle_c.remove()
                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer2']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer2.pdf"))
                            # plt.close()
                            # handle_d.remove()
                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer3']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer3.pdf"))
                            # plt.close()
                            # handle_e.remove()
                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer4']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer4.pdf"))
                            # plt.close()
                            # handle_f.remove()
                        ##
                        
                        

                        
                        
                else:
                    outputs_v, _ = model(images_v)
                    ##
                    if need_to_savefig and dist.get_rank()==0:
                        path_to_feature_maps = os.path.join(path_to_model_dir, "ftrmps")
                        
                        if not os.path.exists(path_to_feature_maps):
                            os.makedirs(path_to_feature_maps,exist_ok=True)
                        
                        need_to_savefig=False

                        if model_type=="EqualCNN":
                            imshow(make_grid(feature_maps_hooked['maxpool2D_1']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_maxpool2D_1.pdf"))
                            plt.close()
                            handle_b.remove()
                            ##
                            imshow(make_grid(feature_maps_hooked['maxpool2D_2']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_maxpool2D_2.pdf"))
                            plt.close()
                            handle_c.remove()
                            ##
                            imshow(make_grid(feature_maps_hooked['maxpool2D_3']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_maxpool2D_3.pdf"))
                            plt.close()
                            handle_d.remove()
                        elif model_type=="resnet18":
                            # imshow(make_grid(feature_maps_hooked['conv1']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_conv1.pdf"))
                            # plt.close()
                            # handle_b.remove()
                            
                            #TODO 21 luglio
                            imshow(make_grid(feature_maps_hooked['starting_block']))
                            plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_strtngBlck.pdf"))
                            plt.close()
                            handle_b.remove()


                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer1']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer1.pdf"))
                            # plt.close()
                            # handle_c.remove()
                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer2']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer2.pdf"))
                            # plt.close()
                            # handle_d.remove()
                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer3']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer3.pdf"))
                            # plt.close()
                            # handle_e.remove()
                            # ##
                            # imshow(make_grid(feature_maps_hooked['layer4']))
                            # plt.savefig(os.path.join(path_to_feature_maps,f"ep{epoch}_layer4.pdf"))
                            # plt.close()
                            # handle_f.remove()
                    
                #print(f"Model output validation: {outputs_v}")
                loss_val = loss_function(outputs_v,labels_v)
                valid_loss += loss_val.item() * images_v.size(0) / len(dataloader_valid.dataset) #TODO#####

                # the class with the highest energy is what we choose as prediction
                predicted = torch.argmax(outputs_v, 1)
                #print(predicted)
                
                total_validation += labels_v.size(0)

                #accuracy_validation += (predicted == labels_v).sum().item()
                count_correct_guess = (torch.eq(predicted,labels_v)).sum().item()
                #print(f"------ Model predicted: {predicted}, labels_v: {labels_v}, correct_guess {count_correct_guess}")
                
                accuracy_validation += count_correct_guess
        
        #list_of_validLosses.append(np.round(valid_loss,decimals=6))
        accuracy_validation = 100 * (accuracy_validation / total_validation)
        #list_of_validAccs.append(valid_acc)

        return valid_loss, accuracy_validation




    #early_stopper = EarlyStopper(patience=1, min_delta=0.025) # 
    early_stopper = EarlyStopper(patience=10, min_delta=0.005) # 
 

    tmp_val_acc_value = 0
    tmp_val_loss_value = 0
    with open(os.path.join(path_to_model_dir,"results.txt"),"w") as fout:
            fout.write("Results\n")




    for epoch in range(epochs):
        if dist.get_rank()==0:
            print(f"EPOCH {epoch}---------")
        dataloader_train.sampler.set_epoch(epoch)    ## 31 march:  if we are using DistributedSampler, we have to tell it which epoch this is
        
        ## TODO 25 may: it’s necessary to use set_epoch to guarantee a different shuffling order
        ## indeed, for the validation we may not want it to be different each time
        #dataloader_valid.sampler.set_epoch(epoch)
        
        ##
        epoch_loss = 0.0 # the running loss

        IS_CHECKPOINT = epoch % 100 == 0 # SAVE MODEL EVERY tot EPOCHS, AS A CHECKPOINT DISREGARDING THE VALUE OF THE VALIDATION LOSS
        model.train()
        for batch_images,batch_labels in tqdm(dataloader_train):
            optimizer.zero_grad(set_to_none=True) 

            step_batch_size = batch_images.size()[0]

            images = batch_images.to(device)
            labels = batch_labels.to(device) 

            

            outputs, _ = model(images)
            #
            # print("images ",images.size())
            # print("output ",outputs.size())
            # print("labels ",labels.size())
            #
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()  
            epoch_loss += loss.item() * step_batch_size / len(dataloader_train.dataset)
            if IS_CHECKPOINT:
                IS_CHECKPOINT = False              
                path_to_model_epoch = os.path.join(path_to_model_dir,f"ep{epoch}_chkpnt")
                torch.save(model.state_dict(), path_to_model_epoch)

            
            """ ## TODO version 7 giugno june 2023, where we do consider causality maps of the training batches:
            outputs, train_batch_cmaps = model(images)

            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step() 
            epoch_loss += loss.item() * step_batch_size / len(dataloader_train.dataset)
            if IS_CHECKPOINT:
                IS_CHECKPOINT = False              
                path_to_model_epoch = os.path.join(path_to_model_dir,f"ep{epoch}_chkpnt")
                torch.save(model.state_dict(), path_to_model_epoch) 
            # here is the addition on 7 giugno june 2023
            cond_asymms_dict = {} #for each class of image, it will contain the histogram of the index pairs corresponding to conditional asymmetries

            for b_i in range(train_batch_cmaps.size()[0]):                           
            
                c_map = train_batch_cmaps[b_i,:,:]
                c_map *= 100 #since they are probability values (0---1), multiply them for 100 to get % (percentage)
                #c_map = c_map.cpu().numpy()

                # versione 1
                # c_map= torch.round(c_map, decimals=1)
                # #c_map_maxidx = np.argmax(c_map, keepdims=True)
                # print(c_map)
                # print(c_map.size())
                # indices_gt0 = torch.argwhere(c_map) #returns the indices of non zero elements
                # indices_gt0_list = indices_gt0.tolist()
                # print(indices_gt0)
                # indices_gt0_rolled = torch.roll(indices_gt0, 1, dims=1) #swap i,j (--> j,i) of each pair of indices
                # del indices_gt0
                # print(indices_gt0_rolled)
                # indices_gt0_rolled_list = indices_gt0_rolled.tolist()
                # del indices_gt0_rolled
                # symmetric_pairs = [(pair in indices_gt0_rolled_list) for pair in indices_gt0_list]
                # symmetric_pairs = list(compress(indices_gt0_list, symmetric_pairs))
                # print(symmetric_pairs)
                # print(len(symmetric_pairs))
                # # poi capire come usare queste coppied
                
                # versione 2a
                #diag={}
                #for offset in range(-c_map.size()-1, c_map.size(), -1):
                #    diag[offset]=torch.diagonal(c_map, offset)
                # ecc, finire

                #versione 2b
                
                triu = torch.triu(c_map, 1) #upper triangular matrx (excluding the principal diagonal)
                tril = torch.tril(c_map, -1).T #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
                
                bool_matrix_ij = (tril > triu).T #True if P(i|j) > P(j|i) (transposed, to put it back in its original ordering)
                indices_cond_asymmetries_ij = torch.argwhere(bool_matrix_ij) 
                
                bool_matrix_ji = (triu > tril) #True if P(j|i) > P(i|j)
                indices_cond_asymmetries_ji = torch.argwhere(bool_matrix_ji)

                indices_cond_asymmetries = torch.cat([indices_cond_asymmetries_ij, indices_cond_asymmetries_ji])
                
                k = c_map.size()[0]

                #indices = [(c_map_overThresh[n][0], c_map_overThresh[n][1]) for n in range(c_map_overThresh.shape[0])]
                #c_map= np.round(c_map, decimals=1) #for visualization purposes """           
            # fine FOR del batch di training.
                
        with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
            fout.write(f"epoch:  {epoch},    training loss: {epoch_loss}\n")
        list_of_epochLosses.append(epoch_loss) # training loss collection

        #if (epoch>0):
        if ((epoch>0) and (epoch % 3 ==0)):
            validation_loss, validation_accuracy = validation_loop(model, dataloader_valid, causality_awareness, loss_function)

            with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                fout.write(f"    validation loss: {validation_loss}, validation accuracy: {validation_accuracy}\n")
            if min_valid_loss > validation_loss:
                min_valid_loss = validation_loss
                path_to_model_epoch = os.path.join(path_to_model_dir,f"ep{epoch}_betterValid")
                torch.save(model.state_dict(), path_to_model_epoch)   

            list_of_validLosses.append(validation_loss)
            list_of_validAccs.append(validation_accuracy)
            tmp_val_loss_value = validation_loss
            tmp_val_acc_value = validation_accuracy

            ## check for early stopping during training:
            flag_tensor = torch.zeros(1).to(device)
            if dist.get_rank()==0:
                if early_stopper.early_stop(validation_loss):
                    flag_tensor += 1
            dist.all_reduce(flag_tensor)
            if flag_tensor == 1:
                with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                    fout.write(f"Exit condition from early stop on validation loss (earlyStopper patience and minDelta: {early_stopper.get_patience_and_minDelta()})")             
                break
                
        else:
            list_of_validLosses.append(tmp_val_loss_value)
            list_of_validAccs.append(tmp_val_acc_value)
        
        ##Crea (o aggiorna) la figura durante il training:
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
        
        ## 13 e 14 giugno june 2023
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        ##
    with open(os.path.join(path_to_model_dir,"results.txt"),"a") as fout:
                fout.write("---End of training---")
    print("---End of training---")

    # Clean up the process groups:
    dist.destroy_process_group() #TODO

    pass

#%% MAIN

import torch.multiprocessing as mp
if __name__ == '__main__': #main(rank, world_size, causality_awareness, learning_rate, causality_method=None, lehmer_param=None):
    world_size=args.number_of_gpus

    print(f"POSSIBLE COMBINATIONS:\n causality_awareness_method {causality_awareness_method}\n adam_LR: {adam_LR}\n wd: {wd}\n; causality_setting: {CAUSALITY_SETTING}\n MULCAT_CAUSES_OR_EFFECTS: {MULCAT_CAUSES_OR_EFFECTS}\n LEHMER_PARAM: {LEHMER_PARAM}")

    
    for CA_method in causality_awareness_method: #none,max,lehmer
       
        if CA_method is None:
            for lr in adam_LR:
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
                    for lr in adam_LR:
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
                        for lr in adam_LR:
                            for we_de in wd:
                                print(f"Sto lanciando CA lehmer con alpha {alpha} e LR={lr}, wd={we_de}, causality_setting {causality_setting}, con mulcat_causes_or_effects {mulcat_causes_or_effects}")
                                mp.spawn(
                                    main,
                                    args=(world_size, True, lr, we_de, "lehmer", alpha, causality_setting, mulcat_causes_or_effects),
                                    nprocs=world_size
                                )
        else:
            print("errore nel ciclo for per CA_method")