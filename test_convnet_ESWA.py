#%% IMPORT
import os
import argparse

#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-6772vh08 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

import numpy as np
from tqdm.auto import tqdm
import torch

from torchvision.transforms import Compose
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from split_train_val_test import get_or_create_datasetsCSVpaths
##
mydir=os.path.join(os.getcwd(), 'pretrained_models') ## these are the regular ImageNet-trained CV models, like resnets weights.
torch.hub.set_dir(mydir)
os.environ['TORCH_HOME']=mydir
##

parser = argparse.ArgumentParser(description="Testing script for a convnet.")
parser.add_argument("number_of_gpus",type=int,help="The number of GPUs you intend to use")
parser.add_argument("gpus_id",type=str,help="The comma separated list of integers representing the id of requested GPUs - such as '0,1'")
parser.add_argument("path_to_saved_model",type=str,help="The path to the trained model you intend to use")
#
parser.add_argument("SEED",type=int,help="fix seed to set reproducibility, eg, seed=42")
parser.add_argument("EXPERIMENT",type=str,help="cifar10, prostate, imagenette")
parser.add_argument("CONDITIONING_FEATURE",type=str,help="for imagenette is imagenette, for prostate can be (aggressiveness, no_tumour, scanner_vendor, disease_yes_no)")
parser.add_argument("IMAGE_SIZE",type=int,help="imagenette 64, prostate 128")
parser.add_argument("BATCH_SIZE_TEST",type=int,help=" 1 image")
parser.add_argument("MODEL_TYPE",type=str,help="eg, EqualCNN, alexnet, SimpleCNN, EqualCNN, BaseCNN, LightCNN, resnet18, resnet34, resnet50, resnet101, ")

parser.add_argument("causality_setting", type=str, default="cat", help="['cat', 'mulcat', 'mulcatbool']")
parser.add_argument("CAUSALITY_AWARENESS_METHOD", type=str, default=None, help="[None, 'max', 'lehmer']")
parser.add_argument("LEHMER_PARAM", type=int, help="if using Lehmer mean, which power utilize among: [-100,-1,0,1,2,100]")
#21-24 luglio 2023:
parser.add_argument("MULCAT_CAUSES_OR_EFFECTS", type=str, help="if CA, which one to use for causality factors computation: ['causes','effects']")

args = parser.parse_args()

###




###repoducibility:#############
SEED = args.SEED
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
###

print("ATTENTION PLEASE: check that all the parameters passed to this script were indeed the same of those used during the corresponding training of the to-be-tested model")

image_size = args.IMAGE_SIZE

if args.EXPERIMENT == "prostate": #PI-CAI
    from split_train_val_test import get_or_create_datasetsCSVpaths
    dataset_name=""
    CONDITIONING_FEATURE = args.CONDITIONING_FEATURE
    channels = 1
    num_classes = 2
    if CONDITIONING_FEATURE == "aggressiveness": # lesion aggressiveness labels: LG and HG
        dataset_name = "UNFOLDED_DATASET_5_LOW_RESOLUTION_NORMALIZED_GUIDED_CROP_GUIDED_SLICE_SELECTION"
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","cs_les_unfolded.csv") ## versione nuova, con train valid e test tutti uniti
    elif CONDITIONING_FEATURE == "disease_yes_no": 
        dataset_name = "UNFOLDED_DATASET_DISEASE_YES_NO"
        csv_path = os.path.join(os.getcwd(),"dataset_PICAI","csv_files","unfolded_disease_YesNo_balanced.csv") #TODO: la versione bilanciata del precedente: per farlo, ho eliminato (frai pazienti senza tumore) quelli che aveano PSA piu alto (e in seconda chiave di ordinamento quelli che avevano il Volume piu alto), e poi ho riordinato le righe questa volta per pazienteID, quindi non sono stratificate come prima (dove aveo prima i tumorali e poi tutti i non tumorali)
elif args.EXPERIMENT == "breakhis": #TODO 31 oct 2023
    dataset_name=""
    CONDITIONING_FEATURE = args.CONDITIONING_FEATURE
    channels = 3 #rgb
    num_classes = 2 #benign or malignant histology type.
    if CONDITIONING_FEATURE == "aggressiveness": # 
        csv_path = os.path.join(os.getcwd(),"dataset_breakhis","csv_files","breakhis_metadata_400X.csv") ##        
else:
    raise ValueError
print(f"Dataset_name: {dataset_name}\n  csv_path: {csv_path}")


##
if args.EXPERIMENT == "prostate": # prostate picai PI-CAI
        from dataset_creator import Dataset2DSL        

        my_transform_valid_and_test = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        _, _, path_to_test_csv = get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
        dataset_test = Dataset2DSL(csv_path=path_to_test_csv, dataset_name=dataset_name, CONDITIONING_FEATURE=CONDITIONING_FEATURE,transform=my_transform_valid_and_test, use_label=True)   
        
elif args.EXPERIMENT == "breakhis": # 
        from dataset_creator import BREAKHISDataset2D        

        my_transform_valid_and_test = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1) #TODO 14 settembre 2023: Provo a togliere la normalizzazione, perche tanto sono già valori fra [0,1] essendo tensori
        ])
        _, _, path_to_test_csv = get_or_create_datasetsCSVpaths(EXPERIMENT=args.EXPERIMENT, CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path)
        dataset_test = BREAKHISDataset2D(csv_path=path_to_test_csv, cls_type="binary", transform=my_transform_valid_and_test)   

##
causality_setting = args.causality_setting
causality_method = args.CAUSALITY_AWARENESS_METHOD
causality_awareness = causality_method == "max" or causality_method == "lehmer"
LEHMER_PARAM = args.LEHMER_PARAM
MULCAT_CAUSES_OR_EFFECTS = args.MULCAT_CAUSES_OR_EFFECTS
print(f"causality_setting {causality_setting}, causality_method {causality_method}, causality_awareness {causality_awareness}, LEHMER_PARAM {LEHMER_PARAM}, MULCAT_CAUSES_OR_EFFECTS: {MULCAT_CAUSES_OR_EFFECTS}, ......")

number_of_feature_maps = None
if args.MODEL_TYPE=="resnet18":
    
    from networks_attn_learnLM_clean import Resnet18CA_clean # we use it with a fixed, static (not learned) Lehmer param

    model = Resnet18CA_clean(
        dim=image_size,
        channels=channels,
        num_classes=num_classes,            
        is_pretrained=True,
        is_feature_extractor=False,            
        causality_aware=causality_awareness,
        causality_method=causality_method,
        LEHMER_PARAM=LEHMER_PARAM,
        causality_setting=causality_setting,
        visual_attention=False,
        MULCAT_CAUSES_OR_EFFECTS=MULCAT_CAUSES_OR_EFFECTS
        )
    print("-#-#-#: intialized a Resnet18CA model from networks_attn")#: pretrainedweights ({is_pretrained}), usedAsFeatureExtractor ({is_feature_extractor}), causality-aware ({causality_awareness})") 

elif args.MODEL_TYPE=="resnet18_ablation":  
    ##--- ablation study 
    from networks_attn_learnLM_clean_ABL import Resnet18CA_clean # TODO ablation study
    model = Resnet18CA_clean(
        dim=image_size,
        channels=channels,
        num_classes=num_classes,            
        is_pretrained=True,# is_pretrained=False,
        is_feature_extractor=False,            
        causality_aware=causality_awareness,
        causality_method=causality_method,
        LEHMER_PARAM=LEHMER_PARAM,
        causality_setting=causality_setting,
        visual_attention=False, 
        MULCAT_CAUSES_OR_EFFECTS=MULCAT_CAUSES_OR_EFFECTS
        )
    print("-#-#-#: intialized a Resnet18CA model from networks_attn_learnLM_clean_ABL")#: pretrainedweights ({is_pretrained}), usedAsFeatureExtractor ({is_feature_extractor}), causality-aware ({causality_awareness})") 


device="cuda"

model = model.to(device)
path_to_savedModel = args.path_to_saved_model #E.g., results/saved_models/cifar10_user/SimpleCNN/20230516111723_400epochs_32imagesize_1000batchsize_0.0008learningrate/epoch80_bestValid <-- .pth file
state_dict = torch.load(path_to_savedModel,map_location=device)
# create new OrderedDict that does not contain `module.` ## Since we trained the model with Torch DDP setting, the model is encapsulated in a module object...
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove the seven char of `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
#print(f"A model trained in DDP fashion has been state_dict loaded in a non-DDP model, and its device is now: {next(model.parameters()).device}, nice!")

results_folder = path_to_savedModel.replace("saved_models","inference")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

######## prepare the dataset
dataloader_test = DataLoader(dataset_test, batch_size=1, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)


######## evaluate
model.eval()
accuracy_test = 0.0
total_test = 0.0

ytrue_validation = []
yscore_validation = []

with torch.no_grad():
    for images_v, labels_v in tqdm(dataloader_test):

        images_v = images_v.to(device)

        ytrue_validation.append(labels_v.item()) #before sending them to the cuda device, keep track of their value in the list
        labels_v = labels_v.to(device)
        
        outputs_v, _ = model(images_v)

        #
        yscore_validation.append(outputs_v.detach().cpu().numpy()[:,1])
        #

        # the class with the highest energy is what we choose as prediction
        predicted = torch.argmax(outputs_v, 1)
        total_test += labels_v.size(0)
        count_correct_guess = (torch.eq(predicted,labels_v)).sum().item()
        accuracy_test += count_correct_guess

accuracy_test = 100 * (accuracy_test / total_test)
from sklearn.metrics import roc_auc_score
auroc_softmax = roc_auc_score(ytrue_validation, yscore_validation)
print(f"test Accuracy: {accuracy_test}; test AUROC: {auroc_softmax}")

with open(os.path.join(results_folder,"out.txt"),"w") as fout:
    fout.write(f"test Accuracy: {accuracy_test}; test AUROC: {auroc_softmax}")

with open(os.path.join(results_folder,"settings_of_this_experiment.txt"),"w") as fout:
            fout.write(f" csv_path: {csv_path}\n\
            dataset_name: {dataset_name}\n\
            SEED: {SEED}\n \
            dataset_name: {dataset_name}\n \
            number of image classes: {num_classes}\n \
            channels: {channels}\n \
            image_size: {image_size}\n \
            batch_size_test: 1\n \
            dataloader_test of size: {len(dataloader_test)} batches\n \
            model_type: {args.MODEL_TYPE}\n \
                --- if EqualCNN, No. f maps: {number_of_feature_maps}\n \
                --- if ResNext, which type: \n \
            is_pretrained: True \n \
            is_feature_extractor: False \n \
            causality_aware: {causality_awareness} \n \
                ---causality_method: {causality_method} \n \
                ---LEHMER PARAM (alpha, or p): {LEHMER_PARAM} \n \
                ---causality_setting: {causality_setting} \n \
            {model}")