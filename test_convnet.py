#%% IMPORT
import os
import argparse

#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-6772vh08 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

from torchvision.transforms import Compose #, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import random
from networks import SimpleCNN
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid #, save_image
from torchvision import transforms
from dataset_creator import Dataset2DSL
from pathlib import Path
from collections import OrderedDict

#from settings import SEED, image_size, channels, num_classes,
#from settings import dataset_name, csv_path, CONDITIONING_FEATURE, channels, image_size, batch_size_valid, batch_size_test, num_classes, loss_type




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











""" NUM_GPUS = args.number_of_gpus #1 #2 TODO
if NUM_GPUS == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_id
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    I_HAVE_CAUSED_AN_ERROR_by_Gianluca """




###
###repoducibility:#############
SEED = args.SEED
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
###




print("ATTENZIONE: Hai controllato che tutti i parametri qui hardcoded siano uguali a quelli usati durante il training di questo modello?")



image_size = args.IMAGE_SIZE

if args.EXPERIMENT == "imagenette":
    dataset_name = "imagenette"
    CONDITIONING_FEATURE = "imagenette"
    channels = 3
    num_classes = 10
    test_root_path=os.path.join(os.getcwd(),"imagenette2","test")
elif args.EXPERIMENT == "prostate":
    from split_train_val_test import get_or_create_datasetsCSVpaths
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



##
if (dataset_name=="imagenette_user" or dataset_name=="imagenette"):
        
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
    
    my_transform_valid_and_test = Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_test = ImageFolder(root=test_root_path, transform=my_transform_valid_and_test)
elif args.EXPERIMENT == "prostate": # prostate
        from dataset_creator import Dataset2DSL

        

        my_transform_valid_and_test = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

        _, _, path_to_test_csv = get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
        dataset_test = Dataset2DSL(csv_path=path_to_test_csv, dataset_name=dataset_name, CONDITIONING_FEATURE=CONDITIONING_FEATURE,transform=my_transform_valid_and_test, use_label=True)   
        

##



causality_setting = args.causality_setting
causality_method = args.CAUSALITY_AWARENESS_METHOD

causality_awareness = causality_method == "max" or causality_method == "lehmer"

LEHMER_PARAM = args.LEHMER_PARAM

MULCAT_CAUSES_OR_EFFECTS = args.MULCAT_CAUSES_OR_EFFECTS

print(f"causality_setting {causality_setting}, causality_method {causality_method}, causality_awareness {causality_awareness}, LEHMER_PARAM {LEHMER_PARAM}, MULCAT_CAUSES_OR_EFFECTS: {MULCAT_CAUSES_OR_EFFECTS}, ......")

number_of_feature_maps = None

if args.MODEL_TYPE=="EqualCNN":

    number_of_feature_maps = 256 #32 #32 #32, 64, 128
    from networks import EqualCNN
    model = EqualCNN( # number of channels (ftmaps) 3>16>32>32>32>32>32>32... remains equally 32, straight
        NUMB_FTRS_MPS=number_of_feature_maps,
        dim=image_size,
        channels=channels,
        num_classes=num_classes,
        causality_aware=causality_awareness,
        causality_method=causality_method,
        LEHMER_PARAM=LEHMER_PARAM
    )
elif args.MODEL_TYPE=="resnet18":
    # if causality_setting == "mulcatbool":
    #     from networks2 import Resnet18CA
    #     causality_setting = "mulcat" #lo richiamo mulcat per far funzionare il codice resnet18ca, tanto ora è la rete giusta

    # elif (causality_setting == "cat") or (causality_setting == "mulcat"):
    #     from networks import Resnet18CA
    # from networks import Resnet18CA
    
    ## TODO 17 luglio 2023:
    # from networks_attn import Resnet18CA
    from networks_attn_ablation import Resnet18CA #TODO 28 july 2023
    model = Resnet18CA(
            dim=image_size,
            channels=channels,
            num_classes=num_classes,
            is_pretrained=False,
            is_feature_extractor=False,            
            causality_aware=causality_awareness,
            causality_method=causality_method,
            LEHMER_PARAM=LEHMER_PARAM,
            causality_setting=causality_setting,
            visual_attention=False, #TODO 12 july 2023
            MULCAT_CAUSES_OR_EFFECTS=MULCAT_CAUSES_OR_EFFECTS #TODO 21-24 july 2023
            )



device="cuda"
#print(f"device={device}")

model = model.to(device)
path_to_savedModel = args.path_to_saved_model #EG, results/saved_models/cifar10_user/SimpleCNN/20230516111723_400ep_32is_1000bs_0.0008LR/ep80_betterValid
state_dict = torch.load(path_to_savedModel,map_location=device)
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
#print(f"A model trained in DDP fashion has been state_dict loaded in a non-DDP model, and its device is now: {next(model.parameters()).device}, nice!")




results_folder = path_to_savedModel.replace("saved_models","inference")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


"""my_transform = Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 my_transform = Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
]) """


denormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
                                ])




######## prepare the dataset

dataloader_test = DataLoader(dataset_test, batch_size=1, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)

#path_to_train_csv, path_to_val_csv, path_to_test_csv = get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
""" path_to_train_csv, path_to_val_csv, _ = get_or_create_datasetsCSVpaths(CONDITIONING_FEATURE=CONDITIONING_FEATURE, csv_path=csv_path, testset_size=0.2, validset_size=0.15)
dataset_train = Dataset2DSL(csv_path=path_to_train_csv, dataset_name=dataset_name, transform=my_transform, use_label=True)
dataset_val = Dataset2DSL(csv_path=path_to_val_csv, dataset_name=dataset_name, transform=my_transform, use_label=True)
dataset_test = Dataset2DSL(csv_path=path_to_test_csv, dataset_name=dataset_name, transform=my_transform, use_label=True)
"""
## 

""" from torchvision.datasets import CIFAR10
classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataset_val = CIFAR10(root='./data', train=False, download=True, transform=my_transform)
print(f"Using {dataset_name}, valid dataset: len={len(dataset_val)}")

dataloader_valid = DataLoader(dataset_val, batch_size=batch_size_valid, pin_memory=False, num_workers=0, drop_last=False, shuffle=False)
print(f"dataloader_valid of size {len(dataloader_valid)} batches, each of {batch_size_valid}")
 """



#########################

model.eval()
accuracy_test = 0.0
total_test = 0.0

with torch.no_grad():
    for images_v, labels_v in tqdm(dataloader_test):
        images_v = images_v.to(device)
        labels_v = labels_v.to(device)
        
        outputs_v, _ = model(images_v)

        # the class with the highest energy is what we choose as prediction
        predicted = torch.argmax(outputs_v, 1)
        total_test += labels_v.size(0)
        count_correct_guess = (torch.eq(predicted,labels_v)).sum().item()
        accuracy_test += count_correct_guess

accuracy_test = 100 * (accuracy_test / total_test)
print(f"test Accuracy: {accuracy_test}")
with open(os.path.join(results_folder,"out.txt"),"w") as fout:
    fout.write(f"test Accuracy: {accuracy_test}")












############# SHOW SOME IMAGES AND THE CORRESPONDING PREDICTED CLASSES ############
""" model = SimpleCNN(dim=image_size, channels=channels,num_classes=num_classes, causality_aware = True, inference_mode=True)
model = model.to(device)
path_to_savedModel = args.path_to_saved_model
state_dict = torch.load(path_to_savedModel,map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict) """

# def imshow(img_tensor, actual_labels_list=None, predicted_labels_list=None):
    
    
#     if args.EXPERIMENT == "prostate":
#         img_tensor = img_tensor / 2 + 0.5     # unnormalize, since they underwent the transforms.Lambda(lambda t: (t * 2) - 1)
#     else:
#         img_tensor = denormalize(img_tensor)   #

#     npimg = img_tensor.cpu().numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     if (actual_labels_list is not None) and (predicted_labels_list is not None):
#         plt.title(f"Actual labels: {str(actual_labels_list)}\nPredicted labels: {str(predicted_labels_list)}")
#     plt.show()

# # get some random images
# dataiter = iter(dataloader_test)
# images, labels = next(dataiter)
# labels = labels.tolist()
# batch_labels = [classes[labels[j]] for j in range(images.size(0))]
# # and compute the respective predicted classes:
# images = images.to(device)

# model.eval()
# #outputs, feature_maps = model(images)
# outputs, _ = model(images) #outputs, causality_maps

# predicted = torch.argmax(outputs, 1)
# predicted = predicted.tolist()
# predicted_labels = [classes[predicted[j]] for j in range(images.size(0))]

# # # show images
# # plt.figure()
# # imshow(make_grid(images), batch_labels, predicted_labels)
# # plt.savefig(os.path.join(results_folder,"some_tested_images.pdf"))
# # plt.close()
# # # print labels

# print(batch_labels)
# print(predicted_labels)




########################
# save feature maps for some images
#print(f"Size() pool2: {feature_maps['pool2'].size()}") #[batch, 16, 8, 8]
""" n_batch = feature_maps["pool2"].size()[0] #3
n_channels = feature_maps["pool2"].size()[1] #of the feature maps, 16
n_TOT = (n_channels+1)*n_batch
i=0
plt.figure()
while i < n_TOT:
    plt.subplot(n_channels+1, n_batch, i+1)
    if i < n_batch:
        plt.imshow(np.transpose(denormalize(images[i,:,:]).cpu().numpy(), (1, 2, 0)))
        plt.axis('off')
    else:
        plt.imshow(feature_maps["pool2"][i//n_batch, i//n_channels,:,:].cpu().numpy(),cmap="summer")
        plt.axis('off')
plt.show()
plt.savefig(os.path.join(results_folder,"some_inference_images_with_featuremaps.pdf"))
plt.close() """




""" for which_img_in_batch in tqdm(range(batch_size_valid)):
    for pooling_level in ["pool1","pool2"]:
        b1 = feature_maps[pooling_level][which_img_in_batch,:,:,:]
        b1_grid = make_grid(b1,nrow=b1.size()[0])
        b1_grid_npy = b1_grid.cpu().detach().numpy() 
        b1_grid_npy = np.transpose(b1_grid_npy, (1, 2, 0)) 
        plt.figure(figsize=(15,15))
        for i in range(b1.size()[0]):      
            plt.subplot(b1.size()[0],1,i+1)
            plt.imshow(b1_grid_npy[:,:,i],cmap="jet")
            plt.axis('off')
            plt.title(f"ftrmp {i+1}",fontsize = 9)
        plt.show()
        plt.savefig(os.path.join(results_folder,f"im{which_img_in_batch}inbatch_{pooling_level}_featuremaps.pdf"))
        plt.close() """