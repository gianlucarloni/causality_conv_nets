#IMPORT
import random
import numpy as np
import torch
import os

#DEFINE

#EXPERIMENT = "cifar10" #"prostate" "imagenette"
EXPERIMENT = "imagenette" # user serve per fare creare la cartella dove mettere i modelli che usavano fromscratch cnn, invece senza user per quelli che necessitano root (eg, resnet)

if EXPERIMENT == "imagenette":
    dataset_name = "imagenette"
    CONDITIONING_FEATURE = "imagenette" # user serve per fare creare la cartella dove mettere i modelli che usavano fromscratch cnn, invece senza user per quelli che necessitano root (eg, resnet)
    CAUSALITY_AWARE = False #False, True
    INFERENCE_MODE = False # True, False
    channels = 3 # 
    image_size = 32 #227 #128 #224 #112 #TODO
    batch_size_train = 200 #200 #20 #10 #1000
    batch_size_valid = 1 #1
    batch_size_test = 1
    epochs = 7 #400 #2000 #6 #100 #\TODO 
    adam_LR = 8e-4 #1e-3 #1e-4 1e-5 TODO
    save_model_every_N_epochs = 10 #25 #50 #round(epochs/4) TODO
    num_classes = 10
    loss_type="CrossEntropyLoss" #CrossEntropyLoss

    model_type="EqualCNN" #alexnet, resnet18, resnet34, resnet50, resnet101, SimpleCNN, EqualCNN, BaseCNN, LightCNN
    is_pretrained = False
    is_feature_extractor = False # SOLO SE USI 3 CANALI HA SENSO, ALTRIMENTI PER FORZA LA RIALLENI TUTTA

    causality_method="lehmer" #"lehmer" #"max"
    LEHMER_PARAM=100 #0 #-1, 2, inf, -inf

    if not model_type.startswith("resnet"):
        EXPERIMENT = EXPERIMENT+"_user"
        CONDITIONING_FEATURE = CONDITIONING_FEATURE+"_user"
        dataset_name = dataset_name+"_user"
    
    csv_path=""
    train_root_path=os.path.join(os.getcwd(),"imagenette2","train")
    val_root_path=os.path.join(os.getcwd(),"imagenette2","val")
    test_root_path=os.path.join(os.getcwd(),"imagenette2","test")
    

elif EXPERIMENT == "cifar10_user":
    CONDITIONING_FEATURE = "cifar10_user" # user serve per fare creare la cartella dove mettere i modelli che usavano fromscratch cnn, invece senza user per quelli che necessitano root (eg, resnet)
    CAUSALITY_AWARE = True #False, True
    INFERENCE_MODE = False # True, False
    channels = 3 # 
    image_size = 32 #TODO
    batch_size_train = 1000 #1000
    batch_size_valid = 200
    batch_size_test = 1
    epochs = 200 #400 #2000 #6 #100 #\TODO 
    adam_LR = 1e-4 #1e-3 #1e-4 TODO
    save_model_every_N_epochs = 14 #25 #50 #round(epochs/4) TODO
    num_classes = 10
    loss_type="CrossEntropyLoss" #CrossEntropyLoss
    model_type="SimpleCNN" #resnet18, resnet34, resnet50, resnet101, SimpleCNN, BaseCNN, LightCNN

    dataset_name=""
    csv_path=""
    if CONDITIONING_FEATURE == "cifar10_user": 
        dataset_name = "cifar10_user"

elif EXPERIMENT == "prostate":
    CONDITIONING_FEATURE = "disease_yes_no" # disease (yes/no), scanner_vendor (siemens/philips), prostate volume (low/high), age(low/high), psa (low/high)
    CAUSALITY_AWARE = False
    channels = 1 # GRAYSCALE, L, SINGLE-CHANNEL
    image_size = 128 #128 #128 #256 #384 #TODO
    batch_size_train = 200
    batch_size_valid = 2
    batch_size_test = 1
    epochs = 100 #2000 #6 #100 #\TODO 
    adam_LR = 1e-4 #1e-3 #1e-4 TODO
    save_model_every_N_epochs = 10 #25 #50 #round(epochs/4) TODO
    num_classes = 2 #or None ---> number of image classes, eg, LG and HG in prostate clinically significatn images.
    loss_type="CrossEntropyLoss" #CrossEntropyLoss
    model_type="resnet18" #resnet18, resnet34, resnet50, resnet101, SimpleCNN, BaseCNN, LightCNN
    is_pretrained = False
    is_feature_extractor = False # SOLO SE USI 3 CANALI HA SENSO, ALTRIMENTI PER FORZA LA RIALLENI TUTTA


    dataset_name=""
    csv_path=""
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


    

    if not model_type.startswith("resnet"):
        EXPERIMENT = EXPERIMENT+"_user"
        CONDITIONING_FEATURE = CONDITIONING_FEATURE+"_user"
        dataset_name = dataset_name+"_user"
#dataset_name = "UNFOLDED_DATASET_5_LOW_RESOLUTION_NORMALIZED_GUIDED_CROP_PROSTATEGUIDED_SLICE_SELECTION"



print(f"SETTINGS\n  dataset_name: {dataset_name}\n  csv_path: {csv_path}")


SEED = 42
#SEED = 1 #1,2,3,4,5,6,7,8,9,10 #TODO 11 April: circa 200/400 inferenze alla volta, cambiando seed

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g_cuda = torch.Generator(device="cuda").manual_seed(SEED)

g_cpu = torch.Generator(device="cpu").manual_seed(SEED)

print(f"LAUNCHING EXPERIMENT FOR {CONDITIONING_FEATURE} WITH seed {SEED} and: image_size {image_size} of {channels} channels, \
in batch_size (train,val,test) of ({batch_size_train,batch_size_valid,batch_size_test}) for {epochs} epochs, \
and LR {adam_LR}, saving model every {save_model_every_N_epochs} epochs, \
number of image classes: {num_classes}, loss-type={loss_type}")

