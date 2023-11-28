import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import time
import random

## Info (november 2023)
# The name of this script indicates the possibility to insert the vision attention mechanism (_attn_)
# and the learning of the Lehmer Mean power (_learnLM_). However, these are improvements for future work,
# and therefore they are here not completely coded nor tested. Please use the settings as in the main published paper.

##########################################################################################
class STEFunction(torch.autograd.Function):
    '''
    https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
    '''
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    '''
    https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
    '''
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x
##########################################################################################

class CausalityMapBlock(nn.Module):
    def __init__(self, elems, causality_method, fixed_lehmer_param=None):
        '''
        elems: (int) square of the number of elements in each featuremap at the conv bottleneck, (eg: F=3x3=9-->81)
        causality_method: (str) max or lehmer
        fixed_lehmer_param: (float) power of the Lehmer Mean, such as -2.0, 0.0, or 1.0.
            When it is not passed - it is None - this makes the lehmer_seed learnable by backprop with torch.nn.Parameter...(to be continued)
        '''
        super(CausalityMapBlock, self).__init__()
        self.elems = elems
        self.causality_method =causality_method
        if self.causality_method=="lehmer":
            if fixed_lehmer_param is not None:
                self.lehmer_seed=float(fixed_lehmer_param)
            else:
                self.lehmer_seed=torch.nn.Parameter(torch.tensor([0.0],device="cuda")) 
            print(f"INIT - CausalityMapBlock: LEHMER with self.lehmer_seed={self.lehmer_seed}")
        else: #"max"
            print(f"INIT - CausalityMapBlock self.causality_method: {self.causality_method}")

    def forward(self,x): #(bs,k,n,n)   
        
        if torch.isnan(x).any():
            print(f"...the current feature maps object contains NaN")
            raise ValueError
        maximum_values = torch.max(torch.flatten(x,2), dim=2)[0]  #flatten: (bs,k,n*n), max: (bs,k) 
        MAX_F = torch.max(maximum_values, dim=1)[0]  #MAX: (bs,) 
        x_div_max=x/(MAX_F.unsqueeze(1).unsqueeze(2).unsqueeze(3) +1e-8) #TODO added epsilon; #implement batch-division: each element of each feature map gets divided by the respective MAX_F of that batch
        x = torch.nan_to_num(x_div_max, nan = 0.0)

        ## After having normalized the feature maps, comes the distinction between the method by which computing causality
        if self.causality_method == "max": #Option 1 : max values

            sum_values = torch.sum(torch.flatten(x,2), dim=2)
            if torch.sum(torch.isnan(sum_values))>0:
                sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
           
            maximum_values = torch.max(torch.flatten(x,2), dim=2)[0]  
            mtrx = torch.einsum('bi,bj->bij',maximum_values,maximum_values) #batch-wise outer product, the max value of mtrx object is 1.0
            tmp = mtrx/(sum_values.unsqueeze(1) +1e-8) #TODO added epsilon
            causality_maps = torch.nan_to_num(tmp, nan = 0.0)

        elif self.causality_method == "lehmer": #Option 2 : Lehmer mean   
           
            x = torch.flatten(x,2) # [b,k,n*n], eg [16,512,8*8]
            #compute the outer product of all the pairs of flattened featuremaps.
            # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
            cross_matrix = torch.einsum('bmi,bnj->bmnij', x, x) #eg, [16,512,512,64,64]  symmetric values 
            cross_matrix = cross_matrix.flatten(3) #eg, [16,512,512,4096]

            # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
            # first, compute the two powers of the cross matrix
            p_plus_1_powers = torch.nan_to_num(torch.pow(cross_matrix+1e-8, self.lehmer_seed+1)) #eg, [16,512,512,4096]
            
            p_powers = torch.nan_to_num(torch.pow(cross_matrix+1e-8, self.lehmer_seed)) #eg, [16,512,512,4096]

            numerators = torch.nan_to_num(torch.sum(p_plus_1_powers, dim=3)) + 1e-8   
            denominators = torch.nan_to_num(torch.sum(p_powers, dim=3)) + 1e-8  #eg, [16,512,512]
            lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators)) + 1e-8    #[bs,k,k]
            #############            
            # then the lehmer denominator of the causality map:
            # it is the lehemr mean of the single feature map, for all the feature maps by column

            p_plus_1_powers_den = torch.nan_to_num(torch.pow(x.abs()+1e-8, self.lehmer_seed+1))

            p_powers_den = torch.nan_to_num(torch.pow(x.abs()+1e-8, self.lehmer_seed)) ##TODO

            numerators_den = torch.nan_to_num(torch.sum(p_plus_1_powers_den, dim=2)) + 1e-8   
            denominators_den = torch.nan_to_num(torch.sum(p_powers_den, dim=2)) + 1e-8   
            lehmer_denominator = torch.nan_to_num(torch.div(numerators_den,denominators_den), nan=0) + 1e-8    
            #and finally obtain the causality map values by computing the division
            causality_maps = torch.nan_to_num(torch.div(lehmer_numerators, lehmer_denominator.unsqueeze(1)), nan=0)
            
        else:
            print(self.causality_method) # we implemented only MAX and LEHMER options, so every other case is a typo/error
            raise NotImplementedError

        # print(causality_maps)    
        return causality_maps

##########################################################################################
class CausalityFactorsExtractor(nn.Module):
    def __init__(self, causality_direction, causality_setting):
        
        super(CausalityFactorsExtractor, self).__init__()
        self.causality_direction = causality_direction #eg, "causes" or "effects"
        self.causality_setting = causality_setting #eg, "mulcat,mulcatbool,mul,mulbool"
        self.STE = StraightThroughEstimator() #Bengio et al 2013
        self.relu = nn.ReLU()
        print(f"INIT - CausalityFactorsExtractor: self.causality_direction={self.causality_direction}, self.causality_setting={self.causality_setting}")

    def forward(self, x, causality_maps):
        '''
        x [bs, k, h, w]: the feature maps from the original (regular CNN) branch;
        causality_maps [bs, k, k]: the output of a CausalityMapsBlock().

        By leveraging algaebric transformations and torch functions, we efficiently extracts the causality factors with few lines of code.
        '''
        triu = torch.triu(causality_maps, 1) #upper triangular matrx (excluding the principal diagonal)
        tril = torch.tril(causality_maps, -1).permute((0,2,1)).contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper

        e = tril - triu
        e = self.STE(e)
        e = e.permute((0,2,1))

        f = triu - tril
        f = self.STE(f)
        bool_matrix = e + f #sum of booleans is the OR logic

        by_col = torch.sum(bool_matrix, 2)
        by_row = torch.sum(bool_matrix, 1)

        if self.causality_direction=="causes":
            if self.causality_setting == "mulcat" or self.causality_setting == "mul":
                causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
                causes_mul_factors=self.relu(causes_mul_factors)

            elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
                # causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                causes_mul_factors = self.STE(by_col - by_row) # differentiable version for torch autograd

        elif self.causality_direction=="effects":
            if self.causality_setting == "mulcat" or self.causality_setting == "mul":
                causes_mul_factors = by_row - by_col # 
                causes_mul_factors=self.relu(causes_mul_factors)

            elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
                # causes_mul_factors = 1.0*((by_row - by_col)>0) # 
                causes_mul_factors = self.STE(by_row - by_col) # differentiable version for torch autograd
        
        else:
            print("Personal error: unrecognised self.causality_direction")
            raise ValueError       
        
        ## Directly return the "attended" ("causally"-weighted) version of x
        return torch.einsum('bkmn,bk->bkmn', x, causes_mul_factors) #multiply each (rectified) factor for the corresponding 2D feature map, for every minibatch

##########################################################################################
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()        
    def forward(self, x):
        return x

##########################################################################################
class Resnet18CA_clean(nn.Module):
    def __init__(self, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            super(Resnet18CA_clean, self).__init__()
            self.img_size = dim
            self.channels = channels
            self.num_classes = num_classes
            self.is_pretrained = is_pretrained #True
            self.is_feature_extractor = is_feature_extractor #False, we do not want to use it as freezed extractor, but train it again.
            
            self.causality_aware = causality_aware
            self.causality_method = causality_method
            self.causality_setting = causality_setting #

            if LEHMER_PARAM is not None: #when specified, e.g. as a flot, it is assigned.
                self.LEHMER_PARAM = LEHMER_PARAM

            # self.visual_attention = visual_attention #boolean
            self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS

            if self.is_pretrained:
                print("is_pretrained=True-------->loading imagenet weights")
                model = resnet18(pretrained=True)
            else:
                print("is_pretrained=False---------->init random weights")
                model = resnet18()

            if self.channels == 1:
                model.conv1 = nn.Conv2d(1, 64,kernel_size=7, stride=2, padding=3, bias=False, device='cpu') #output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
            elif self.channels==3:
                if self.is_feature_extractor:
                    for param in model.parameters(): #freeze the extraction layers
                        param.requires_grad = False
            
            self.starting_block = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool) #output size is halved           
            self.layer1 = model.layer1 #output size is halved
            self.layer2 = model.layer2 #output size is halved
            self.layer3 = model.layer3 #output size is halved
            self.layer4 = model.layer4 #output size is halved

            model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get 2D feature maps (e.g. of size 7x7, 3x3, etc.)       
            model.fc = Identity() # Cancel classification layer to get only feature extractor
            self.ending_block = nn.Sequential(model.avgpool, model.fc)            

            self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.
            self.last_ftrmap_number = 512 #It is 512 for ResNet18

            if self.causality_aware:
                ## initialize the modules for causality-driven networks
                
                # if LEHMER_PARAM is not None:
                #     self.causality_map_extractor = CausalityMapBlock(elems=self.last_ftrmap_number, causality_method=self.causality_method, fixed_lehmer_param = self.LEHMER_PARAM)
                # else:
                #     self.causality_map_extractor = CausalityMapBlock(elems=self.last_ftrmap_number, causality_method=self.causality_method)
                print(f"It is Ablation: we do not initialize any CausalityMapBlock!")
                
                # self.causality_factors_extractor = CausalityFactorsExtractor(self.MULCAT_CAUSES_OR_EFFECTS, causality_setting)
                print(f"It is Ablation: we do not initialize any CausalityFactorsExtractor!")
                
                ## and then set the classifier dimension, accordingly
                if self.causality_setting == "cat": #[1, n*n*k + k*k]
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes)
                elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
                    self.classifier = nn.Linear(2 * self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)
                elif self.causality_setting == "mul" or self.causality_setting == "mulbool":
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)
           
            else: #regular, not causally driven network                    
                # if self.visual_attention: #Per ora, i valori sono scritti a mano, considerando attn2_4 e attn3_4 intanto solo per la versione non causale; poi preparare il codice anche per quella causale quindi mettere visual_attention anche nel IF sopra
                #     self.classifier = nn.Linear(128*12*12 + 256*6*6 + 512*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
                # else:
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)


            # self.STE = StraightThroughEstimator() #Bengio et al 2013
            self.softmax = nn.Softmax(dim=1) #
            # self.softmax = nn.LogSoftmax(dim=1) #

    def forward(self, x):
        if torch.isnan(x).any():
            print("Personal error: FORWARD - the input was corrupted with NaN")
            raise ValueError

        x = self.starting_block(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x = self.ending_block(x_layer4)
      
        if torch.isnan(x).any():
            print("Personal error: FORWARD - after passing through the feature extractor (conv net), x was corrupted with NaN")
            raise ValueError
        
        if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]:
            x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size))

        causality_maps = None #initialized to none

        if self.causality_aware:
            #TODO ablation: create random cmap instead of computing it
            # causality_maps = self.causality_map_extractor(x)              
                 
            if self.causality_setting == "cat":
                    causality_maps = torch.rand(size=(int(x.size(0)),self.last_ftrmap_number*self.last_ftrmap_number), device=x.get_device()) * (1 + 1e-6)  # la versione già flattenata
                    causality_maps.clamp_(0, 1)
                    x = torch.cat((torch.flatten(x, 1), causality_maps), dim=1)            
            elif self.causality_setting == "mulcat" or self.causality_setting == "mulcatbool": #need to concatenate the x_c to the actual original features x
                    # x_c = self.causality_factors_extractor(x, causality_maps)        
                    #             
                    ##TODO: in ablation study, x_c is obtained by multiplying actual features and random mul_factors
                    causes_mul_factors = F.relu(torch.randint(low=-(self.last_ftrmap_number-1), high=self.last_ftrmap_number, size=(x.size(0), self.last_ftrmap_number),device=x.get_device())) 
                    if self.causality_setting == "mulcatbool":
                        causes_mul_factors = 1.0*((causes_mul_factors)>0)                    
                    x_c = torch.einsum('bkmn,bk->bkmn', x, causes_mul_factors)                    
                    x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version  

        else: #traditional, non causal:
            raise ValueError  #in ABLATION mode, the code should never enter this line since ablation is computed for causality-driven nets only.
            x = torch.flatten(x, 1) # flatten all dimensions except batch      
   
        x = self.classifier(x)
        x = self.softmax(x) 

        return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
    






























######## previous version, do not use: please utilize the clean version of the resnet18 setup (above)
# class Resnet18CA(nn.Module):
#     def __init__(self, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
#             super(Resnet18CA, self).__init__()
#             self.img_size = dim
#             self.channels = channels
#             self.num_classes = num_classes
#             self.is_pretrained = is_pretrained
#             self.is_feature_extractor = is_feature_extractor
            
#             self.causality_aware = causality_aware
#             self.causality_method = causality_method
#             # self.LEHMER_PARAM = LEHMER_PARAM
#             self.causality_setting = causality_setting #

#             self.visual_attention = visual_attention #boolean
#             self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

#             if self.is_pretrained:
#                 model = resnet18(pretrained=True)
#             else:
#                 model = resnet18()

#             if self.channels == 1:
#                 model.conv1 = nn.Conv2d(1, 64,kernel_size=7, stride=2, padding=3, bias=False, device='cpu') #output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
#             elif self.channels==3:
#                 if self.is_feature_extractor:
#                     for param in model.parameters(): #freeze the extraction layers
#                         param.requires_grad = False
            
#             ## creating the structure of our custom model starting from the original building blocks of resnet:
#             # starting block, layer1, layer2, layer3, layer4, ending block, and classifier

#             self.starting_block = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool) #output size is halved
#             self.layer1 = model.layer1 #output size is halved
#             self.layer2 = model.layer2 #output size is halved
#             self.layer3 = model.layer3 #output size is halved
#             self.layer4 = model.layer4 #output size is halved
#             #here, outputsize is the original one divided by 32.

#             model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
#             model.fc = Identity() # Cancel classification layer to get only feature extractor
#             self.ending_block = nn.Sequential(model.avgpool, model.fc)
            
#             ##self.features = model
#             if self.visual_attention:
#                 ## define the attention blocks
#                 self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
#                 self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

#             self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.
#             self.last_ftrmap_number = 512 #512 for ResNet18

#             if self.causality_aware:

#                 self.causality_map_extractor = CausalityMapBlock(elems=self.last_ftrmap_size**4, causality_method=self.causality_method)

#                 print(f"causality_map_extractor LEAFS:")
#                 print(self.causality_map_extractor.mask.is_leaf)
#                 print()

#                 if self.causality_setting == "cat": #[1, n*n*k + k*k]
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes)
#                 elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * self.last_ftrmap_number, self.num_classes)
#                 elif self.causality_setting == "mul" or self.causality_setting == "mulbool": #TODO 18 settembre
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)
#             else:
#                 if self.visual_attention: #TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
#                     self.classifier = nn.Linear(128*12*12 + 256*6*6 + self.last_ftrmap_number*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
#                 else:
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

#             self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added

#     def forward(self, x):
#         if torch.isnan(x).any():
#             l_ingresso_era_gia_corrotto_con_nan
#         # print(f"FORWARD - x (input): {x.requires_grad}")

#         # print(f"x[0] (forward):\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
#         x = self.starting_block(x)
#         # print(f"x[0] (starting_block):\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
#         x_layer1 = self.layer1(x)
#         # print(f"x_layer1  (layer1):\t MIN:{x_layer1[0].min()}\t MAX:{x_layer1[0].max()}\t AVG:{torch.mean(x_layer1[0].float())}\t MED:{torch.median(x_layer1[0].float())}")
#         x_layer2 = self.layer2(x_layer1)
#         # print(f"x_layer2(x_layer2):\t MIN:{x_layer2[0].min()}\t MAX:{x_layer2[0].max()}\t AVG:{torch.mean(x_layer2[0].float())}\t MED:{torch.median(x_layer2[0].float())}")
#         x_layer3 = self.layer3(x_layer2)
#         # print(f"x_layer3(x_layer3):\t MIN:{x_layer3[0].min()}\t MAX:{x_layer3[0].max()}\t AVG:{torch.mean(x_layer3[0].float())}\t MED:{torch.median(x_layer3[0].float())}")
#         x_layer4 = self.layer4(x_layer3)
#         # print(f"x_layer4(x_layer4):\t MIN:{x_layer4[0].min()}\t MAX:{x_layer4[0].max()}\t AVG:{torch.mean(x_layer4[0].float())}\t MED:{torch.median(x_layer4[0].float())}")
#         x = self.ending_block(x_layer4)
#         # print(f"x at -ending_block:\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
        
#         # compute the attention probabilities:
#         # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
#         # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

#         #check nan
#         if torch.isnan(x).any():
#             corrotto_con_nan
#         if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]:
#             x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size))

#         causality_maps = None #initialized to none

#         if self.causality_aware:
#             x, causality_maps = self.causality_map_extractor(x)
#             # print(f"FORWARD - x (causality_map_extractor): {x.requires_grad}")
#             # print(f"FORWARD - causality_maps (causality_map_extractor): {causality_maps.requires_grad}")
            

#             if self.causality_setting == "cat":
#                 x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
#             # elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"):
#             elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool") or (self.causality_setting == "mul") or (self.causality_setting == "mulbool"):
#                 b = x.size()[0] # number of images in the batch
#                 k = x.size()[1] # number of feature maps
#                 x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

#                 for n in range(causality_maps.size()[0]): #batch size
#                     causality_map = causality_maps[n]
#                     triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
#                     tril = torch.tril(causality_map, -1).T.contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
#                     bool_ij = (tril>triu).T
#                     bool_ji = (triu>tril)
#                     bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
#                     by_col = torch.sum(bool_matrix, 1)
#                     by_row = torch.sum(bool_matrix, 0)

#                     if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
#                         # if self.causality_setting == "mulcat":
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
#                     elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_row - by_col # 
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_row - by_col)>0) # 
#                     else:
#                         raise ValueError
                
#                     x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
#                     x_c[n] = self.relu(x_causes) #rectify every negative value to zero

#                 # x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
#                 #TODO 18 settembre 2023: commented the above and run the below 
#                 if self.causality_setting == "mul" or self.causality_setting == "mulbool":
#                     x = torch.flatten(x_c, 1) #substitute the actual features with the filtered version of them.
#                 else: #mulcat or mulcatbool need to concatenate the X_c to the actual original features x
#                     x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version

            
#         else: #traditional, non causal:
#             x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
#         x = self.classifier(x)
#         x = self.softmax(x) #TODO 12 luglio 2023
#         # print(torch.round(x,decimals=2))

#         # print(f"FORWARD - x (return): {x.requires_grad}")
#         # print(f"FORWARD - causality_maps (return): {causality_maps.requires_grad}")
#         return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
    


## Convnext models not completely curated yet, utilize the resnet18 version instead. TODO.
# from torchvision.models import convnext_tiny, convnext_base
# from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
# class ResNextCA(nn.Module):
#     def __init__(self, which_resnext, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            
#             super(ResNextCA, self).__init__()            
                
#             self.which_resnext = which_resnext
#             self.img_size = dim
#             self.channels = channels
#             self.num_classes = num_classes
#             self.is_pretrained = is_pretrained
#             self.is_feature_extractor = is_feature_extractor
            
#             self.causality_aware = causality_aware
#             self.causality_method = causality_method
#             self.LEHMER_PARAM = LEHMER_PARAM
#             self.causality_setting = causality_setting #

#             self.visual_attention = visual_attention #boolean
#             self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

#             if self.which_resnext=="tiny":
#                 self.first_output_ftrsmap_number = 96
#                 if self.is_pretrained:
#                     model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
#                 else:
#                     model = convnext_tiny()
#             elif self.which_resnext=="base":
#                 self.first_output_ftrsmap_number = 128
#                 if self.is_pretrained:
#                     model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
#                 else:
#                     model = convnext_base()

#             if self.channels == 1:
#                 model.features[0][0] = nn.Conv2d(1, self.first_output_ftrsmap_number, kernel_size=4, stride=4)
#             elif self.channels==3:
#                 if self.is_feature_extractor:
#                     for param in model.parameters(): #freeze the extraction layers
#                         param.requires_grad = False
            
#             ## creating the structure of our custom model starting from the original building blocks of Resnext:
#             # layer0, layer1, layer2, layer3, layer4, layer5, layer6,layer7, and classifier
#             self.layer0 = model.features[0] #
#             self.layer1 = model.features[1] #
#             self.layer2 = model.features[2] #
#             self.layer3 = model.features[3] #
#             self.layer4 = model.features[4] #
#             self.layer5 = model.features[5] #
#             self.layer6 = model.features[6] #
#             self.layer7 = model.features[7] #

#             model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
#             self.avgpool = model.avgpool
            
#             ##self.features = model

#             # if self.visual_attention: #TODO commentato per ora, capire quali layer attenzionare a differenza di resnet18
#             #     ## define the attention blocks
#             #     self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
#             #     self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

#             self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.
#             if self.which_resnext=="tiny":
#                 self.last_ftrmap_number = 768
#             elif self.which_resnext=="base":
#                 self.last_ftrmap_number = 1024

#             if self.causality_aware:
#                 if self.causality_setting == "cat": #[1, n*n*k + k*k]
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes)
#                 elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * self.last_ftrmap_number, self.num_classes)
#                 elif (self.causality_setting == "mul") or (self.causality_setting == "mulbool"): #TODO 
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)
#             else: 
#                 if self.visual_attention: ##TODO False per ora, capire quali dimensioni al posto di 12 e 6 in base ai layer che si scelgono; TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
#                     self.classifier = nn.Linear(128*12*12 + 256*6*6 + self.last_ftrmap_number*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
#                 else:
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

#             self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added

#     def forward(self, x):
        
#         if torch.isnan(x).any():
#             l_ingresso_era_gia_corrotto_con_nan

#         # print(f"resnext feature size:\t {x.size()}") #torch.Size([N, 1, 96, 96]) #esempio
#         x_layer0 = self.layer0(x)
#         # print(f"resnext x_layer0 size:\t {x_layer0.size()}") torch.Size([N, 128, 24, 24]) ##esempio, valori con ResNext BASE
#         x_layer1 = self.layer1(x_layer0)
#         # print(f"resnext x_layer1 size:\t {x_layer1.size()}") torch.Size([N, 128, 24, 24])
#         x_layer2 = self.layer2(x_layer1)
#         # print(f"resnext x_layer2 size:\t {x_layer2.size()}")  torch.Size([N, 256, 12, 12])
#         x_layer3 = self.layer3(x_layer2)
#         # print(f"resnext x_layer3 size:\t {x_layer3.size()}")  torch.Size([N, 256, 12, 12])
#         x_layer4 = self.layer4(x_layer3)
#         # print(f"resnext x_layer4 size:\t {x_layer4.size()}") torch.Size([N, 512, 6, 6])
#         x_layer5 = self.layer5(x_layer4)
#         # print(f"resnext x_layer5 size:\t {x_layer5.size()}") torch.Size([N, 512, 6, 6])
#         x_layer6 = self.layer6(x_layer5)
#         # print(f"resnext x_layer6 size:\t {x_layer6.size()}") torch.Size([N, 1024, 3, 3])
#         x_layer7 = self.layer7(x_layer6)
#         # print(f"resnext x_layer7 size:\t {x_layer7.size()}") torch.Size([N, 1024, 3, 3])
#         x = self.avgpool(x_layer7)
#         # print(f"resnext avgpool size:\t {x.size()}") #torch.Size([N, 1024, 3, 3]) <-- siccome avevo messo Identity
        

#         # compute the attention probabilities:
#         # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
#         # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

#         #check nan
#         if torch.isnan(x).any():
#             corrotto_con_nan
#         if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]:
#             print("Correcting shape mismatch...(in forward, after avgpool, before causality module)")
#             x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size))

#         causality_maps = None #initialized to none

#         if self.causality_aware:
#             # x, causality_maps = self.get_causality_maps(x) # code for computing causality maps given a batch of featuremaps x
#             x, causality_maps = GetCausalityMaps(x, self.causality_method, self.LEHMER_PARAM)

#             if self.causality_setting == "cat":
#                 x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
#             elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool") or (self.causality_setting == "mul") or (self.causality_setting == "mulbool"):
#                 b = x.size()[0] # number of images in the batch
#                 k = x.size()[1] # number of feature maps
#                 x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

#                 for n in range(causality_maps.size()[0]): #batch size
#                     causality_map = causality_maps[n]
#                     triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
#                     tril = torch.tril(causality_map, -1).T.contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
#                     bool_ij = (tril>triu).T.contiguous()
#                     bool_ji = (triu>tril)
#                     bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
#                     by_col = torch.sum(bool_matrix, 1)
#                     by_row = torch.sum(bool_matrix, 0)

#                     if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
#                     elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_row - by_col # 
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_row - by_col)>0) # 
#                     else:
#                         raise ValueError
                
#                     x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
#                     x_c[n] = self.relu(x_causes) #rectify every negative value to zero
                    
#                 # x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
#                 #TODO commented the above and run the below 
#                 if self.causality_setting == "mul" or self.causality_setting == "mulbool":
#                     x = torch.flatten(x_c, 1) #substitute the actual features with the filtered version of them.
#                 else: #mulcat or mulcatbool need to concatenate the X_c to the actual original features x
#                     x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version

#         else: #traditional, non causal:
#             x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
#         x = self.classifier(x)
#         x = self.softmax(x) #TODO 12 luglio 2023
#         # print(x)
#         return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
   