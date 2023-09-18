import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet
import functools
import time

class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        '''
        The intermediate feature vector(F) is the output of pool-3 or pool-4 (the "local" features information)
        and the global feature vector (output of pool-5) is fed as input to the attention layer.

        Both the feature vectors pass through a convolution layer.
        When the spatial size of global and intermediate features are different, feature upsampling
        is done via bilinear interpolation.
        The up_factor determines by what factor the convoluted global feature vector has to be upscaled.

        After that an element wise sum is done followed by a convolution operation that just
        reduces the many channels to 1.

        This is then fed into a Softmax layer, which gives us a normalized Attention map (A).
        Each scalar element in A represents the degree of attention to the corresponding spatial feature vector in F.

        The new feature vector 𝐹̂ is then computed by pixel-wise multiplication.
        That is, each feature vector f is multiplied by the attention element a.

        So, the attention map A and the new feature vector 𝐹̂ are the outputs of the Attention Layer.
        '''
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    
    def forward(self, l, g):
        N, C, W, H = l.size() #batch, channels (number of feature maps), width and height
        l_ = self.W_l(l) #local
        g_ = self.W_g(g) #global
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C) # global average pooling
        return a, output


## 12 luglio, funzione al posto della classe            
# def GetCausalityMaps(x, causality_method, LEHMER_PARAM):

#     feature_maps = x.detach().clone()
#     if feature_maps.get_device() != x.get_device():
#         feature_maps.to(x.get_device())
#     b = feature_maps.size()[0] # number of images in the batch
#     k = feature_maps.size()[1] # number of feature maps
    
#     causality_maps = torch.zeros((b, k, k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
    
#     for b_i in range(b):

#         current_feature_maps = feature_maps[b_i,:,:,:]

#         if torch.isnan(current_feature_maps).any():
#             print(f"In b_i {b_i}, the current feature maps object contains NaN")
#             raise ValueError
        
#         maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
#         MAX_F = torch.max(maximum_values)
#         current_feature_maps = torch.nan_to_num(current_feature_maps/MAX_F, nan = 0.0)

#         ## After having normalized the feature maps, comes the distinction between the method by which computing causality
#         if causality_method == "max": #Option 1 : max values
#             sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
#             if torch.sum(torch.isnan(sum_values))>0:
#                 sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
#             maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
            
#             mtrx = torch.outer(maximum_values, maximum_values) #the max value of mtrx object is 1.0
#             tmp = torch.div(mtrx, sum_values)
#             causality_map = torch.nan_to_num(tmp, nan = 0.0)
#             causality_maps[b_i] = causality_map
        
#         elif causality_method == "lehmer": #Option 2 : Lehmer mean
            
#             current_feature_maps = torch.flatten(current_feature_maps,1) # [k,n*n], eg [512, 8*8]    

#             #compute the outer product of all the pairs of flattened featuremaps.
#             # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
#             cross_matrix = torch.einsum('ai,bj->abij', current_feature_maps, current_feature_maps) #eg, [512,512,64,64]  symmetric values                  
#             cross_matrix = cross_matrix.flatten(2) #eg, [512,512,4096]
            
#             # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
#             # first, compute the two powers of the cross matrix
#             p_plus_1_powers = torch.pow(cross_matrix, LEHMER_PARAM + 1)
#             p_powers = torch.pow(cross_matrix, LEHMER_PARAM)
#             numerators = torch.sum(p_plus_1_powers, dim=2)
#             denominators = torch.sum(p_powers, dim=2)
#             lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators), nan=0)
            
#             # then the lehmer denominator of the causality map:
#             # it is the lehemr mean of the single feature map, for all the feature maps by column
#             p_plus_1_powers_den = torch.pow(current_feature_maps, LEHMER_PARAM + 1)
#             p_powers_den = torch.pow(current_feature_maps, LEHMER_PARAM)
#             numerators_den = torch.sum(p_plus_1_powers_den, dim=1)
#             denominators_den = torch.sum(p_powers_den, dim=1)
#             lehmer_denominator = torch.nan_to_num(torch.div(numerators_den,denominators_den), nan=0)
            
#             #and finally obtain the causality map values by computing the division
#             causality_map = torch.nan_to_num(torch.div(lehmer_numerators, lehmer_denominator), nan=0)
#             causality_maps[b_i] = causality_map
#         else:
#             print(causality_method) # we implemented only MAX and LEHMER options, so every other case is a typo/error
#             raise NotImplementedError
        
        
#     return x, causality_maps    
                     




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Resnet18CA(nn.Module):
    def __init__(self, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            super(Resnet18CA, self).__init__()
            self.img_size = dim
            self.channels = channels
            self.num_classes = num_classes
            self.is_pretrained = is_pretrained
            self.is_feature_extractor = is_feature_extractor
            
            self.causality_aware = causality_aware
            self.causality_method = causality_method
            self.LEHMER_PARAM = LEHMER_PARAM
            self.causality_setting = causality_setting # #"cat" (paper originale), "mulcat"; ("mulcat1x1") , 15 giugno 2023 aggiunto

            self.visual_attention = visual_attention #boolean
            self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

            if self.is_pretrained:
                model = resnet18(pretrained=True)
            else:
                model = resnet18()

            if self.channels == 1:
                model.conv1 = nn.Conv2d(1, 64,kernel_size=7, stride=2, padding=3, bias=False)
            elif self.channels==3:
                if self.is_feature_extractor:
                    for param in model.parameters(): #freeze the extraction layers
                        param.requires_grad = False
            
            ## creating the structure of our custom model starting from the original building blocks of resnet:
            # starting block, layer1, layer2, layer3, layer4, ending block, and classifier

            self.starting_block = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool) #output size is halved
            self.layer1 = model.layer1 #output size is halved
            self.layer2 = model.layer2 #output size is halved
            self.layer3 = model.layer3 #output size is halved
            self.layer4 = model.layer4 #output size is halved
            #here, outputsize is the original one divided by 32.

            model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
            model.fc = Identity() # Cancel classification layer to get only feature extractor
            self.ending_block = nn.Sequential(model.avgpool, model.fc)
            
            ##self.features = model
            if self.visual_attention:
                ## define the attention blocks
                self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
                self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

            self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.

            if self.causality_aware:

                # self.get_causality_maps = GetCausalityMaps(self.causality_method, self.LEHMER_PARAM)

                if self.causality_setting == "cat": #[1, n*n*k + k*k]
                    self.classifier = nn.Linear(512 * self.last_ftrmap_size * self.last_ftrmap_size + 512*512, self.num_classes)
                elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
                    self.relu = nn.ReLU()
                    self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * 512, self.num_classes)
                
            else:
                if self.visual_attention: #TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
                    self.classifier = nn.Linear(128*12*12 + 256*6*6 + 512*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
                else:
                    self.classifier = nn.Linear(512 * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

            self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added
            print("Created instance of ABLATION resnet18ca###############")

    def forward(self, x):
        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        x = self.starting_block(x)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x = self.ending_block(x_layer4)

        # compute the attention probabilities:
        # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
        # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

        #check nan
        if torch.isnan(x).any():
            corrotto_con_nan
        if list(x.size()) != [int(x.size(0)), 512, self.last_ftrmap_size, self.last_ftrmap_size]:
            x = torch.reshape(x, (int(x.size(0)), 512, self.last_ftrmap_size, self.last_ftrmap_size))

        causality_maps = None #initialized to none

        if self.causality_aware:
            b = x.size()[0] # number of images in the batch
            k = x.size()[1] # number of feature maps

            # x, causality_maps = self.get_causality_maps(x) # code for computing causality maps given a batch of featuremaps x
            # x, causality_maps = GetCausalityMaps(x, self.causality_method, self.LEHMER_PARAM)
            ##TODO ablation: al posto di computare le vere causality map, faccio finta e la genero random, vedi sotto:
            
            if self.causality_setting == "cat":
                # x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
                ##TODO ablation:
                causality_maps = torch.rand(size=(b,k*k), device=x.get_device()) * (1 + 1e-6)  # la versione già flattenata
                causality_maps.clamp_(0, 1)   
                # print(f"causality_maps MIN e MAX: {causality_maps.min()}, {causality_maps.max()}")           
               
                x = torch.cat((torch.flatten(x, 1), causality_maps), dim=1)

            elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"):
                
                x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

                for n in range(b): #batch size
                    # causality_map = causality_maps[n]
                    # triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
                    # tril = torch.tril(causality_map, -1).T #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
                    # bool_ij = (tril>triu).T
                    # bool_ji = (triu>tril)
                    # bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
                    # by_col = torch.sum(bool_matrix, 1)
                    # by_row = torch.sum(bool_matrix, 0)


                    #TODO
                    # if self.causality_setting == "mulcat":
                        # causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
                    
                    causes_mul_factors = torch.randint(low=-(k-1), high=k, size=(k,),device=x.get_device()) #high exclusive therefore is k-1, that is right, the maximum number of times a feature can cause every OTHER feature.

                    if self.causality_setting == "mulcatbool":
                        # causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                        causes_mul_factors = 1.0*((causes_mul_factors)>0)
                    

                    #TODO
                    # print(f"self.causality_setting {self.causality_setting}, causes_mul_factors {causes_mul_factors[:11]}")
                    #

                    x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
                    x_c[n] = self.relu(x_causes) #rectify every negative value to zero
                    
                x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
            
        else: #traditional, non causal:
            raise ValueError #in ABLATION qui non ci deve entrare, solo per i causali lo faccio
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
        x = self.classifier(x)
        x = self.softmax(x) #TODO 12 luglio 2023
        # print(x)
        return x, causality_maps #return the logit (or probabilities if using softmax) of the classification, and the causality maps for optional visualization or some metric manipulation during training
    