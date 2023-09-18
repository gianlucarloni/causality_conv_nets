import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet
import functools
import time
from torch import sparse


##
def sparse_reshape(sparse_tensor, size):
    '''
    https://discuss.pytorch.org/t/runtimeerror-reshape-is-not-implemented-for-sparse-tensors/105313
    '''
    mask = torch.sparse_coo_tensor(indices=sparse_tensor.indices(),
                                   values=torch.ones(sparse_tensor._nnz(),
                                                     device=sparse_tensor.device,
                                                     dtype=torch.bool),
                                   size=sparse_tensor.size())
    mask = mask.to_dense().reshape(size)

    sparse_tensor = torch.sparse_coo_tensor(indices=mask.to_sparse().indices(),
                                            values=sparse_tensor.values(),
                                            size=size)
    return sparse_tensor.coalesce()
##

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


#### 10 luglio 2023
# class GetCausalityMaps(nn.Module):
#     def __init__(self, causality_method, LEHMER_PARAM):
#         super(GetCausalityMaps, self).__init__()
#         self.causality_method = causality_method # max or lehmer
#         self.LEHMER_PARAM = LEHMER_PARAM

#     def forward(self, x): 
#         feature_maps = x.detach().clone()
#         if feature_maps.get_device() != x.get_device():
#             feature_maps.to(x.get_device())
#         b = feature_maps.size()[0] # number of images in the batch
#         k = feature_maps.size()[1] # number of feature maps
        
#         causality_maps = torch.zeros((b, k, k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
        
#         for b_i in range(b):

#             current_feature_maps = feature_maps[b_i,:,:,:]

#             if torch.isnan(current_feature_maps).any():
#                 print(f"In b_i {b_i}, the current feature maps object contains NaN")
#                 raise ValueError
            
#             maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
#             MAX_F = torch.max(maximum_values)
#             current_feature_maps = torch.nan_to_num(current_feature_maps/MAX_F, nan = 0.0)

#             ## After having normalized the feature maps, comes the distinction between the method by which computing causality
#             if self.causality_method == "max": #Option 1 : max values
#                 sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
#                 if torch.sum(torch.isnan(sum_values))>0:
#                     sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
#                 maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                
#                 mtrx = torch.outer(maximum_values, maximum_values) #the max value of mtrx object is 1.0
#                 tmp = torch.div(mtrx, sum_values)
#                 causality_map = torch.nan_to_num(tmp, nan = 0.0)
#                 causality_maps[b_i] = causality_map
            
#             elif self.causality_method == "lehmer": #Option 2 : Lehmer mean
                
#                 current_feature_maps = torch.flatten(current_feature_maps,1) # [k,n*n], eg [512, 8*8]    

#                 #compute the outer product of all the pairs of flattened featuremaps.
#                 # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
#                 cross_matrix = torch.einsum('ai,bj->abij', current_feature_maps, current_feature_maps) #eg, [512,512,64,64]  symmetric values                  
#                 cross_matrix = cross_matrix.flatten(2) #eg, [512,512,4096]
                
#                 # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
#                 # first, compute the two powers of the cross matrix
#                 p_plus_1_powers = torch.pow(cross_matrix, self.LEHMER_PARAM + 1)
#                 p_powers = torch.pow(cross_matrix, self.LEHMER_PARAM)
#                 numerators = torch.sum(p_plus_1_powers, dim=2)
#                 denominators = torch.sum(p_powers, dim=2)
#                 lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators), nan=0)
                
#                 # then the lehmer denominator of the causality map:
#                 # it is the lehemr mean of the single feature map, for all the feature maps by column
#                 p_plus_1_powers_den = torch.pow(current_feature_maps, self.LEHMER_PARAM + 1)
#                 p_powers_den = torch.pow(current_feature_maps, self.LEHMER_PARAM)
#                 numerators_den = torch.sum(p_plus_1_powers_den, dim=1)
#                 denominators_den = torch.sum(p_powers_den, dim=1)
#                 lehmer_denominator = torch.nan_to_num(torch.div(numerators_den,denominators_den), nan=0)
                
#                 #and finally obtain the causality map values by computing the division
#                 causality_map = torch.nan_to_num(torch.div(lehmer_numerators, lehmer_denominator), nan=0)
#                 causality_maps[b_i] = causality_map
#             else:
#                 print(self.causality_method) # we implemented only MAX and LEHMER options, so every other case is a typo/error
#                 raise NotImplementedError
            
            
#         return x, causality_maps    

## 12 luglio, funzione al posto della classe            
def GetCausalityMaps(x, causality_method, LEHMER_PARAM):

    feature_maps = x.detach().clone()
    if feature_maps.get_device() != x.get_device():
        feature_maps.to(x.get_device())
    b = feature_maps.size()[0] # number of images in the batch
    k = feature_maps.size()[1] # number of feature maps
    
    causality_maps = torch.zeros((b, k, k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
    
    #TODO 15 settembre 2023: provo ad usare SPARSE matrix computation (to_sparse())
    causality_maps=causality_maps.to_sparse() #COO format

    for b_i in range(b):

        current_feature_maps = feature_maps[b_i,:,:,:]
        print(f"current_feature_maps: {current_feature_maps.size()}")

        if torch.isnan(current_feature_maps).any():
            print(f"In b_i {b_i}, the current feature maps object contains NaN")
            raise ValueError
        
        maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
        MAX_F = torch.max(maximum_values)
        current_feature_maps = torch.nan_to_num(current_feature_maps/MAX_F, nan = 0.0)

        ## After having normalized the feature maps, comes the distinction between the method by which computing causality
        if causality_method == "max": #Option 1 : max values
            sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
            if torch.sum(torch.isnan(sum_values))>0:
                sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
            maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
            
            mtrx = torch.outer(maximum_values, maximum_values) #the max value of mtrx object is 1.0
            tmp = torch.div(mtrx, sum_values)
            causality_map = torch.nan_to_num(tmp, nan = 0.0)
            causality_maps[b_i] = causality_map
        
        elif causality_method == "lehmer": #Option 2 : Lehmer mean
            
            # #TODO ########### 15 settembre 2023: provo ad usare SPARSE matrix computation #################################

            # current_feature_maps = torch.flatten(current_feature_maps,1) # [k,n*n], eg [512, 8*8]    

            # #compute the outer product of all the pairs of flattened featuremaps.
            # # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
            # cross_matrix = torch.einsum('ai,bj->abij', current_feature_maps, current_feature_maps) #eg, [512,512,64,64]  symmetric values                  

            # cross_matrix=cross_matrix.to_sparse()
            # # print(f"cross_matrix sparse: {cross_matrix.size()}")

            # # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
            # # first, compute the two powers of the cross matrix
            # p_plus_1_powers = torch.pow(cross_matrix, LEHMER_PARAM + 1)
            # p_powers = torch.pow(cross_matrix, LEHMER_PARAM)

            # #TODO 15 settembre 2023: provo ad usare SPARSE matrix computation 
            # numerators = sparse.sum(p_plus_1_powers, dim=[2,3])
            # # print(f"numerators sparse: {numerators.size()}") #eg, [768,768]
            # numerators = sparse_reshape(numerators, size=(768*768,))
            # # print(f"numerators sparse reshaped (flattened): {numerators.size()}") 
            # numerators=numerators.to_dense()

            # #TODO 15 settembre 2023: provo ad usare SPARSE matrix computation 
            # denominators = sparse.sum(p_powers, dim=[2,3])
            # # print(f"denominators sparse: {denominators.size()}")
            # denominators = sparse_reshape(denominators, size=(768*768,))
            # # print(f"denominators sparse reshaped (flattened): {denominators.size()}")
            # denominators=denominators.to_dense()

            # lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators), nan=0)
                
            
            #######versione vecchia funzionante############
            current_feature_maps = torch.flatten(current_feature_maps,1) # [k,n*n], eg [512, 8*8]    

            #compute the outer product of all the pairs of flattened featuremaps.
            # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
            cross_matrix = torch.einsum('ai,bj->abij', current_feature_maps, current_feature_maps) #eg, [512,512,64,64]  symmetric values                  
            cross_matrix = cross_matrix.flatten(2) #eg, [512,512,4096]

            # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
            # first, compute the two powers of the cross matrix
            p_plus_1_powers = torch.pow(cross_matrix, LEHMER_PARAM + 1)
            p_powers = torch.pow(cross_matrix, LEHMER_PARAM)

            numerators = torch.sum(p_plus_1_powers, dim=2)
            denominators = torch.sum(p_powers, dim=2)
            lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators), nan=0)
            #############           
            
            esci: la versione sopra, con imgssize 128, regge anche batchsize di >50, quindi sembra non ci sia bisogno di sparse.
            Trovare, se c'è, un limite oltre il quale regge solamente con sparse, per giustificarne l'utilizzo, altrimenti nulla.
            
            # then the lehmer denominator of the causality map:
            # it is the lehemr mean of the single feature map, for all the feature maps by column
            p_plus_1_powers_den = torch.pow(current_feature_maps, LEHMER_PARAM + 1)
            p_powers_den = torch.pow(current_feature_maps, LEHMER_PARAM)
            numerators_den = torch.sum(p_plus_1_powers_den, dim=1)
            denominators_den = torch.sum(p_powers_den, dim=1)
            lehmer_denominator = torch.nan_to_num(torch.div(numerators_den,denominators_den), nan=0)
            
            #and finally obtain the causality map values by computing the division
            causality_map = torch.nan_to_num(torch.div(lehmer_numerators, lehmer_denominator), nan=0)
            causality_maps[b_i] = causality_map
        else:
            print(causality_method) # we implemented only MAX and LEHMER options, so every other case is a typo/error
            raise NotImplementedError
        
        
    return x, causality_maps    
                     

# ###
# def lehmer_mean(x, p):
#     # p is a positive number
#     # returns the Lehmer mean L_p(x)
#     numerator = 0 # initialize the numerator
#     denominator = 0 # initialize the denominator
#     #print(f"LEHMER mean computation: x {x.size()}")
#     for xi in x: # loop over the elements of x
#         #print(f"---LEHMER mean computation:    xi value {xi}")
#         numerator += xi ** (p+1) # update the numerator
#         denominator += xi ** (p) # update the denominator
    
#     div = torch.nan_to_num(numerator/denominator, nan=0)
    
#     #print(f"LEHMER mean computation: div size {div.size()} and value: {div}")

#     ###
    
    
#     return div # return the quotient

# def weighted_lehmer_mean(x, w, p):
#     # x and w are tuples of positive numbers of the same length
#     # p is a positive number
#     # returns the weighted Lehmer mean L_p,w(x)
#     numerator = 0 # initialize the numerator
#     denominator = 0 # initialize the denominator
#     for i in range(len(x)): # loop over the elements of x and w
#         numerator += w[i] * x[i] ** (p+1) # update the numerator
#         denominator += w[i] * x[i] ** p # update the denominator
#     return numerator / denominator # return the quotient
# ###


class LightCNN(nn.Module):
    def __init__(self,dim = 128, channels = 1, num_classes = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(8, 32, kernel_size=3, padding = 1) 
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.25613)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

###############################################################################

class SimpleCNN(nn.Module):
    def __init__(self,dim = 128, channels = 1, num_classes = 2, causality_aware = False, inference_mode=False):
        super().__init__()
        self.image_dim = dim
        self.causality_aware = causality_aware
        self.inference_mode = inference_mode
        
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding = 1) 
        """ self.fc1 = nn.Linear(16 * 32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) """

        last_ftrmap_size = int(self.image_dim / 4) #since it has two conv layers, and (in this case) each halves the dimension
        
        if self.causality_aware:
            self.fc1 = nn.Linear(16 * last_ftrmap_size * last_ftrmap_size + 16*16, 256)
        else:
            self.fc1 = nn.Linear(16 * last_ftrmap_size * last_ftrmap_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        if not self.causality_aware:
            if not self.inference_mode:
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
            else:
                ftrs_mps = {}
                x = self.pool(F.relu(self.conv1(x)))
                ftrs_mps["pool1"] = x
                x = self.pool(F.relu(self.conv2(x)))
                ftrs_mps["pool2"] = x
                #print(f"JUST BEFORE FLATTENING: {x.size()}") ## with input image 32x32, viene [1000, 16, 8, 8]
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                #print(f"JUST after FLATTENING: {x.size()}") ##[1000, 1024]
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x, ftrs_mps
        else:
            ## In this case, implement the computation of causality map P(F|F) on the feature maps of the last pooling layer
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            feature_maps = x.clone()
            if feature_maps.get_device() != x.get_device():
                print("Entro a modificare il device, perdita di tempo")
                feature_maps.to(x.get_device())
            b = feature_maps.size()[0] # number of images in the batch
            k = feature_maps.size()[1] # number of feature maps
            #print(f"b: {b}, k: {k}")
            #print(f"feature maps [0] [0] before normalization: {feature_maps[0,0,:,:]}, with maximum: {feature_maps[0,0,:,:].max()}")
            #print(f"feature maps [0] [1] before normalization: {feature_maps[0,1,:,:]}, with maximum: {feature_maps[0,1,:,:].max()}")
            #print(f"max value across the different feature maps of the first image in the batch: {feature_maps[0,:,:,:].max()}")
            
            """ start_time1 = time.perf_counter()
            ### Version 1; vanilla for-loop based version, works but it is really slow
            batch_causality_maps = torch.empty((b,k,k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
            for b_i in range(b): #TODO
                max_value = 0
                for k_i in range(k):
                    current_max = feature_maps[b_i,k_i,:,:].max()
                    if current_max > max_value:
                        max_value = current_max               
                feature_maps[b_i,:,:,:] /= max_value
                #print()
                #print(f"feature maps [0] [0] AFTER normalization: {feature_maps[0,0,:,:]}, with maximum: {feature_maps[0,0,:,:].max()}")
                #print(f"feature maps [0] [1] AFTER normalization: {feature_maps[0,1,:,:]}, with maximum: {feature_maps[0,1,:,:].max()}")
                
                causality_map = torch.empty((k,k),device=x.get_device())
                for i in range(k):
                    max_F_i = feature_maps[b_i, i,:,:].max()
                    for j in range(k):
                        sum_F_j = feature_maps[b_i, j,:,:].sum()
                        if sum_F_j==0:
                            causality_map[i,j] = 0
                            #print(f"WARNING: with image {b_i} in the batch, when i={i}, the sum of F_j (j={j}) was 0. Instead of obtaining NaN, I set the causality value to 0.")
                        else:
                            max_F_j = feature_maps[b_i, j,:,:].max()                            
                            causality_map[i,j] = (max_F_i*max_F_j)/sum_F_j
                    
                batch_causality_maps[b_i] = causality_map
            end_time1 = time.perf_counter()
            print(f"FOR LOOPS, elapsed time: {end_time1-start_time1}")
            with open(os.path.join(os.getcwd(),"version_P_FF_andElapsedTime.txt"),"w") as fout:
                fout.write(f"FOR LOOPS, elapsed time: {end_time1-start_time1} and P(F|F): {batch_causality_maps[0]}\n") """


            #start_time2 = time.perf_counter()
            ### Version 2; leveraging numpy.outer() multiplication between the vector of maximum featuremaps values with itself, and then dividing each row for the sum values 
            batch_causality_maps = torch.empty((b,k,k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
            for b_i in range(b):
                current_feature_maps = feature_maps[b_i,:,:,:]
                assert not torch.isnan(current_feature_maps).any()
                
                #compute the MAX of each feature map, return a 1D tensor of size 16. 
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                #print(f"maximum_values: {maximum_values}")
                # and then compute the maximum of all maximum values, ie MAX(F)
                MAX_F = torch.max(maximum_values)

                #print(f"MAX_F: {MAX_F}")

                #normalize each feature map with that MAX_F value, return a probability map
                current_feature_maps = torch.nan_to_num(current_feature_maps/MAX_F, nan = 0.0)

                #print(f"current_feature_maps: {current_feature_maps}")
                #if torch.sum(torch.isnan(current_feature_maps))>0:
                #    #print(f"current_feature_maps has NaN, at b_i {b_i}")
                #   current_feature_maps = torch.nan_to_num(current_feature_maps,nan=0.0)#sostituire gli eventuali nan con degli zeri


                #compute the SUM of each feature map, return a 1D tensor of size 16. 
                sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
                if torch.sum(torch.isnan(sum_values))>0:
                    #print(f"sum_values has NaN, at b_i {b_i}")
                    sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
                #print(f"sum: {sum_values}")

                #compute the MAX of each new feature map, return a 1D tensor of size 16. 
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                #print(f"maximum_values new: {maximum_values}")

                #first compute the outer product between that tensor and itself, return a matrix (2D tensor) with all possible elementwise multiplications
                mtrx = torch.outer(maximum_values, maximum_values)
                #print(f"Matrix {mtrx} of size: {mtrx.size()}")
                # and then divide the rows 1:k for the vector containing SUM1:SUMk, and add the result to the outer collecting tensor  
                batch_causality_maps[b_i] = torch.nan_to_num(torch.div(mtrx, sum_values), nan = 0.0)
                #print(batch_causality_maps[b_i])
            
            #end_time2 = time.perf_counter()
            #print(f"VERSION 2, elapsed time={end_time2-start_time2}")
            #with open(os.path.join(os.getcwd(),"version_P_FF_andElapsedTime.txt"),"a") as fout:
            #    fout.write(f"VERSION 2, elapsed time: {end_time2-start_time2} and P(F|F): {batch_causality_maps[0]}")

            ## [bsize, 16, 16]
            #print(f"causality maps prior to flattening: {batch_causality_maps.size()}")
            batch_causality_maps = torch.flatten(batch_causality_maps, 1) # flatten all dimensions except batch, results in [bsize,16x16]

            ## with input image 32x32, viene [bsize, 16, 8, 8]
            #print(f"x prior to flattening: {x.size()}")
            x = torch.flatten(x, 1) # flatten all dimensions except batch

            ## concatenate the flattened featuremaps and the flattened causality maps
            #print(f"causalities maps after flattening: {batch_causality_maps.size()}\nx after flattening: {x.size()}")
            x = torch.cat((x,batch_causality_maps), dim=1)

            #[bsize, 1024] (or [bsize, 1280] with Causality awareness)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            return x



#####################################################################



class EqualCNN(nn.Module):
    def __init__(self, NUMB_FTRS_MPS = 32, dim = 64, channels = 1, num_classes = 2, causality_aware=False, causality_method="max", LEHMER_PARAM=None):
        super(EqualCNN, self).__init__()
        self.NUMB_FTRS_MPS = NUMB_FTRS_MPS
        self.causality_aware = causality_aware
        self.causality_method = causality_method
        self.LEHMER_PARAM = LEHMER_PARAM
       
        #3x3 + 3x3 convolution (5x5 convolution)--> 1 block
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=16, 
                                kernel_size=3, padding = 1)
        self.conv1_2 =  nn.Conv2d(in_channels=16, out_channels=self.NUMB_FTRS_MPS, 
                                kernel_size=3, padding = 1)
        self.ReLU = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(self.NUMB_FTRS_MPS)
        
        # 3x3 convolution --> 2 block
        self.conv1x1_2 = nn.Conv2d(in_channels = self.NUMB_FTRS_MPS, out_channels = self.NUMB_FTRS_MPS, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels= self.NUMB_FTRS_MPS, out_channels= self.NUMB_FTRS_MPS, 
                                kernel_size=3, padding = 1)
        self.batch2 = nn.BatchNorm2d(self.NUMB_FTRS_MPS)
        
        # max pooling
        self.maxpool2D_1 = nn.MaxPool2d((2,2))
        
        # 3x3 convolution --> 3 block
        self.conv1x1_3 = nn.Conv2d(in_channels = self.NUMB_FTRS_MPS, out_channels = self.NUMB_FTRS_MPS, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels= self.NUMB_FTRS_MPS, out_channels= self.NUMB_FTRS_MPS, 
                                kernel_size=3, padding = 1)
        self.batch3 = nn.BatchNorm2d(self.NUMB_FTRS_MPS)
        
        # 4 block
        self.conv1x1_4 = nn.Conv2d(in_channels = self.NUMB_FTRS_MPS, out_channels = self.NUMB_FTRS_MPS, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels= self.NUMB_FTRS_MPS, out_channels= self.NUMB_FTRS_MPS, 
                                kernel_size=3, padding = 1)
        self.batch4 = nn.BatchNorm2d(self.NUMB_FTRS_MPS)
        
        # max pooling
        self.maxpool2D_2 = nn.MaxPool2d((2,2))



        ## TODO; 29 maggio, provato ad aggiungere altri due strati conv con relativo pooling
        # 5 block
        self.conv1x1_5 = nn.Conv2d(in_channels = self.NUMB_FTRS_MPS, out_channels = self.NUMB_FTRS_MPS, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=self.NUMB_FTRS_MPS, out_channels=self.NUMB_FTRS_MPS, 
                                kernel_size=3, padding = 1)
        self.batch5 = nn.BatchNorm2d(self.NUMB_FTRS_MPS)
        # 6 block
        self.conv1x1_6 = nn.Conv2d(in_channels = self.NUMB_FTRS_MPS, out_channels = self.NUMB_FTRS_MPS, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=self.NUMB_FTRS_MPS, out_channels=self.NUMB_FTRS_MPS, 
                                kernel_size=3, padding = 1)
        ## 1 giugno debug TODO  self.conv6 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding = 1)
        self.batch6 = nn.BatchNorm2d(self.NUMB_FTRS_MPS)
        # max pooling
        self.maxpool2D_3 = nn.MaxPool2d((2,2))
        ## TODO; 29 maggio.


        
        # Fully connected layers + dropout
        self.drop = nn.Dropout(0.4)

        #number_of_pooling = 2 #TODO 29 maggio, vedi sopra
        number_of_pooling = 3

        final_feature_map_dim = round(dim/(2**number_of_pooling)) #venendo dimezzata ad ogni pooling
        
        if self.causality_aware:  
            self.fc1 = nn.Linear(self.NUMB_FTRS_MPS*final_feature_map_dim*final_feature_map_dim + self.NUMB_FTRS_MPS*self.NUMB_FTRS_MPS, 256)  #(in_features=32*32*80)

            #TODO 1 giugno debug: self.fc1 = nn.Linear(3*final_feature_map_dim*final_feature_map_dim + 3*3, 256)  #(in_features=32*32*80)
        else:
            self.fc1 = nn.Linear(self.NUMB_FTRS_MPS*final_feature_map_dim*final_feature_map_dim, 256)  #(in_features=32*32*80)

        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features = 64, out_features = num_classes)
    

    def forward(self, x):

        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        # First block
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)
        x = self.batch1(x)
        
        # Second block
        x = self.conv1x1_2(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.batch2(x)
        
        #x = F.max_pool2d(x, 2, 2)
        x = self.maxpool2D_1(x) #halves the dimension of the image
        
        # Third block
        x = self.conv1x1_3(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.batch3(x)
        
        # Fourth block        
        x = self.conv1x1_4(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.ReLU(x)              
        #x = self.batch4(x) # qui escono valori minimi negativi, problema
        #print(f"x after batch4: MIN ({torch.min(x)}) and MAX ({torch.max(x)})")
        
        #x = F.max_pool2d(x, 2, 2)
        x = self.maxpool2D_2(x) #halves the dimension of the image

        ##TODO 29 maggio, vedi sopra
        # Fifth block
        x = self.conv1x1_5(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.ReLU(x)              
        #x = self.batch5(x)

        # Sixth block
        x = self.conv1x1_6(x)
        x = self.ReLU(x)
        x = self.conv6(x)
        x = self.ReLU(x)              
        #x = self.batch6(x)

        #x = F.max_pool2d(x, 2, 2)
        x = self.maxpool2D_3(x) #halves the dimension of the image
        #print(f"x after maxpool2D_3: MIN ({torch.min(x)}) and MAX ({torch.max(x)})")

        if torch.isnan(x).any():
            corrotto_con_nan

        if self.causality_aware: #########implement causality awareness
            feature_maps = x.detach().clone()
            if feature_maps.get_device() != x.get_device():
                feature_maps.to(x.get_device())
            b = feature_maps.size()[0] # number of images in the batch
            k = feature_maps.size()[1] # number of feature maps
            #print(f"{b} images in the batch, each has {k} feature maps")
            
            batch_causality_maps = torch.zeros((b,k,k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
            for b_i in range(b):
                #print(b_i)
                current_feature_maps = feature_maps[b_i,:,:,:]
                
                if torch.isnan(current_feature_maps).any():
                    print(f"in b_i {b_i} la current feature maps e corrotto_con_nan")
                    esci
                
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                MAX_F = torch.max(maximum_values)
                current_feature_maps = torch.nan_to_num(current_feature_maps/MAX_F, nan = 0.0)

                ## After having normalized the feature maps, comes the distinction between the method by which computing causality
                if self.causality_method == "max": #Option 1 : max values
                    sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
                    if torch.sum(torch.isnan(sum_values))>0:
                        sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
                    maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                    ###   
                    
                    mtrx = torch.outer(maximum_values, maximum_values) #the max value of mtrx object is 1.0
                    tmp = torch.div(mtrx, sum_values)
                    batch_causality_maps[b_i] = torch.nan_to_num(tmp, nan = 0.0)
                    #print(f"batch_causality_maps[b_i]: MIN ({torch.min(batch_causality_maps[b_i])}) and MAX ({torch.max(batch_causality_maps[b_i])})")   
                
                elif self.causality_method == "lehmer": #Option 2 : Lehmer mean
                    #torch.set_printoptions(threshold=10_000)
                    #torch.set_printoptions(edgeitems=32)

                    current_feature_maps = torch.flatten(current_feature_maps,1) # [k,n*n], eg [32, 8*8]    
                    #([32, 64])

                    #compute the outer product of all the pairs of flattened featuremaps.
                    # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
                    cross_matrix = torch.einsum('ai,bj->abij', current_feature_maps, current_feature_maps) #eg, [32,32,64,64]  symmetric values                  
                    #([32, 32, 64, 64])
                    cross_matrix = cross_matrix.flatten(2) #eg, [32,32,4096]
                    #([32, 32, 4096])
                    
                    # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
                    # first, compute the two powers of the cross matrix
                    p_plus_1_powers = torch.pow(cross_matrix, self.LEHMER_PARAM + 1)
                    p_powers = torch.pow(cross_matrix, self.LEHMER_PARAM)
                    numerators = torch.sum(p_plus_1_powers, dim=2)
                    #([32, 32])
                    denominators = torch.sum(p_powers, dim=2)
                    #([32, 32])
                    lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators), nan=0)
                    #([32, 32])
                    #
                    
                    
                    # then the lehmer denominator of the causality map:
                    # it is the lehemr mean of the single feature map, for all the feature maps by column
                    p_plus_1_powers_den = torch.pow(current_feature_maps, self.LEHMER_PARAM + 1)
                    #([32, 64])
                    p_powers_den = torch.pow(current_feature_maps, self.LEHMER_PARAM)
                    #([32, 64])
                    numerators_den = torch.sum(p_plus_1_powers_den, dim=1)
                    #([32])
                    denominators_den = torch.sum(p_powers_den, dim=1)
                    #([32])
                    lehmer_denominator = torch.nan_to_num(torch.div(numerators_den,denominators_den), nan=0)
                    #([32])
                    #print(lehmer_denominator)
                    
                    #and finally obtain the causality map values by computing the division
                    batch_causality_maps[b_i] = torch.nan_to_num(torch.div(lehmer_numerators, lehmer_denominator), nan=0)
                    #[32, 32]
                    #print(batch_causality_maps[b_i])
                    



                    #apply lehmer mean function iteratively over all the k featuremaps
                    #Questa la versione funzionante ma molto lunga... da ottimizzare
                    """ denominator = torch.tensor(list(map(functools.partial(lehmer_mean, p=self.LEHMER_PARAM), current_feature_maps)), device=batch_causality_maps.get_device())

                    for k_i in range(k):
                        F_i = current_feature_maps[k_i]
                        for k_j in range(k):
                            batch_causality_maps[b_i,k_i,k_j] = lehmer_mean(torch.mul(F_i, current_feature_maps[k_j]).flatten(0), self.LEHMER_PARAM)
                    """
                    


                    #batch_causality_maps[b_i] = torch.nan_to_num(torch.div(batch_causality_maps[b_i], denominator), nan=0) 
                    #print(f"batch_causality_maps[b_i]: MIN ({torch.min(batch_causality_maps[b_i])}) and MAX ({torch.max(batch_causality_maps[b_i])})")   
                    """proposta da bing, ma non la ho indagata:
                      def optimize_loop(k, current_feature_maps, batch_causality_maps, LEHMER_PARAM):
                        F = torch.stack(current_feature_maps) # shape (k, n*n)
                        M = torch.einsum('ik,jk->ijk', F, F) # shape (k, k, n*n)
                        L = lehmer_mean(M, LEHMER_PARAM) # shape (k, k)
                        batch_causality_maps = L
                    return batch_causality_maps """
                else:
                    print(self.causality_method)
                    raise NotImplementedError
            
            batch_causality_maps_flat = torch.flatten(batch_causality_maps, 1) # flatten all dimensions except batch, results in [bsize,16x16]
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = torch.cat((x,batch_causality_maps_flat), dim=1)

            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc3(x)
            x = self.drop(x)
            return x, batch_causality_maps
        
        else: #traditional, non causal:
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc3(x)
            x = self.drop(x)
            return x, None


########################################################################


class BaseCNN(nn.Module):
    def __init__(self, dim = 128, channels = 1, num_classes = 2, causality_aware=False):
        super(BaseCNN, self).__init__()

        self.causality_aware = causality_aware

        # 1x1 convolution
        
        #3x3 + 3x3 convolution (5x5 convolution)--> 1 block
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=16, 
                                kernel_size=3, padding = 1)
        self.conv1_2 =  nn.Conv2d(in_channels=16, out_channels=32, 
                                kernel_size=3, padding = 1)
        self.ReLU = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(32)
        
        # 3x3 convolution --> 2 block
        self.conv1x1_2 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=48, 
                                kernel_size=3, padding = 1)
        self.batch2 = nn.BatchNorm2d(48)
        
        # max pooling
        
        # 3x3 convolution --> 3 block
        self.conv1x1_3 = nn.Conv2d(in_channels = 48, out_channels = 16, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, 
                                kernel_size=3, padding = 1)

        self.batch3 = nn.BatchNorm2d(64)
        
        # 4 block
        self.conv1x1_4 = nn.Conv2d(in_channels = 64, out_channels = 16, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=80, 
                                kernel_size=3, padding = 1)
        self.batch4 = nn.BatchNorm2d(80)
        
    
        
        # Fully connected layers + dropout
        self.drop = nn.Dropout(0.4)

        number_of_pooling = 2
        final_feature_map_dim = round(dim/(2**number_of_pooling)) #con img 128x128, venendo dimezzata ad ogni pooling, ottengo 32x32
        
        if self.causality_aware:  
            self.fc1 = nn.Linear(80*final_feature_map_dim*final_feature_map_dim + 80*80, 256)  #(in_features=32*32*80)
        else:
            self.fc1 = nn.Linear(80*final_feature_map_dim*final_feature_map_dim, 256)  #(in_features=32*32*80)

        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features = 64, out_features = num_classes)
    

    def forward(self, x):

        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        # First block
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.ReLU(x)
        x = self.batch1(x)
        
        # Second block
        x = self.conv1x1_2(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.batch2(x)
        
        x = F.max_pool2d(x, 2, 2)
        
        # Third block
        x = self.conv1x1_3(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.batch3(x)
        
        # Fourth block
        
        x = self.conv1x1_4(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.ReLU(x) 
        ## qui sono giustamente con valore minimo pari a zero
             
        #x = self.batch4(x)
        # qui escono valori minimi negativi, problema
        #print(f"x after batch4: MIN ({torch.min(x)}) and MAX ({torch.max(x)})")
        
        x = F.max_pool2d(x, 2, 2)

        if torch.isnan(x).any():
            corrotto_con_nan

        if self.causality_aware: #########implement causality awareness
            feature_maps = x.detach().clone()
            if feature_maps.get_device() != x.get_device():
                feature_maps.to(x.get_device())
            b = feature_maps.size()[0] # number of images in the batch
            k = feature_maps.size()[1] # number of feature maps
            
            batch_causality_maps = torch.zeros((b,k,k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
            for b_i in range(b):
                current_feature_maps = feature_maps[b_i,:,:,:]
                
                if torch.isnan(current_feature_maps).any():
                    print(f"in b_i {b_i} la current feature maps e corrotto_con_nan")
                    esci
                
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                MAX_F = torch.max(maximum_values)
                current_feature_maps = torch.nan_to_num(torch.div(current_feature_maps,MAX_F), nan = 0.0)
                sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
                if torch.sum(torch.isnan(sum_values))>0:
                    sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                ###   
                
                mtrx = torch.outer(maximum_values, maximum_values) #the max value of mtrx object is 1.0
                tmp = torch.div(mtrx, sum_values)
                batch_causality_maps[b_i] = torch.nan_to_num(tmp, nan = 0.0)
                #print(f"batch_causality_maps[b_i]: MIN ({torch.min(batch_causality_maps[b_i])}) and MAX ({torch.max(batch_causality_maps[b_i])})")   
            
            batch_causality_maps_flat = torch.flatten(batch_causality_maps, 1) # flatten all dimensions except batch, results in [bsize,16x16]
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = torch.cat((x,batch_causality_maps_flat), dim=1)

            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc3(x)
            x = self.drop(x)
            return x, batch_causality_maps
        
        else: #traditional, non causal:
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
            x = self.fc1(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.ReLU(x)
            x = self.drop(x)
            x = self.fc3(x)
            x = self.drop(x)
            return x, None


############################################################################

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
            self.causality_setting = causality_setting #

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

    def forward(self, x):
        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        # print(f"x[0] (forward):\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
        x = self.starting_block(x)
        # print(f"x[0] (starting_block):\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
        x_layer1 = self.layer1(x)
        # print(f"x_layer1  (layer1):\t MIN:{x_layer1[0].min()}\t MAX:{x_layer1[0].max()}\t AVG:{torch.mean(x_layer1[0].float())}\t MED:{torch.median(x_layer1[0].float())}")
        x_layer2 = self.layer2(x_layer1)
        # print(f"x_layer2(x_layer2):\t MIN:{x_layer2[0].min()}\t MAX:{x_layer2[0].max()}\t AVG:{torch.mean(x_layer2[0].float())}\t MED:{torch.median(x_layer2[0].float())}")
        x_layer3 = self.layer3(x_layer2)
        # print(f"x_layer3(x_layer3):\t MIN:{x_layer3[0].min()}\t MAX:{x_layer3[0].max()}\t AVG:{torch.mean(x_layer3[0].float())}\t MED:{torch.median(x_layer3[0].float())}")
        x_layer4 = self.layer4(x_layer3)
        # print(f"x_layer4(x_layer4):\t MIN:{x_layer4[0].min()}\t MAX:{x_layer4[0].max()}\t AVG:{torch.mean(x_layer4[0].float())}\t MED:{torch.median(x_layer4[0].float())}")
        x = self.ending_block(x_layer4)
        # print(f"x at -ending_block:\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
        
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
            # x, causality_maps = self.get_causality_maps(x) # code for computing causality maps given a batch of featuremaps x
            x, causality_maps = GetCausalityMaps(x, self.causality_method, self.LEHMER_PARAM)

            if self.causality_setting == "cat":
                x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
            elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"):
                b = x.size()[0] # number of images in the batch
                k = x.size()[1] # number of feature maps
                x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

                for n in range(causality_maps.size()[0]): #batch size
                    causality_map = causality_maps[n]
                    triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
                    tril = torch.tril(causality_map, -1).T #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
                    bool_ij = (tril>triu).T
                    bool_ji = (triu>tril)
                    bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
                    by_col = torch.sum(bool_matrix, 1)
                    by_row = torch.sum(bool_matrix, 0)

                    if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
                        if self.causality_setting == "mulcat":
                            causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
                        elif self.causality_setting == "mulcatbool":
                            causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
                    elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
                        if self.causality_setting == "mulcat":
                            causes_mul_factors = by_row - by_col # 
                        elif self.causality_setting == "mulcatbool":
                            causes_mul_factors = 1.0*((by_row - by_col)>0) # 
                    else:
                        raise ValueError
                
                    x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
                    x_c[n] = self.relu(x_causes) #rectify every negative value to zero
                    
                x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
            
        else: #traditional, non causal:
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
        x = self.classifier(x)
        x = self.softmax(x) #TODO 12 luglio 2023
        # print(x)
        return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
    
# TODO 6 settembre 2023:
from torchvision.models import convnext_tiny, convnext_base
from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
class ResNextCA(nn.Module):
    def __init__(self, which_resnext, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            
            super(ResNextCA, self).__init__()            
                
            self.which_resnext = which_resnext
            self.img_size = dim
            self.channels = channels
            self.num_classes = num_classes
            self.is_pretrained = is_pretrained
            self.is_feature_extractor = is_feature_extractor
            
            self.causality_aware = causality_aware
            self.causality_method = causality_method
            self.LEHMER_PARAM = LEHMER_PARAM
            self.causality_setting = causality_setting #

            self.visual_attention = visual_attention #boolean
            self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

            if self.which_resnext=="tiny":
                self.first_output_ftrsmap_number = 96
                if self.is_pretrained:
                    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                else:
                    model = convnext_tiny()
            elif self.which_resnext=="base":
                self.first_output_ftrsmap_number = 128
                if self.is_pretrained:
                    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
                else:
                    model = convnext_base()

            if self.channels == 1:
                model.features[0][0] = nn.Conv2d(1, self.first_output_ftrsmap_number, kernel_size=4, stride=4)
            elif self.channels==3:
                if self.is_feature_extractor:
                    for param in model.parameters(): #freeze the extraction layers
                        param.requires_grad = False
            
            ## creating the structure of our custom model starting from the original building blocks of Resnext:
            # layer0, layer1, layer2, layer3, layer4, layer5, layer6,layer7, and classifier
            self.layer0 = model.features[0] #
            self.layer1 = model.features[1] #
            self.layer2 = model.features[2] #
            self.layer3 = model.features[3] #
            self.layer4 = model.features[4] #
            self.layer5 = model.features[5] #
            self.layer6 = model.features[6] #
            self.layer7 = model.features[7] #

            model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
            self.avgpool = model.avgpool
            
            ##self.features = model

            # if self.visual_attention: #TODO commentato per ora, capire quali layer attenzionare a differenza di resnet18
            #     ## define the attention blocks
            #     self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
            #     self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

            self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.
            if self.which_resnext=="tiny":
                self.last_ftrmap_number = 768
            elif self.which_resnext=="base":
                self.last_ftrmap_number = 1024

            if self.causality_aware:
                if self.causality_setting == "cat": #[1, n*n*k + k*k]
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes)
                elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
                    self.relu = nn.ReLU()
                    self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * self.last_ftrmap_number, self.num_classes)
                
            else: 
                if self.visual_attention: ##TODO False per ora, capire quali dimensioni al posto di 12 e 6 in base ai layer che si scelgono; TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
                    self.classifier = nn.Linear(128*12*12 + 256*6*6 + self.last_ftrmap_number*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
                else:
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

            self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added

    def forward(self, x):
        
        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        # print(f"resnext feature size:\t {x.size()}") #torch.Size([N, 1, 96, 96]) #esempio
        x_layer0 = self.layer0(x)
        # print(f"resnext x_layer0 size:\t {x_layer0.size()}") torch.Size([N, 128, 24, 24]) ##esempio, valori con ResNext BASE
        x_layer1 = self.layer1(x_layer0)
        # print(f"resnext x_layer1 size:\t {x_layer1.size()}") torch.Size([N, 128, 24, 24])
        x_layer2 = self.layer2(x_layer1)
        # print(f"resnext x_layer2 size:\t {x_layer2.size()}")  torch.Size([N, 256, 12, 12])
        x_layer3 = self.layer3(x_layer2)
        # print(f"resnext x_layer3 size:\t {x_layer3.size()}")  torch.Size([N, 256, 12, 12])
        x_layer4 = self.layer4(x_layer3)
        # print(f"resnext x_layer4 size:\t {x_layer4.size()}") torch.Size([N, 512, 6, 6])
        x_layer5 = self.layer5(x_layer4)
        # print(f"resnext x_layer5 size:\t {x_layer5.size()}") torch.Size([N, 512, 6, 6])
        x_layer6 = self.layer6(x_layer5)
        # print(f"resnext x_layer6 size:\t {x_layer6.size()}") torch.Size([N, 1024, 3, 3])
        x_layer7 = self.layer7(x_layer6)
        # print(f"resnext x_layer7 size:\t {x_layer7.size()}") torch.Size([N, 1024, 3, 3])
        x = self.avgpool(x_layer7)
        # print(f"resnext avgpool size:\t {x.size()}") #torch.Size([N, 1024, 3, 3]) <-- siccome avevo messo Identity
        

        # compute the attention probabilities:
        # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
        # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

        #check nan
        if torch.isnan(x).any():
            corrotto_con_nan
        if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]:
            print("Correcting shape mismatch...(in forward, after avgpool, before causality module)")
            x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size))

        causality_maps = None #initialized to none

        if self.causality_aware:
            # x, causality_maps = self.get_causality_maps(x) # code for computing causality maps given a batch of featuremaps x
            x, causality_maps = GetCausalityMaps(x, self.causality_method, self.LEHMER_PARAM)

            if self.causality_setting == "cat":
                x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
            elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"):
                b = x.size()[0] # number of images in the batch
                k = x.size()[1] # number of feature maps
                x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

                for n in range(causality_maps.size()[0]): #batch size
                    causality_map = causality_maps[n]
                    triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
                    tril = torch.tril(causality_map, -1).T.contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
                    bool_ij = (tril>triu).T.contiguous()
                    bool_ji = (triu>tril)
                    bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
                    by_col = torch.sum(bool_matrix, 1)
                    by_row = torch.sum(bool_matrix, 0)

                    if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
                        if self.causality_setting == "mulcat":
                            causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
                        elif self.causality_setting == "mulcatbool":
                            causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
                    elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
                        if self.causality_setting == "mulcat":
                            causes_mul_factors = by_row - by_col # 
                        elif self.causality_setting == "mulcatbool":
                            causes_mul_factors = 1.0*((by_row - by_col)>0) # 
                    else:
                        raise ValueError
                
                    x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
                    x_c[n] = self.relu(x_causes) #rectify every negative value to zero
                    
                x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
            
        else: #traditional, non causal:
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
        x = self.classifier(x)
        x = self.softmax(x) #TODO 12 luglio 2023
        # print(x)
        return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
    
class ResNext_Base_CA(nn.Module):
    def __init__(self, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            super(ResNext_Base_CA, self).__init__()
            self.img_size = dim
            self.channels = channels
            self.num_classes = num_classes
            self.is_pretrained = is_pretrained
            self.is_feature_extractor = is_feature_extractor
            
            self.causality_aware = causality_aware
            self.causality_method = causality_method
            self.LEHMER_PARAM = LEHMER_PARAM
            self.causality_setting = causality_setting #

            self.visual_attention = visual_attention #boolean
            self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

            if self.is_pretrained:
                model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            else:
                model = convnext_base()

            if self.channels == 1:
                model.features[0][0] = nn.Conv2d(1, 128,kernel_size=4, stride=4)
            elif self.channels==3:
                if self.is_feature_extractor:
                    for param in model.parameters(): #freeze the extraction layers
                        param.requires_grad = False
            
            ## creating the structure of our custom model starting from the original building blocks of Resnext:
            # layer0, layer1, layer2, layer3, layer4, layer5, layer6,layer7, and classifier
            self.layer0 = model.features[0] #
            self.layer1 = model.features[1] #
            self.layer2 = model.features[2] #
            self.layer3 = model.features[3] #
            self.layer4 = model.features[4] #
            self.layer5 = model.features[5] #
            self.layer6 = model.features[6] #
            self.layer7 = model.features[7] #

            model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
            self.avgpool = model.avgpool
            
            ##self.features = model

            # if self.visual_attention: #TODO commentato per ora, capire quali layer attenzionare a differenza di resnet18
            #     ## define the attention blocks
            #     self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
            #     self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

            self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.

            if self.causality_aware:

                # self.get_causality_maps = GetCausalityMaps(self.causality_method, self.LEHMER_PARAM)

                if self.causality_setting == "cat": #[1, n*n*k + k*k]
                    self.classifier = nn.Linear(1024 * self.last_ftrmap_size * self.last_ftrmap_size + 1024*1024, self.num_classes)
                elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
                    self.relu = nn.ReLU()
                    self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * 1024, self.num_classes)
                
            else: 
                if self.visual_attention: ##TODO False per ora, capire quali dimensioni al posto di 12 e 6 in base ai layer che si scelgono; TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
                    self.classifier = nn.Linear(128*12*12 + 256*6*6 + 1024*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
                else:
                    self.classifier = nn.Linear(1024 * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

            self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added

    def forward(self, x):
        
        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        # print(f"resnext feature size:\t {x.size()}") #torch.Size([N, 1, 96, 96])
        x_layer0 = self.layer0(x)
        # print(f"resnext x_layer0 size:\t {x_layer0.size()}") torch.Size([N, 128, 24, 24])
        x_layer1 = self.layer1(x_layer0)
        # print(f"resnext x_layer1 size:\t {x_layer1.size()}") torch.Size([N, 128, 24, 24])
        x_layer2 = self.layer2(x_layer1)
        # print(f"resnext x_layer2 size:\t {x_layer2.size()}")  torch.Size([N, 256, 12, 12])
        x_layer3 = self.layer3(x_layer2)
        # print(f"resnext x_layer3 size:\t {x_layer3.size()}")  torch.Size([N, 256, 12, 12])
        x_layer4 = self.layer4(x_layer3)
        # print(f"resnext x_layer4 size:\t {x_layer4.size()}") torch.Size([N, 512, 6, 6])
        x_layer5 = self.layer5(x_layer4)
        # print(f"resnext x_layer5 size:\t {x_layer5.size()}") torch.Size([N, 512, 6, 6])
        x_layer6 = self.layer6(x_layer5)
        # print(f"resnext x_layer6 size:\t {x_layer6.size()}") torch.Size([N, 1024, 3, 3])
        x_layer7 = self.layer7(x_layer6)
        # print(f"resnext x_layer7 size:\t {x_layer7.size()}") torch.Size([N, 1024, 3, 3])
        x = self.avgpool(x_layer7)
        # print(f"resnext avgpool size:\t {x.size()}") torch.Size([N, 1024, 3, 3]) <-- siccome avevo messo Identity
        

        # compute the attention probabilities:
        # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
        # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

        #check nan
        if torch.isnan(x).any():
            corrotto_con_nan
        if list(x.size()) != [int(x.size(0)), 1024, self.last_ftrmap_size, self.last_ftrmap_size]:
            x = torch.reshape(x, (int(x.size(0)), 1024, self.last_ftrmap_size, self.last_ftrmap_size))

        causality_maps = None #initialized to none

        if self.causality_aware:
            # x, causality_maps = self.get_causality_maps(x) # code for computing causality maps given a batch of featuremaps x
            x, causality_maps = GetCausalityMaps(x, self.causality_method, self.LEHMER_PARAM)

            if self.causality_setting == "cat":
                x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
            elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"):
                b = x.size()[0] # number of images in the batch
                k = x.size()[1] # number of feature maps
                x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

                for n in range(causality_maps.size()[0]): #batch size
                    causality_map = causality_maps[n]
                    triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
                    tril = torch.tril(causality_map, -1).T #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
                    bool_ij = (tril>triu).T
                    bool_ji = (triu>tril)
                    bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
                    by_col = torch.sum(bool_matrix, 1)
                    by_row = torch.sum(bool_matrix, 0)

                    if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
                        if self.causality_setting == "mulcat":
                            causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
                        elif self.causality_setting == "mulcatbool":
                            causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
                    elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
                        if self.causality_setting == "mulcat":
                            causes_mul_factors = by_row - by_col # 
                        elif self.causality_setting == "mulcatbool":
                            causes_mul_factors = 1.0*((by_row - by_col)>0) # 
                    else:
                        raise ValueError
                
                    x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
                    x_c[n] = self.relu(x_causes) #rectify every negative value to zero
                    
                x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
            
        else: #traditional, non causal:
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
        x = self.classifier(x)
        x = self.softmax(x) #TODO 12 luglio 2023
        # print(x)
        return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
    
""" class Resnet18CA(nn.Module):
    def __init__(self, channels, num_classes, is_pretrained, is_feature_extractor, img_size, causality_aware):
            super(Resnet18CA, self).__init__()
            self.channels = channels
            self.num_classes = num_classes
            self.is_pretrained = is_pretrained
            self.is_feature_extractor = is_feature_extractor
            self.img_size = img_size
            self.causality_aware = causality_aware

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

            model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
            model.fc = Identity() # Cancel classification layer to get only feature extractor
            
            self.features = model

            self.last_ftrmap_size = int(img_size/(2**5))

            if self.causality_aware:
                self.classifier = nn.Linear(512 * self.last_ftrmap_size * self.last_ftrmap_size + 512*512, self.num_classes)
            else:
                self.classifier = nn.Linear(512 * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)


    def forward(self, x):
        if torch.isnan(x).any():
            l_ingresso_era_gia_corrotto_con_nan

        x = self.features(x)
        if torch.isnan(x).any():
            corrotto_con_nan

        if list(x.size()) != [int(x.size(0)), 512, self.last_ftrmap_size, self.last_ftrmap_size]:
            x = torch.reshape(x, (int(x.size(0)), 512, self.last_ftrmap_size, self.last_ftrmap_size))

        if self.causality_aware:
            #########implement causality awareness
            feature_maps = x.detach().clone()
            if feature_maps.get_device() != x.get_device():
                print("Entro a modificare il device, perdita di tempo")
                feature_maps.to(x.get_device())
            b = feature_maps.size()[0] # number of images in the batch
            k = feature_maps.size()[1] # number of feature maps
            
            #batch_causality_maps = torch.empty((b,k,k), device=x.get_device())
            batch_causality_maps = torch.zeros((b,k,k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
            for b_i in range(b):

                current_feature_maps = feature_maps[b_i,:,:,:]
                #assert not torch.isnan(current_feature_maps).any()
                if torch.isnan(current_feature_maps).any():
                    print(f"in b_i {b_i} la current feature maps e corrotto_con_nan")
                    esci
                
                #compute the MAX of each feature map, return a 1D tensor of size 512. 
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                # and then compute the maximum of all maximum values, ie MAX(F)
                MAX_F = torch.max(maximum_values)
                #normalize each feature map with that MAX_F value, return a probability map
                current_feature_maps = torch.nan_to_num(torch.div(current_feature_maps,MAX_F), nan = 0.0)
                #compute the SUM of each feature map, return a 1D tensor of size 512. 
                sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
                if torch.sum(torch.isnan(sum_values))>0:
                    sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
                #compute the MAX of each new feature map, return a 1D tensor of size 512. 
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                #first compute the outer product between that tensor and itself, return a matrix (2D tensor) with all possible elementwise multiplications
                mtrx = torch.outer(maximum_values, maximum_values)
                # and then divide the rows 1:k for the vector containing SUM1:SUMk, and add the result to the outer collecting tensor  
                batch_causality_maps[b_i] = torch.nan_to_num(torch.div(mtrx, sum_values), nan = 0.0)   
                #print(f"min max values of batch_causality_maps[b_i]: {torch.min(batch_causality_maps[b_i])}-{torch.max(batch_causality_maps[b_i])}")     
            
            ## [bsize, 512, 512]
            batch_causality_maps_flat = torch.flatten(batch_causality_maps, 1) # flatten all dimensions except batch, results in [bsize,16x16]
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = torch.cat((x,batch_causality_maps_flat), dim=1)

            out = self.classifier(x)
            return out, batch_causality_maps
            #########
        
        else: #traditional, non causal:
            x = torch.flatten(x, 1) # flatten all dimensions except batch 
            out = self.classifier(x)
            return out """







############################################################################

class AlexnetCA(nn.Module):
    def __init__(self, img_size, num_classes, causality_aware):
            super(AlexnetCA, self).__init__()
            self.img_size = img_size
            self.num_classes = num_classes
            self.causality_aware = causality_aware

            model = alexnet()

            self.features = model.features #end with a maxpool2d
            self.avgpool = model.avgpool #ends with a 6x6 feature map obtained via adaptiveavgpool2d

            if self.causality_aware:
                self.classifier = nn.Linear(256 * 6 * 6 + 256*256, self.num_classes)
            else:
                self.classifier = nn.Linear(256 * 6 * 6, self.num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.causality_aware:
            feature_maps = x.clone()
            """ if feature_maps.get_device() != x.get_device():
                print("Entro a modificare il device, perdita di tempo")
                feature_maps.to(x.get_device()) """
            b = feature_maps.size()[0] # number of images in the batch
            k = feature_maps.size()[1] # number of feature maps
            
            batch_causality_maps = torch.empty((b,k,k), device=x.get_device()) # It has a "kxk" causality map for each of the "b" images in the batch
            for b_i in range(b):
                current_feature_maps = feature_maps[b_i,:,:,:]
                if torch.isnan(current_feature_maps).any():
                    print(f"in b_i {b_i} la current feature maps e corrotto_con_nan")
                    esci
                
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                MAX_F = torch.max(maximum_values)
                current_feature_maps = torch.nan_to_num(torch.div(current_feature_maps,MAX_F), nan = 0.0)
                sum_values = torch.sum(torch.flatten(current_feature_maps,1), dim=1)
                if torch.sum(torch.isnan(sum_values))>0:
                    sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri
                maximum_values = torch.max(torch.flatten(current_feature_maps,1), dim=1)[0]
                mtrx = torch.outer(maximum_values, maximum_values)
                batch_causality_maps[b_i] = torch.nan_to_num(torch.div(mtrx, sum_values), nan = 0.0)        
            
            batch_causality_maps = torch.flatten(batch_causality_maps, 1) # flatten all dimensions except batch
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = torch.cat((x,batch_causality_maps), dim=1)
        else:
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x