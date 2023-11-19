# causality_conv_nets
This is repository contains the code to experiment with our framework of Causality-driven Convolutional Neural Networks.

[[**Conference Paper**](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Carloni_Causality-Driven_One-Shot_Learning_for_Prostate_Cancer_Grading_from_MRI_ICCVW_2023_paper.html)] [[**Journal Paper**](https://arxiv.org/abs/2309.10399)]

## Main idea and related work

The rationale behind the whole project is the concept of _causal disposition_ from [Lopez-Paz, D. (2017)](https://github.com/gianlucarloni/causality_conv_nets/assets/91902479/a4040479-d4ef-4e6b-afc5-07fb73018f71).
Given an image dataset, we can have insights into observable footprints that reveal the dispositions of the object categories appearing in the images.
For instance, if two objects/artifacts _A_ and _B_ are present in the images, we can define the causal disposition of _A_ w.r.t. _B_ by counting the number of images in the dataset where if we remove _A_ then _B_ also disappear.

<img src="./car_bridge.png" width=200 height=200>

**Intuition**: any causal disposition induces a set of conditional asymmetries between the artifacts from an image (features, object categories, etc.) that represent (weak) causality signals regarding the real-world scene. --> Can computer vision models infer such asymmetries autonomously?

[Terziyan and Vitko (2023)](https://www.sciencedirect.com/science/article/pii/S1877050922023237) suggests a way to compute estimates for possible causal relationships within images via CNNs. 
When a feature map $F^i$ contains only non-negative numbers (e.g., thanks to ReLU functions) and is normalized in the interval $[0,1]$, we can interpret its values as probabilities of that feature to be present in a specific location. For instance, $F^i_{r,c}$ is the probability that the feature $i$ is recognized at coordinates ${r,c}$.
By assuming that the last convolutional layer outputs and localizes to some extent the object-like features, we may modify the architecture of a CNN such that the $n \times n$ feature maps ($F^1,F^2,\dots F^k$) obtained from that layer got fed into a new module that computes pairwise conditional probabilities of the feature maps. The resulting $k \times k$ map would represent the causality estimates for the features and be called **causality map**. 


## Get started 

...
