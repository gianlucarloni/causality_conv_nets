FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN pip install tqdm einops seaborn
##ENTRYPOINT [ "python", "./train_convnet.py"]
##ENTRYPOINT [ "python", "./train_convnet_loop.py"]
##ENTRYPOINT [ "python", "./train_convnet_loop_inner.py"]
#ENTRYPOINT [ "python", "./train_convnet_loop_inner_B.py"]
##ENTRYPOINT [ "python", "./train_convnet_loop_inner_dict.py"]
#
# ENTRYPOINT [ "python", "./train_convnet_loop_inner_B2.py"]
ENTRYPOINT [ "python", "./train_convnet_loop_inner_B2_ablation.py"]
