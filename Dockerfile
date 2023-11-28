FROM nvcr.io/nvidia/pytorch:22.09-py3
RUN pip install tqdm einops seaborn scikit-learn==1.2.0
ENTRYPOINT [ "python", "./train_convnet_loop_inner_B2.py"]
# ENTRYPOINT [ "python", "./train_convnet_loop_inner_B2_ABL.py"]