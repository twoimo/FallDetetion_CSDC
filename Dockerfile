ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt update && apt install -y git vim libgl1-mesa-glx libglib2.0-0
RUN pip install torch==2.0.0+cu117 torchaudio==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install python library
RUN pip install opencv-python
RUN pip install scipy
RUN pip install tqdm
RUN pip install natsort
RUN pip install openpyxl
RUN pip install matplotlib

# Install Project
RUN apt-get install git-lfs
RUN git clone https://github.com/twoimo/FallDetetion_CSDC
WORKDIR FallDetetion_CSDC
RUN git lfs pull

# Set the default command to run when the container starts
CMD ["bash"]