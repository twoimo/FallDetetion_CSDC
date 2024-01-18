ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

WORKDIR /FallDetetion_CSDC

RUN apt update && apt install -y git  vim libgl1-mesa-glx 
RUN apt install libglib2.0-0

# Install python library
RUN pip install opencv-python
RUN pip install scipy
RUN pip install tqdm
RUN pip install natsort
RUN pip install openpyxl
RUN pip install matplotlib

# Private git clone config
RUN git config --global user.name "FallDetetion_CSDC"
RUN git config --global user.email "twoimo@dgu.ac.kr"

# Install Project
RUN git clone https://github.com/twoimo/FallDetetion_CSDC
RUN pip install -r requirements.txt

# Set the default command to run when the container starts
CMD ["bash"]