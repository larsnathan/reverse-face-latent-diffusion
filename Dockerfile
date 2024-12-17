# Dockerfile
# FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    libcurl4 liberror-perl \
    && rm -rf /var/lib/apt/lists/*

    # Install git and its requirements
WORKDIR /git
RUN wget https://apllinuxdepot.jhuapl.edu/linux/GIT-Openssl/git-2.30.2-20.04.tar && \
    tar -xvf git-2.30.2-20.04.tar && \
    dpkg -i git-man_2.30.2-0ppa1~ubuntu20.04.1_all.deb && \
    dpkg -i git_2.30.2-0ppa1~ubuntu20.04.1_amd64.deb

COPY requirements.txt /app/requirements.txt
WORKDIR /app
# RUN pip3 --default-timeout=1000 install -U packaging torch
RUN pip3 --default-timeout=1000 install --no-cache-dir -r requirements.txt

# Copy the service code
COPY diffusers_retrain.py /app/diffusers_retrain.py

# Run the application
CMD ["python3", "diffusers_retrain.py"]
