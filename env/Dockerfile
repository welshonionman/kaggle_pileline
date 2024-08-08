FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get upgrade -y
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.10 get-pip.py

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends\
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        libzmq3-dev \
        libspatialindex-dev \
        libsm6 \
        libgl1-mesa-dev \
        vim \
        git \
        curl \
        wget \
        zip \
        zsh \
        openssh-server \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY vscode_extentions.sh /tmp/vscode_extentions.sh
COPY .bashrc /root/.bashrc

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt
RUN bash /tmp/vscode_extentions.sh && \
    rm -rf /tmp/vscode_extentions.sh


RUN pip3 install --upgrade pip setuptools
RUN pip3 install jupyter

RUN mkdir /kaggle

RUN apt-get install -y

RUN mkdir /var/run/sshd
RUN echo 'root:welshonionman' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication/PasswordAuthentication/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
WORKDIR /kaggle

ENTRYPOINT ["/usr/sbin/sshd", "-D"]

