FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

LABEL maintainer.email="maksim.carkov.201300@gmail.com"
LABEL maintainer.git="s1Sharp"

EXPOSE 22

RUN apt update && apt upgrade -y
RUN apt install wget python3 python3-pip -y

WORKDIR /cnn

COPY ./src .

RUN bash ./load_dataset.sh

RUN pip3 install -r requirements.txt