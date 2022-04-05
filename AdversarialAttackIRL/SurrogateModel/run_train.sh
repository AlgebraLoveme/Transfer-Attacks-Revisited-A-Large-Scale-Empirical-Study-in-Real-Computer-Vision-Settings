#!/bin/bash

read -p 'GPU: ' -r gpu_index
read -p 'Arch: ' -r arch
read -p 'Depth: ' -r depth
#depths=("18" "34" "50")
read -p 'Dataset: ' dataset

if [[ "$arch" == "resnet" ]]; then
  python train.py --model "${arch}" --depth "${depth}" --dataset "${dataset}" --gpu_index "${gpu_index}" --pretrained --batch_size 16
  python train.py --model "${arch}" --depth "${depth}" --dataset "${dataset}" --gpu_index "${gpu_index}" --augment --pretrained --batch_size 16
  python train.py --model "${arch}" --depth "${depth}" --dataset "${dataset}" --gpu_index "${gpu_index}" --batch_size 16
  python train.py --model "${arch}" --depth "${depth}" --dataset "${dataset}" --gpu_index "${gpu_index}" --augment --batch_size 16
else
  python train.py --model vgg --depth 16 --dataset ${dataset} --gpu_index "${gpu_index}" --pretrained --batch_size 16
  python train.py --model inception --depth V3 --dataset ${dataset} --gpu_index "${gpu_index}" --pretrained --batch_size 16
fi