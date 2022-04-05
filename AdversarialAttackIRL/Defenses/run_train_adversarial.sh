#!/bin/bash

read -p 'GPU: ' -r gpu_index
read -p 'Depth: ' -r depth
#depths=("18" "34" "50")
read -p 'Dataset: ' dataset
read -p 'Arch: ' arch

python NAT_Test.py --model "${arch}" --depth "${depth}" --dataset ${dataset} --gpu_index "${gpu_index}" --pretrained --batch_size 8
python NAT_Test.py --model "${arch}" --depth "${depth}" --dataset ${dataset} --gpu_index "${gpu_index}" --batch_size 8
