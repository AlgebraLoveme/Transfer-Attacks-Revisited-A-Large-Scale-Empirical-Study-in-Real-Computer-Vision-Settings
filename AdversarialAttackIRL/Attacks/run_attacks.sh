#!/bin/bash

read -p "Dataset: " -r dataset
# AdienceGenderG ImageNet NudeNet
read -p "GPU: " -r gpu_index
# 0 1 2 3
read -p "Model Architecture: " -r arch
# resnet inception vgg
read -p "Data type: " -r type
# raw augmented adversarial

attacks=("BL-BFGS" "CW2" "DeepFool" "LLC" "PGD" "RFGSM" "FGSM" "STEP_LLC" "UAP")

if [[ "$arch" == "resnet" ]]; then
  depth=("18" "34" "50")
  #  data_type=("raw" "adversarial" "augmented")
  for d in "${depth[@]}"; do
  for attack in "${attacks[@]}"; do
    python ${attack}_Generation.py --dataset_type ${type} --model ${arch} --model_depth ${d} --dataset ${dataset} \
     --gpu_index ${gpu_index} --name ${attack}
    python ${attack}_Generation.py --dataset_type ${type} --model ${arch} --model_depth ${d} --dataset ${dataset} \
     --gpu_index ${gpu_index} --name ${attack} --pretrained
  done
  done
else
  for attack in "${attacks[@]}"; do
    python ${attack}_Generation.py --dataset_type raw --model vgg --model_depth 16 --dataset ${dataset} \
     --gpu_index ${gpu_index} --name ${attack} --pretrained
    python ${attack}_Generation.py --dataset_type raw --model inception --model_depth V3 --dataset ${dataset} \
     --gpu_index ${gpu_index} --name ${attack} --pretrained
  done
fi
