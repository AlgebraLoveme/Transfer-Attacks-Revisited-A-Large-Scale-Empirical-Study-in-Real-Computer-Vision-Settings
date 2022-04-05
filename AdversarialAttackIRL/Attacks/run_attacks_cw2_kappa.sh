#!/bin/bash

read -p "GPU: " -r gpu_index
# 0 1 2 3
read -p "KappaGroup: " -r kappa_group_id
# 0 1
read -p "Pretrained: " -r is_pretrained
# 0 1

dataset="ImageNet"
type="raw"
arch="resnet"

if [[ "$kappa_group_id" == "0" ]]; then
  kappa_settings=("200" "190" "180" "170" "160" "150" "140")
else
  kappa_settings=("130" "120" "110" "100" "90" "80" "70" "60")
fi

depth=("18" "34" "50")

for kappa_value in "${kappa_settings[@]}"; do
for d in "${depth[@]}"; do
  if [[ "$is_pretrained" == "0" ]]; then
    python CW2_Generation.py --dataset_type ${type} --model ${arch} --model_depth "${d}" --dataset ${dataset} \
     --gpu_index "${gpu_index}" --kappa "${kappa_value}"
  else
    python CW2_Generation.py --dataset_type ${type} --model ${arch} --model_depth "${d}" --dataset ${dataset} \
     --gpu_index "${gpu_index}" --pretrained --kappa "${kappa_value}"
  fi
done
done
