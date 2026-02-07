#!/bin/bash

values=(1.0 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
timestamp=$(date +"%Y%m%d_%H%M%S")
for xx in "${values[@]}"
do
    echo "Running with keep_ratio=${xx}"

    python /DCN/DCN_torch/run_expid_prefeat.py --expid DCN_feat_retrain_criteo --gpu 1 --keep_ratio ${xx} >> "DCN_feat_retrain_criteo_${timestamp}.log" 2>&1

    echo -e "\n----------------------------------------------------Running done with keep_ratio=${xx}\n" >> "DCN_feat_retrain_criteo_${timestamp}.log"
done
