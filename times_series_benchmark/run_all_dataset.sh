#!/bin/bash
# Copyright 2026 Kuo-Chung Peng and Samuel Yen-Chi Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
echo "Running all QKAN-FWP time-series benchmark experiments..."

# --- Common Arguments ---
EPOCHS=100
BATCH_SIZE=4
LR=1e-3
DATASET=('jaynes_cummings')

INPUT_SIZE=1
QNN_DEPTH=2


# --- Models to test ---
# GQKAN-QKANFWP, GQKANFWP, GQKAN-FWP, GQKAN-QFWP
MODELS=('gqkan_qkanfwp' 'gqkanfwp' 'gqkan_fwp' 'gqkan_qfwp')
# HIDDEN_SIZE=(7 6 5 4 3 2)
HIDDEN_SIZE=(8)
SEQ_LEN=(64)
# SEQ_LEN=(16)
SEEDS=(0 1 2)

# Loop over each model
for dataset in "${DATASET[@]}"; do
    for model in "${MODELS[@]}"; do
        for hidden_dim in "${HIDDEN_SIZE[@]}"; do
            for time_length in "${SEQ_LEN[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    exp_name="${dataset}_${model}_hidden_size_${hidden_dim}_qnn_depth_${QNN_DEPTH}_seq_len_${time_length}_seed_${seed}"
                    echo "RUNNING EXP: ${exp_name}"
                    echo "Training ${model} model with hidden dim ${hidden_dim}..."
                    python train.py \
                        --epochs ${EPOCHS} \
                        --lr ${LR} \
                        --model "${model}"\
                        --dataset "${dataset}" \
                        --exp_name "${exp_name}" \
                        --window_len ${time_length} \
                        --input_size ${INPUT_SIZE} \
                        --hidden_size ${hidden_dim} \
                        --qnn_depth ${QNN_DEPTH} \
                        --batch_size ${BATCH_SIZE}\
                        --seed ${seed} \
                        --device cpu
                done
            done
        done       
    done
done


echo "All experiments finished."
