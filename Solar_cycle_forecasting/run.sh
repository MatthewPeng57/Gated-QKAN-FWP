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

set -uo pipefail

DATASET=sunspots
SEEDS=(0 1 2 3 4)
EPOCHS=${EPOCHS:-100}
SAVE_DIR=results
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

H=48; I=10; O=20; Q=25
LR=2.5e-3
ALPHA=1.0
LR_SCHEDULE=keras_decay
LOSS=peak_aware_mse

sweep_start=$(date +%s)
echo "GQKAN-QKANFWP — 5 seeds on ${DATASET}"
echo "Config: H=${H} I=${I} O=${O} Q=${Q} LR=${LR} α=${ALPHA}"
echo "Started: $(date)"

pids=()
for seed in "${SEEDS[@]}"; do
    log="${LOG_DIR}/gqkan_qkanfwp_seed${seed}.log"
    python train.py \
        --epochs ${EPOCHS} --lr ${LR} \
        --lr_schedule ${LR_SCHEDULE} --loss ${LOSS} \
        --model gqkan_qkanfwp --dataset ${DATASET} \
        --exp_name "gqkan_qkanfwp_seed${seed}" \
        --save_dir "${SAVE_DIR}" \
        --window_len 528 --horizon 132 \
        --input_size 1 --output_size 132 \
        --hidden_size ${H} --batch_size 32 --seed ${seed} \
        --device cuda --alpha ${ALPHA} --qnn_depth 2 \
        --in_resize ${I} --out_resize ${O} \
        --qkan_s_dim_1 ${Q} --qkan_s_dim_2 ${Q} \
        --fast_in 16 --fast_out 16 \
        > "${log}" 2>&1 &
    pids+=($!)
    # 3-way parallel.
    if (( ${#pids[@]} >= 3 )); then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do wait "${pid}" || echo "  !!! pid=${pid}"; done

echo ""
echo "SWEEP COMPLETE in $(( $(date +%s) - sweep_start ))s"
echo "  checkpoints: ${SAVE_DIR}/"
echo "  logs:        ${LOG_DIR}/"
echo "  finished:    $(date)"
