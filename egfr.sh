#!/usr/bin/env bash
# Training launch script — iterates over all architectures × fine-tuning modes
#
# Runs torchrun jobs sequentially for each (arch, ft) combination.
# GPU assignment and most paths can be overridden via environment variables
# before calling this script (defaults shown below).
#
# Fine-tuning modes per arch:
#   LoRA   : ft_blks=full, lora_rank=4
#   Partial: ft_blks=1
#   Linear : batch_size=32 (larger batch since only head is trained)
#
# Override examples:
#   CUDA_VISIBLE_DEVICES="0,1" BATCH_DEFAULT=4 bash run_train.sh
#   EXTRA_ARGS="--enable_amp" EPOCHS=50 bash run_train.sh


# -------------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"2,3,4,5,6,7,8,9"}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
unset TORCH_DISTRIBUTED_DEBUG


# -------------------------------------------------------------------------
# Paths — update to match your environment
# -------------------------------------------------------------------------
SCRIPT=${SCRIPT:-/path/to/trainer.py}
CSV=${CSV:-/path/to/master.csv}
IMG_DIR=${IMG_DIR:-/path/to/AutoMorph/M0/images/}
GOOD_DIR=${GOOD_DIR:-/path/to/AutoMorph/M1/Good_quality/}
OUT_DIR=${OUT_DIR:-./models_egfr}
SPLIT_DIR=${SPLIT_DIR:-./splits_egfr}
ROOT_DIR=${ROOT_DIR:-./ckpts_egfr}


# -------------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------------
IMG_SIZE_ALL=${IMG_SIZE_ALL:-448}
BATCH_DEFAULT=${BATCH_DEFAULT:-8}
BATCH_OPENCLIP=${BATCH_OPENCLIP:-8}
EPOCHS=${EPOCHS:-100}
EVAL_WORKERS=${EVAL_WORKERS:-0}
EXTRA_ARGS=${EXTRA_ARGS:-"--enable_amp"}

NP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')


# -------------------------------------------------------------------------
# Job runner
# -------------------------------------------------------------------------
run_job() {
    local ARCH="$1";     shift
    local FT_DESC="$1";  shift
    local BATCH="$1";    shift
    local EXTRA_LRS="$1"; shift

    local COMMON_ARGS="
        --img_size   ${IMG_SIZE_ALL}
        --csv        ${CSV}
        --img_dir    ${IMG_DIR}
        --good_dir   ${GOOD_DIR}
        --out_dir    ${OUT_DIR}
        --split_dir  ${SPLIT_DIR}
        --root_dir   ${ROOT_DIR}
        --epochs     ${EPOCHS}
        --batch_size ${BATCH}
        --eval_workers ${EVAL_WORKERS}
        ${EXTRA_ARGS}
        ${EXTRA_LRS}
    "

    echo ""
    echo "==============================="
    echo "ARCH=${ARCH} | ${FT_DESC} | img_size=${IMG_SIZE_ALL} | batch=${BATCH}"
    echo "==============================="
    set -x
    torchrun --standalone --nproc_per_node="${NP}" "${SCRIPT}" \
        --arch "${ARCH}" "$@" ${COMMON_ARGS}
    { set +x; } 2>/dev/null
}


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
ARCHES=("retfound" "retfound_dinov2" "openclip" "dinov2" "dinov3" "mae")

for ARCH in "${ARCHES[@]}"; do
    if [[ "${ARCH}" == "openclip" ]]; then
        BATCH="${BATCH_OPENCLIP}"
    else
        BATCH="${BATCH_DEFAULT}"
    fi

    run_job "${ARCH}" "LoRA ft_blks=full rank=4" "${BATCH}" \
        "--lr_head 3e-3 --lr_body 5e-4 --weight_decay 1e-5" \
        --ft lora --ft_blks "full" --lora_rank 4 --strong_aug

    run_job "${ARCH}" "Partial ft_blks=1" "${BATCH}" \
        "--lr_head 3e-3 --lr_body 5e-4 --weight_decay 1e-5" \
        --ft partial --ft_blks 1

    run_job "${ARCH}" "Linear" "32" \
        "--lr_head 3e-3 --weight_decay 1e-5" \
        --ft linear
done

echo "All jobs finished."