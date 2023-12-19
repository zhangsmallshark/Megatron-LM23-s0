#!/bin/bash --login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_EXTENSIONS_DIR=/home/czh5/.cache/polaris_torch_extensions
# export CUDA_LAUNCH_BLOCKING=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/home/czh5/genome/Megatron-LM23-s0/checkpoint
rm -rf $CHECKPOINT_PATH/*
VOCAB_FILE="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets/gpt2-vocab.json"
MERGE_FILE="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets/gpt2-merges.txt"
DATA_PATH="/lus/eagle/projects/MDClimSim/chengming/gpt_datasets/BookCorpusDataset_text_document"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Runs the "345M" parameter model
# GPT_ARGS="
#     --tensor-model-parallel-size $WORLD_SIZE \
#     --num-layers 24 \
#     --hidden-size 1024 \
#     --num-attention-heads 16 \
#     --seq-length 1024 \
#     --max-position-embeddings 1024 \
#     --micro-batch-size 8 \
#     --global-batch-size 8 \
#     --lr 0.00015 \
#     --train-iters 200 \
#     --lr-decay-iters 320000 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --fp16
# "

# Runs the "1.3B" parameter model
GPT_ARGS="
    --tensor-model-parallel-size $WORLD_SIZE \
    --num-layers 20 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 16 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 10 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 2
"

# torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

