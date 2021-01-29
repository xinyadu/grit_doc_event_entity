#!/usr/bin/env bash

export MAX_LENGTH_SRC=470
export MAX_LENGTH_TGT=40
export BERT_MODEL=bert-base-uncased

export BATCH_SIZE=1
export NUM_EPOCHS=3
export SEED=1

export OUTPUT_DIR_NAME=model
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

for th in 100 # 1 10 100 1000 10000 100000, 100 200 300 400 500
do
echo "=========================================================================================="
echo "                                           threshold (${th})                              "
echo "=========================================================================================="
python3 run_pl_s_t.py  \
--data_dir ../data/muc/processed \
--model_type bert \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length_src  $MAX_LENGTH_SRC \
--max_seq_length_tgt $MAX_LENGTH_TGT \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--seed $SEED \
--n_gpu 0 \
--debug \
--do_predict \
--thresh $th \
# --do_train 
done