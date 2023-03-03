#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

sudo pkill python3.6
echo "killing all previous python processes"
export NCCL_MIN_NCHANNELS=4

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <num_gpus, please enter 1 or 2 or 4 or 8>"
  exit
fi

num_gpus=${1}
precision=${3:-"fp32"}
warmup_proportion=${5:-"0.0"}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
seed=${12:-$RANDOM}
job_name=${13:-"bert_lamb_pretraining"}
train_batch_size_phase2=${17:-8}
learning_rate_phase2=${18:-"3.5e-4"}
warmup_proportion_phase2=${19:-"0"}
train_steps_phase2=${20:-15000}
allreduce_post_accumulation=${14:-"false"}
allreduce_post_accumulation_fp16=${15:-"false"}
gradient_accumulation_steps_phase2=${21:-1}
BERT_CONFIG="/workspace/bert-dataset-v1.1/bert_config.json"
CODEDIR=${24:-"./"}
init_checkpoint=${25:-"/workspace/bert-dataset-v1.1/model.ckpt-28252.pt"}
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

mkdir -p $CHECKPOINTS_DIR

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

#Start Phase2

DATA_DIR_PHASE2='/workspace/bert-dataset-v1.1/hdf5_4320_shards_varlength'
PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --bert_config_path=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=76"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
#CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $INIT_CHECKPOINT"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --opt_lamb_beta_1=${OPT_LAMB_BETA_1:-0.9}"
CMD+=" --opt_lamb_beta_2=${OPT_LAMB_BETA_2:-0.999}"
CMD+=" --weight_decay_rate=${WEIGHT_DECAY_RATE:-0.01}"
CMD+=" --do_train --phase2"
CMD+=" --train_mlm_accuracy_window_size=0"
CMD+=" --target_mlm_accuracy=0.72"
CMD+=" --fused_gelu_bias --dense_seq_output --bypass_amp --exchange_padding"
CMD+=" --eval_dir=/workspace/bert-dataset-v1.1/eval_varlength"
CMD+=" --eval_iter_start_samples=150000"
CMD+=" --eval_iter_samples=150000"
CMD+=" --eval_batch_size=16"
CMD+=" --cache_eval_data"


CMD="python -u -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished phase2"
