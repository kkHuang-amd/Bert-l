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

module av

pkill python3
echo "killing all previous python processes"

#echo "Clear page cache"
#sync && sudo /sbin/sysctl vm.drop_caches=3

NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}

export NCCL_MIN_NCHANNELS=4
#workaround for MI250 nccl issue
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <num_gpus, please enter 1 or 2 or 4 or 8>"
  exit
fi

DATA_DIR=${DATA_DIR:-"/datasets/mlperf_dataset/wiki_20200101"}
MASTER_NODE=${MASTER_NODE:-`scontrol show hostnames | head -n 1`}
echo $MASTER_NODE

echo $DATA_DIR

num_gpus=${1}
dis_fused_lamb=${2:-0}
ddp_method=${3:-"torchdpp"}
precision=${4:-"fp16"}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
job_name=${13:-"bert_lamb_pretraining"}
seed=${SEED:-$RANDOM}
train_batch_size_phase2=${BATCHSIZE:-56} ##Decrease BS to 27 to run on MI100
learning_rate_phase2=${LR:-"3.25e-4"}
train_steps_phase2=${MAX_STEPS:-15000} 
warmup_proportion_phase2=${WARMUP_PROPORTION:-"0.0"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
gradient_accumulation_steps_phase2=${GRADIENT_STEPS:-1}
BERT_CONFIG="${DATA_DIR}/bert_config.json"
CODEDIR=${24:-"./"}
init_checkpoint=${25:-"${DATA_DIR}/model.ckpt-28252.pt"}
RESULTS_DIR=${RESULT_DIR:-"$CODEDIR/results"}
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints
EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-150000}
EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-150000}

#NUMA settings
nnuma_nodes=`lscpu | grep "NUMA node(s)" | sed -e "s/.*:\s*//g"`
nnuma_nodes=$((nnuma_nodes < num_gpus ? nnuma_nodes : num_gpus))
nsockets=`lscpu | grep "Socket(s)" | sed -e "s/.*:\s*//g"`
ncores_per_socket=`lscpu | grep "Core(s) per socket:" | sed -e "s/.*:\s*//g"`
ncores=$((nsockets*ncores_per_socket))

mkdir -p $RESULTS_DIR
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

DATA_DIR_PHASE2="${DATA_DIR}/hdf5_4320_shards_varlength"
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
CMD+=" --warmup_steps=${WARMUP_STEPS:-0.0}"
CMD+=" --start_warmup_step=${START_WARMUP_STEP:-0.0}"
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
CMD+=" --do_train --phase2 --skip_checkpoint"
CMD+=" --train_mlm_accuracy_window_size=0"
CMD+=" --target_mlm_accuracy=0.720"
CMD+=" --max_samples_termination=${MAX_SAMPLES_TERMINATION:-18400000}"
CMD+=" $EXTRA_PARAMS"
#--fused_bias_fc --fused_bias_mha --fused_dropout_add
#CMD+=" --unpad_fmha" 
if [[ $dis_fused_lamb -eq 1 ]] ; then
  echo "enable distributed_lamb"
  CMD+=" --distributed_lamb --dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-ag-pg=1 --dwu-num-blocks=1"
fi
CMD+=" --eval_dir=${DATA_DIR}/eval_varlength"
CMD+=" --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES}"
CMD+=" --eval_iter_samples=${EVAL_ITER_SAMPLES}"
CMD+=" --eval_batch_size=16"
CMD+=" --cache_eval_data --num_eval_examples 10000"  
#CMD+=" --use_ddp --ddp_type=native"
CMD+=" --log_freq=1"
if [ "$ddp_method" == "deepspeed" ]; then
    CMD+=" --deepspeed --deepspeed_config ds_config.json"
fi

#CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"
CMD="python3 -m mlperf_utils.bind_launch --no_membind --nnuma_nodes $nnuma_nodes --nsockets_per_node ${nsockets} --ncores_per_socket ${ncores_per_socket} --node_rank ${SLURM_NODEID} --nnodes ${SLURM_NTASKS} --master_addr ${MASTER_NODE} --master_port 23456 --nproc_per_node=$num_gpus $CMD"
echo "CMD => ${CMD}"
#fi
hostname=`hostname`

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  if [[ $dis_fused_lamb -eq 0 ]] ; then
      printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d_lr-${learning_rate_phase2}_${hostname}_${ddp_method}" "$precision" $GBS
  else
      printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d_dis-lamb_lr-${learning_rate_phase2}_max-ch-${NCCL_MAX_NCHANNELS}_${hostname}_${ddp_method}" "$precision" $GBS
  fi
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
echo $CMD
#if [ -z "$LOGFILE" ] ; then
#   $CMD
#else
#   (
#     $CMD
#   ) |& tee $LOGFILE
#fi

set +x

echo "finished phase2"
