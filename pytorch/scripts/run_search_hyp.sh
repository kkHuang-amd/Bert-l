#!/bin/bash

rm -rf configs
mkdir configs
config_file_base="configs/config_HYABUSA-MI250X_128x16x1"

rm -rf results/HYABUSA/128-nodes

ulimit -n 100000

beta1=(0.71 0.75)
beta2=(0.88 0.9)

startwarmsteps=(0 -76 -100)
warmsteps=(0 256 290)

for batch in 2;do
 for lr in $(seq 0.0021 0.0004 0.0033);do
  for maxstep in 645;do
   for i in ${!beta1[@]};do
    for j in ${!warmsteps[@]};do
     for initscale in 128 1024 1048576;do
      for evalsamples in 175000 325000;do
           config_file="${config_file_base}_${batch}x${lr}_${maxstep}_${beta1[i]}x${beta2[i]}_${startwarmsteps[j]}x${warmsteps[j]}_${initscale}_${evalsamples}"
           #echo $config_file
	   echo "export BATCHSIZE=${batch}" >> ${config_file}
	   echo "export GRADIENT_STEPS=1" >> ${config_file}
	   echo "export LR=${lr}" >> ${config_file}
	   echo "export MAX_SAMPLES_TERMINATION=12000000" >> ${config_file}
	   echo "export MAX_STEPS=${maxstep}" >> ${config_file}
	   echo "export OPT_LAMB_BETA_1=${beta1[i]}" >> ${config_file}
	   echo "export OPT_LAMB_BETA_2=${beta2[i]}" >> ${config_file}
	   echo "export START_WARMUP_STEP=${startwarmsteps[j]}" >> ${config_file}
	   echo "export WARMUP_STEPS=${warmsteps[j]}" >> ${config_file}
	   echo "export WARMUP_PROPORTION=0.0" >> ${config_file}
	   echo "export WEIGHT_DECAY_RATE=0.01" >> ${config_file}
	   echo "export INIT_LOSS_SCALE=${initscale}" >> ${config_file}
	   echo "export EXTRA_PARAMS=\"--dense_seq_output --unpad --exchange_padding --fused_gelu_bias --fused_mha\"" >> ${config_file}
	   echo "export PHASE=2" >> ${config_file}
	   echo "export EVAL_ITER_START_SAMPLES=${evalsamples}" >> ${config_file}
	   echo "export EVAL_ITER_SAMPLES=${evalsamples}" >> ${config_file}
      done
     done
    done
   done
  done
 done
done


CONFIG_FILES=$(ls configs/*)

for count in $(seq 1 5); do
    mkdir -p "results/HYABUSA/128-nodes/${count}"
    echo $count
    for file in ${CONFIG_FILES}; do
	log=$(ls $file | cut -d "_" -f 2-)
	log="results/HYABUSA/128-nodes/${count}/${log}.log"

	source $file
        ./scripts/run_pretraining_rocm_fp16_n-nodes_numa.sh 8 1 2>&1 | tee $log
    done
done
