#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=6
data='./datasets'
output='./run_out/extented/'

for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi
	        python run_bert_absa.py --task_type 'absa' \
                --data_dir "${src_domain}-${tar_domain}"  \
                --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_batch_size 16 \
                --bert_model 'bert_extented' \
                --seed 42 \
                --do_train \
                --do_eval 
        fi
    done
done




