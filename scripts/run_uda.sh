#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=0
declare -A dic
dic=(["service-device"]="device-service" ["service-laptop"]="laptop-service" ["device-rest"]="rest-device" ["laptop-rest"]="rest-laptop" ["service-rest"]="rest-service")

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
            if ! [[ "${dic[*]}" =~ "$src_domain-$tar_domain" ]];
            then
                pair="${dic[$src_domain-$tar_domain]}"
            else
                pair="$src_domain-$tar_domain"
            fi

			python run_uda.py --task_type 'absa' \
                --data_dir "${src_domain}-${tar_domain}"  \
                --output_dir "./run_out/uda-${src_domain}-${tar_domain}"  \
                --bert_model "bert_base"\
                --train_batch_size 16 \
                --domain_dataset "./datasets/unlabel/${pair}-rel" \
                --features_model "./out_feature_models/base-${pair}/epoch2/model.pt" \
                --seed 42 \
                --do_train \
                --do_eval 
        fi
    done
done




