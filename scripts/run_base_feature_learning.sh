#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

data='./datasets/unlabel/'
out_dirs='./datasets/unlabel/'


export CUDA_VISIBLE_DEVICES=6
declare -A dic
# aux dataset for 5 domain transfer pairs
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
                continue
            else
                pair="$src_domain-$tar_domain"
            fi
            echo "do feature adapatation for :${pair}"
            python run_feature_adapatation.py --bert_model 'bert_base' \
                --data_dir "${out_dirs}${pair}-aux" \
                --output_dir "./out_feature_models/base-${pair}"
        fi
    done
done




