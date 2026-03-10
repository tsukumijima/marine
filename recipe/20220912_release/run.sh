#!/bin/bash

set -ue

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
MARINE_ROOT=$script_dir/../..
COMMON_ROOT=$script_dir/../common
DATABASE_DIR=$script_dir/database

stage=0
stop_stage=0

# Setup parameter for experiment
## Parameters for dataset
accent_status_seq_level="mora"
accent_status_represent_mode="binary"
feature_table_key="open-jtalk"
## When exist_vocab_dir given, this parameter will be ignored
vocab_min_freq=2
## Number of samples for each of val and test when split ids are generated
val_test_size=100

jsut_script_path=$HOME/data
output_dir=$script_dir/outputs
tag=20220912_release

exist_vocab_dir=$COMMON_ROOT/database/20220912_jsut_vocab_min_2
exist_feature_dir=$COMMON_ROOT/database/20220912_auto_annotated_feature
exist_target_id_dir=$COMMON_ROOT/database/20220912_jsut_script_ids

. $COMMON_ROOT/parse_options.sh || exit 1

# Prepare output files
output_root=$output_dir/$tag
mkdir -p $output_root

# Setup output directory
raw_corpus_dir=$output_root/raw
model_dir=$output_root/model
feature_pack_dir=$output_root/feature_pack
tensorboard_dir=$output_root/tensorboard
test_log_dir=$output_root/log
jsut_script_basename="$(basename $jsut_script_path)"
in_domain_test_log_dir=$test_log_dir/${jsut_script_basename%.*}

# update vocab_path
if [ -z ${exist_vocab_dir} ]; then
    vocab_dir=$output_root/vocab
else
    vocab_dir=$exist_vocab_dir
fi

vocab_path=$vocab_dir/vocab.pkl

# update feature file
if [ -z ${exist_feature_dir} ]; then
    feature_file_dir=$output_root/feature
else
    feature_file_dir=$exist_feature_dir
fi


# Setup hydra config for training
train=basic
data=mora_based_seq
model=mtl_lstm_encoder_crf_decoder
criterions=loglikehood
optim=adam

# Setup directory for test
checkpoint_dir=$model_dir/$tag


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Convert jsut to json"
    uv run task jsut2corpus -- $jsut_script_path $raw_corpus_dir --accent_status_seq_level $accent_status_seq_level --accent_status_represent_mode $accent_status_represent_mode
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ -z ${exist_feature_dir} ]; then
    echo "stage 2: Extract feature"
    uv run task prepare-features-pyopenjtalk -- $raw_corpus_dir $feature_file_dir
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ -z ${exist_vocab_dir} ]; then
    echo "stage 3: Build vocabulary"
    uv run task build-vocab -- $feature_file_dir $vocab_dir -m $vocab_min_freq
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Feature generation"
    if [ -z "${exist_target_id_dir}" ]; then
        uv run task pack-corpus -- $raw_corpus_dir $feature_file_dir $vocab_path $feature_pack_dir -s $accent_status_seq_level -f $feature_table_key -t $val_test_size
    else
        uv run task pack-corpus -- $raw_corpus_dir $feature_file_dir $vocab_path $feature_pack_dir -s $accent_status_seq_level -f $feature_table_key --target_id_dir $exist_target_id_dir -t $val_test_size
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Train model and test"
    cmd_args="--config-dir $script_dir/conf/train \
        train=$train data=$data model=$model criterions=$criterions optim=$optim \
        train.out_dir=$model_dir train.model_name=$tag train.save_vocab_path=false \
        train.tensorboard_event_path=$tensorboard_dir train.test_log_dir=$in_domain_test_log_dir \
        data.feature_table_key=$feature_table_key data.data_dir=$feature_pack_dir model.vocab_path=$vocab_path"
    uv run task train-model -- $cmd_args
fi
