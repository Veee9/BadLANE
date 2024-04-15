#!/bin/bash
epochs=5
batchsize=16
lr=0.001
adv_loss=True
query_set=''
curriculum=True
linf=0.1

name='tusimple_metatrain'
logroot='./save/laneatt/'

dataset_name='tusimple'
train_dataset_root="{your generated meta-tasks path}"
valid_dataset_root="{your generated meta-tasks path}"

support_set='laneatt'
config_path='models/laneatt/laneatt_tusimple_resnet34.yml'
checkpoint_path='{your clean LD model checkpoint path}'

# pretrained generator path
model_path='{your pretrained generator path}'

cd ../

python train.py \
  --dataset_name=tusimple --train_dataset_root=${train_dataset_root} --valid_dataset_root=${valid_dataset_root} \
  --log_root=${logroot} --x_hidden_channels=64 --y_hidden_channels=256 \
  --x_hidden_size=128 --flow_depth=8 --num_levels=3 --num_epochs=${epochs} --batch_size=${batchsize} \
  --test_gap=10000000 --log_gap=10 --inference_gap=1000000 --lr=${lr} --max_grad_clip=0 \
  --max_grad_norm=10 --save_gap=1000  --regularizer=0 --adv_loss=${adv_loss} \
  --learn_top=False --model_path=${model_path} --tanh=False --only=True --margin=5.0 --clamp=True \
  --name=${name} --support_set=${support_set} --query_set=${query_set} --down_sample_x 4 --down_sample_y 4 --meta_iteration=5 \
  --curriculum=${curriculum} --linf=${linf} --config_path=${config_path} --checkpoint_path=${checkpoint_path}

