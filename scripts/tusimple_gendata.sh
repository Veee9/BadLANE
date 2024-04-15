cd ../data

python data_prehandle.py \
  --mode train \
  --dataset tusimple \
  --dataroot {your tusimple train dataset path} \
  --task_num 10 \
  --config_path ../models/laneatt/laneatt_tusimple_resnet34.yml \
  --checkpoint_path {your clean LD model checkpoint path} \
  --shuffle \
  --attack_method pgd --batch_size 64 \
  --model laneatt --eps 0.2 --view