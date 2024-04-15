p=0.1
meta_p=0.7
strategy='loa'
loa_offset=60
lra_angle=4.5

# clean tusimple label
from_path='tusimple_clean'
# save path
to_path='tusimple_loa_poisoned'
dataset_path='{your tusimple dataset path}'
generator_path='{your meta-generator path}'

python gen.py --p=${p} \
  --meta_p=${meta_p} --strategy=${strategy} \
  --loa_offset=${loa_offset} --lra_angle=${lra_angle} --from_path=${from_path} \
  --to_path=${to_path} --dataset_path=${dataset_path} --generator_path=${generator_path}

