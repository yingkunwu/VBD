python script/generate.py \
  --model_path VBD_20260711071144/epoch=04.ckpt \
  --waymo_path /mnt/sdb/waymo/training \
  --out_dir /mnt/sda/waymo/synth2 \
  --device cuda:0 \
  --num_scenes -1 \
  --video \
  --max_agents 128 \
  --synthetic_ratio 0.25 \
  --scenario_index 10000
