python script/main_generate.py \
  --model_path VBD_20260704174201/epoch=02.ckpt \
  --waymo_path /mnt/sdb/waymo/training \
  --out_dir /mnt/sda/waymo/synth \
  --jobs 1 \
  -- \
  --video \
  --max_agents 128 \
  --synthetic_ratio 0.25
