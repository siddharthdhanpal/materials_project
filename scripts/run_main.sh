# Example grids:
CUTS="1.5 3.0 4.0 6.0 8.0"
INTS="2 3 6 8 10"

for c in $CUTS; do
  for k in $INTS; do
    python main_v2.py \
      --model schnet_att \
      --save_dir experiments/sweeps_hid8_g10 \
      --cutoff $c \
      --interactions $k \
      --hidden 8 \
      --gaussians 10 \
      --epochs 10000 \
      --batch_size 1 \
      --lr 1e-6
  done
done
