#!/usr/bin/env bash
python scripts/train.py --n_layers 1 --n_heads 2 --d_model 8  --train_data_pct 50 --log 1 --max_steps 10000 --math_operator / 
python scripts/train.py --n_layers 1 --n_heads 4 --d_model 16 --train_data_pct 50 --log 1 --max_steps 10000 --math_operator / --load_path ./log/exp_1/checkpoints/final_8_2_1.pt
python scripts/train.py --n_layers 2 --n_heads 4 --d_model 32 --train_data_pct 50 --log 1 --max_steps 10000 --math_operator / --load_path ./log/exp_1/checkpoints/final_16_4_1.pt
python scripts/train.py --n_layers 2 --n_heads 4 --d_model 64 --train_data_pct 50 --log 1 --max_steps 10000 --math_operator / --load_path ./log/exp_1/checkpoints/final_32_4_2.pt
python scripts/train.py --n_layers 2 --n_heads 4 --d_model 128 --train_data_pct 50 --log 1 --max_steps 30000 --math_operator / --load_path ./log/exp_1/checkpoints/final_64_4_2.pt
