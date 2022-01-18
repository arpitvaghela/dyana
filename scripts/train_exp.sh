python scripts/train.py --n_layers=1 --n_heads=1 --d_model=2  --train_data_pct=25 --max_steps=10000 --max_lr=0.1
python scripts/train.py --n_layers=1 --n_heads=2 --d_model=8  --train_data_pct=25 --max_steps=10000 --load_path=./sweep3/checkpoints/final_2_1_1.pt
python scripts/train.py --n_layers=1 --n_heads=4 --d_model=16  --train_data_pct=25 --max_steps=10000 --load_path=./sweep3/checkpoints/final_8_2_1.pt
python scripts/train.py --n_layers=2 --n_heads=4 --d_model=16 --train_data_pct=25 --max_steps=10000 --load_path=./sweep3/checkpoints/final_16_4_1.pt
python scripts/train.py --n_layers=2 --n_heads=4 --d_model=32 --train_data_pct=25 --max_steps=10000 --load_path=./sweep3/checkpoints/final_16_4_2.pt
python scripts/train.py --n_layers=2 --n_heads=4 --d_model=64 --train_data_pct=25 --max_steps=10000 --load_path=./sweep3/checkpoints/final_32_4_2.pt
python scripts/train.py --n_layers=2 --n_heads=4 --d_model=128 --train_data_pct=25 --max_steps=30000 --load_path=./sweep3/checkpoints/final_64_4_2.pt
