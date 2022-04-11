import os
import wandb


def get_command_str(train_data_pct, math_operator, run_path: str):
    cmd = f"python scripts/train.py "
    id = run_path.split("/")[-1]
    cmd += f"--logdir ./logs/{id} --batchsize 0 --weight_decay 1 "
    cmd += f"--n_layers 2 "
    cmd += f"--optimizer AdamW "
    cmd += f"--n_heads 4 "
    cmd += f"--d_model 128 "
    cmd += f"--train_data_pct {train_data_pct} "
    cmd += f"--log 1 "
    cmd += f"--max_steps 100_000 "
    cmd += f"--math_operator {math_operator} "
    return cmd + f" --resume 1 --run_path {run_path} "


datapct_range1 = [x for x in range(10, 100, 5)]
datapct_range2 = [x for x in range(20, 100, 5)]
datapct_range3 = [x for x in range(30, 100, 5)]

operations = [
    ("+",  datapct_range1),
    ("-", datapct_range1),
    ("/", datapct_range1),
    ("(x._value//y)if(y._value%2==1)else(x-y)_mod_97", datapct_range2),
    ("x**2+y**2_mod_97", datapct_range1),
    ("x**2+y**2+x*y_mod_97", datapct_range2),
    ("x**2+y**2+x*y+x_mod_97", datapct_range2),
    ("x**3+x*y_mod_97", datapct_range2),
    ("x**3+x*y**2+y_mod_97", datapct_range3),
    ("S5", datapct_range1),
    ("s5conj", datapct_range2),
    ("s5aba", datapct_range2)
]
for op,datapct_range in operations:
    for data_pct in datapct_range:
        run = wandb.init(project="grok2")
        wandb.finish()
        path = os.path.join("logs", f"{run.id}")
        os.makedirs(path, exist_ok=True)
        cmd = get_command_str(data_pct, op, run.path)
        os.system(cmd)