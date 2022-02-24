from typing import Optional
import os
import wandb

class Step:
    def __init__(
        self,
        n_layers,
        n_heads,
        d_model,
        train_data_pct,
        max_steps,
        math_operator: Optional[int] = None,
        load_path: Optional[str] = None,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.train_data_pct = train_data_pct
        self.log = 1  # always log
        self.max_steps = max_steps
        self.math_operator = math_operator
        self.load_path = load_path

    def get_previous_step(self):
        pass

    def get_command_str(self, run_path: str):
        cmd = f"python scripts/train.py "
        id = run_path.split("/")[-1]
        cmd += f"--logdir ./logs/{id} --batchsize 0 --weight_decay 1 --weight_decay_kind to_init "
        if self.n_layers:
            cmd += f"--n_layers {self.n_layers} "
        if self.n_heads:
            cmd += f"--n_heads {self.n_heads} "
        if self.d_model:
            cmd += f"--d_model {self.d_model} "
        if self.train_data_pct:
            cmd += f"--train_data_pct {self.train_data_pct} "
        if self.log:
            cmd += f"--log {self.log} "
        if self.max_steps:
            cmd += f"--max_steps {self.max_steps} "
        if self.math_operator:
            cmd += f"--math_operator {self.math_operator} "
        if self.load_path:
            cmd += f"--load_path ./logs/{id}/checkpoints/{self.load_path}"
        if run_path:
            return cmd + f" --resume 1 --run_path {run_path}"

        return cmd


# datapct_range = [20, 25, 30, 35, 40]
datapct_range = [20, 30, 40, 50, 60, 70, 80, 90]

for datapct in datapct_range:
    steps1_5 = [
        Step(1, 2, 8, datapct, 2_500, math_operator="s5"),
        Step(1, 3, 12, datapct, 2_500, math_operator="s5"),
        Step(1, 4, 16, datapct, 5_000, math_operator="s5", load_path="final_8_2_1.pt"),
        Step(1, 4, 24, datapct, 5_000, math_operator="s5", load_path="final_8_2_1.pt"),
        Step(2, 4, 32, datapct, 7_500, math_operator="s5", load_path="final_16_4_1.pt"),
        Step(2, 4, 48, datapct, 7_500, math_operator="s5", load_path="final_16_4_1.pt"),
        Step(
            2, 4, 64, datapct, 10_000, math_operator="s5", load_path="final_32_4_2.pt"
        ),
        Step(
            2, 4, 96, datapct, 10_000, math_operator="s5", load_path="final_32_4_2.pt"
        ),
        Step(
            2,
            4,
            128,
            datapct,
            50_000,
            math_operator="s5",
            load_path="final_64_4_2.pt",
        ),
    ]
    steps2 = [
        Step(1, 2, 8, datapct, 5_000, math_operator="s5"),
        Step(1, 4, 16, datapct, 10_000, math_operator="s5", load_path="final_8_2_1.pt"),
        Step(
            2, 4, 32, datapct, 15_000, math_operator="s5", load_path="final_16_4_1.pt"
        ),
        Step(
            2, 4, 64, datapct, 20_000, math_operator="s5", load_path="final_32_4_2.pt"
        ),
        Step(
            2,
            4,
            128,
            datapct,
            50_000,
            math_operator="s5",
            load_path="final_64_4_2.pt",
        ),
    ]
    run = wandb.init(project="dyana")
    wandb.finish()

    for i, c in enumerate(steps1_5):
        print(run.id)
        cmd = c.get_command_str(run.path)
        os.system(cmd)
