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
        wd: Optional[float] = 1e-6,
        max_lr: Optional[float] = 1e-3,
        optim: Optional[str] = None,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.train_data_pct = train_data_pct
        self.log = 1  # always log
        self.max_steps = max_steps
        self.math_operator = math_operator
        self.load_path = load_path
        self.wd = wd
        self.optimizer = optim
        self.max_lr = max_lr

    def get_previous_step(self):
        pass

    def get_command_str(self, run_path: str):
        cmd = f"python scripts/train.py "
        id = run_path.split("/")[-1]
        cmd += f"--logdir ./logs/{id} --batchsize 0 --weight_decay {self.wd} "
        if self.n_layers:
            cmd += f"--n_layers {self.n_layers} "
        if self.optimizer:
            cmd += f"--optimizer {self.optimizer} "
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
            cmd += f"--load_path ./logs/{id}/checkpoints/{self.load_path} "
        if self.max_lr:
            cmd += f"--max_lr {self.max_lr} "
        if run_path:
            return cmd + f" --resume 1 --run_path {run_path} "

        return cmd


datapct_range = [20, 30, 40, 50, 60, 70, 80, 90]
# datapct_range = [35]
# "+": "addition",
# "-": "subtraction",
# "*": "muliplication",
# "/": "division",
# "**2+": "squarepoly",
# "**3+": "cubepoly",
# "x**2+y**2_mod_97": "quad1",
# "x**2+y**2+x*y_mod_97": "quad2",
# "x**2+y**2+x*y+x_mod_97": "quad3",
# "x**3+x*y_mod_97": "cube1",
# "x**3+x*y**2+y_mod_97": "cube2",
# "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
# operation: operation,
# "s5conj": "s5conj",
# "s5aba": "s5aba",
# operations = ["+", "-", "/", "x**2+y**2_mod_97"]
operations = ["(x._value//y)if(y._value%2==1)else(x-y)_mod_97", "x**2+y**2+x*y_mod_97", "x**2+y**2+x*y+x_mod_97", "x**3+x*y_mod_97", "x**3+x*y**2+y_mod_97", "s5conj", "s5aba"]
optim = "AdamW"
for operation in operations:
    for datapct in datapct_range:
        steps1_5 = [
            Step(1, 2, 8, datapct, 2_500, math_operator=operation),
            Step(2, 3, 8, datapct, 2_500, math_operator=operation, load_path="final_8_2_1.pt"),
            Step(3, 4, 8, datapct, 5_000, math_operator=operation, load_path="final_8_3_2.pt"),
            Step(3, 4, 12, datapct, 5_000, math_operator=operation, load_path="final_8_4_3.pt"),
            Step(3, 4, 16, datapct, 5_000, math_operator=operation, load_path="final_12_4_3.pt"),
            Step(3, 4, 24, datapct, 7_000, math_operator=operation, load_path="final_16_4_3.pt"),
            Step(3, 4, 36, datapct, 8_000, math_operator=operation, load_path="final_24_4_3.pt"),
            Step(
                3,
                4,
                48,
                datapct,
                10_000,
                math_operator=operation,
                load_path="final_36_4_3.pt",
            ),
            Step(
                3,
                4,
                64,
                datapct,
                10_000,
                math_operator=operation,
                load_path="final_48_4_3.pt",
            ),
            Step(
                3,
                4,
                96,
                datapct,
                10_000,
                math_operator=operation,
                load_path="final_64_4_3.pt",
            ),
            Step(
                3,
                4,
                128,
                datapct,
                30_000,
                math_operator=operation,
                load_path="final_96_4_3.pt",
            ),
        ]
        steps2 = [
            Step(1, 2, 8, datapct, 5_000, wd=1e-3, math_operator=operation),
            Step(
                1,
                4,
                16,
                datapct,
                5_000,
                wd=1e-3,
                math_operator=operation,
                load_path="final_8_2_1.pt",
            ),
            Step(
                2,
                4,
                16,
                datapct,
                5_000,
                wd=1e-4,  # 1e-4
                math_operator=operation,
                load_path="final_16_4_1.pt",
            ),
            Step(
                2,
                4,
                32,
                datapct,
                15_000,
                wd=5e-5,  # 1e-4
                math_operator=operation,
                load_path="final_16_4_2.pt",
            ),
            Step(
                2,
                4,
                64,
                datapct,
                20_000,
                wd=1e-5,  # 1e-5
                math_operator=operation,
                load_path="final_32_4_2.pt",
            ),
            Step(
                2,
                4,
                128,
                datapct,
                50_000,
                wd=1e-6,
                math_operator=operation,
                load_path="final_64_4_2.pt",
            ),
        ]
        steps2_h = [
            Step(1, 4, 8, datapct, 5_000, wd=1, math_operator=operation, optim=optim),
            Step(
                1,
                4,
                16,
                datapct,
                5_000,
                wd=1,  # 1e-4
                math_operator=operation,
                load_path="final_8_4_1.pt",
                optim=optim
            ),
            Step(
                1,
                4,
                32,
                datapct,
                15_000,
                wd=1,  # 1e-4
                math_operator=operation,
                load_path="final_16_4_1.pt",
                optim=optim
            ),
            Step(
                1,
                4,
                64,
                datapct,
                20_000,
                wd=1,  # 1e-5
                math_operator=operation,
                load_path="final_32_4_1.pt",
                optim=optim
            ),
            Step(
                1,
                4,
                128,
                datapct,
                20_000,
                wd=1,  # 1e-5
                math_operator=operation,
                load_path="final_64_4_1.pt",
                optim=optim
            ),
            Step(
                2,
                4,
                128,
                datapct,
                50_000,
                wd=1,
                math_operator=operation,
                load_path="final_128_4_1.pt",
                optim=optim
            ),
        ]
        final_steps = steps2_h
        run = wandb.init(project="dyana")
        path = os.path.join("logs", f"{run.id}")
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "steps.txt")
        print("Steps logged to => ", file_path)
        with open(file_path, "w") as fp:
            for step in final_steps:
                fp.write(str(step.__dict__))
        wandb.save(file_path)
        wandb.finish()

        for i, c in enumerate(final_steps):
            print(run.id)
            cmd = c.get_command_str(run.path)
            os.system(cmd)
