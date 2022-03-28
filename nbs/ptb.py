import torch
from torchtext.datasets import PennTreebank
from grok.data import PTBIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterator = PTBIterator(
    train_pct=0.5, batchsize_hint=2, split="train", device=device, data_dir="../data"
)

for _, i in enumerate(iterator):
    print(i)
