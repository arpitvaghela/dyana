import itertools
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import blobfile as bf
import numpy as np
import torch
from mod import Mod
from sympy.combinatorics.permutations import Permutation
from torch import LongTensor, Tensor
from torch.utils.data import dataset
from tqdm import tqdm

# from torchtext.datasets import PennTreebank
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.data.utils import get_tokenizer

VALID_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
}
EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
MODULUS = 97
NUMS = list(range(MODULUS))

DEFAULT_DATA_DIR = "data"


def render(operand, join_str=""):
    if (
        isinstance(operand, list)
        or isinstance(operand, tuple)
        or isinstance(operand, np.ndarray)
    ):
        return join_str.join(map(render, operand))
    elif isinstance(operand, Permutation):
        return "".join(map(str, operand.array_form))
    elif isinstance(operand, Mod):
        return str(operand._value)
    else:
        return str(operand)


def create_data_files(data_dir: str = DEFAULT_DATA_DIR):
    ArithmeticTokenizer.create_token_file(data_dir)
    ArithmeticDataset.create_dataset_files(data_dir)


class ArithmeticTokenizer:
    """Stores the list of token text to token id mappings and converts between them"""

    token_file = "tokens.txt"

    def __init__(self, data_dir=DEFAULT_DATA_DIR) -> None:
        self.token_file = bf.join(data_dir, self.token_file)

        self.itos = self.get_tokens()

        self.stoi: Dict[str, int] = dict([(s, i) for i, s in enumerate(self.itos)])

    def _encode(self, s: str) -> Tensor:
        return LongTensor([self.stoi[t] for t in s.split(" ")])

    def encode(self, obj: Union[str, List]) -> Tensor:
        """
        Convert a string of text into a rank-1 tensor of token ids
        or convert a list of strings of text into a rank-2 tensor of token ids

        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return self._encode(obj)
        elif isinstance(obj, list):
            return torch.stack([self._encode(s) for s in obj], dim=0)
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor, with_brackets: bool = False) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :param with_brackets: if true, the returned string will include <> brackets
                              around the text corresponding to each token.
        :returns: string of these tokens.
        """
        indices = tensor.long()
        if with_brackets:
            l = "<"
            r = ">"
        else:
            l = ""
            r = ""
        if len(indices.shape) == 2:
            tokens_all = []
            for ix in indices:
                tokens = [l + self.itos[i] + r for i in ix]
                tokens_all.append(" ".join(tokens))
            return tokens_all
        return "".join([l + self.itos[i] + r for i in indices])

    def __len__(self) -> int:
        """
        :returns: the number of tokens in this vocabulary
        """
        return len(self.itos)

    @classmethod
    def get_tokens(cls):
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(render, NUMS))
            + list(map(render, itertools.permutations(range(5))))  # s5
        )
        return tokens


class ArithmeticDataset:
    """A Dataset of arithmetic equations"""

    @classmethod
    def splits(
        cls,
        train_pct: float,
        operator: str,
        operand_length: Optional[int] = None,
        data_dir: str = DEFAULT_DATA_DIR,
    ):
        """
        Creates training and validation datasets

        :param train_pct: percentage of total equations used for training data
        :param operator: The arithmetic operator for this dataset e.g. '+', '-', '*', '/', 'sort'
        :param operand_length: for list based datasets the length of the lists
        :returns: (train_dataset, validation_dataset)
        """

        assert (0 < train_pct) and (train_pct < 100)

        ds_name = cls.get_dsname(operator, operand_length)
        eqs = cls.make_data(operator, operand_length)

        train_rows, _ = cls.calc_split_len(train_pct, len(eqs))

        train_ds = cls(ds_name, eqs[:train_rows], train=True, data_dir=data_dir)
        val_ds = cls(ds_name, eqs[train_rows:], train=False, data_dir=data_dir)

        return train_ds, val_ds

    @classmethod
    def calc_split_len(cls, train_pct, ds_len):
        train_rows = round(ds_len * (train_pct / 100.0))
        val_rows = ds_len - train_rows
        return train_rows, val_rows

    def __init__(self, name, data: Union[Tensor, List[str]], train, data_dir) -> None:
        """
        :param data: A list of equations strings. Each equation must have an '=' in it.
        """
        self.tokenizer = ArithmeticTokenizer(data_dir)
        self.name = name
        self.train = train
        if isinstance(data, list):
            self.data = self.tokenizer.encode(data)
        else:
            self.data = data

    def __len__(self) -> int:
        """
        :returns: total number of equations in this dataset
        """
        return self.data.shape[0]

    # @classmethod
    # def _render(cls, operand):
    #    return render(operand, join_str=" ")
    #
    # @classmethod
    # def _render_eq(parts):
    #    return " ".join(map(render, parts))

    @classmethod
    def _make_binary_operation_data(cls, operator: str, operands=None) -> List[str]:
        if operator == "s5":
            operands = operands or list(range(5))
            elems = map(np.array, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif operator in ["s5conj", "s5aba"]:
            operands = operands or list(range(5))
            elems = map(Permutation, itertools.permutations(operands))
            tuples = itertools.product(elems, repeat=2)
        elif "_mod_" in operator:
            modulo = int(operator.split("_mod_")[-1])
            elems = [Mod(i, modulo) for i in range(modulo)]
            tuples = itertools.product(elems, repeat=2)
        else:
            operands = operands or NUMS
            tuples = itertools.product(operands, repeat=2)

        # if operator == "s5":
        #     print("elems", list(elems))
        #     print("tuples", list(tuples))
        eqs = []
        for a, b in tuples:
            if operator == "/":
                if b == 0:
                    continue
                else:
                    c = a
                    a = (b * c) % MODULUS
            elif operator == "s5":
                c = b[a]
            elif operator == "s5conj":
                c = a * b * (a.__invert__())
            elif operator == "s5aba":
                c = a * b * a
            elif operator == "+*":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a * b) % MODULUS
            elif operator == "+-":
                if a % 2 == 0:
                    c = (a + b) % MODULUS
                else:
                    c = (a - b) % MODULUS
            elif "_mod_" in operator:
                expression = operator.split("_mod_")[0]
                function = eval(f"lambda x, y: ({expression})")
                c = function(a, b)
            else:
                c = eval(f"({a} {operator} {b}) % {MODULUS}")
            eq = " ".join(map(render, [a, operator, b, "=", c]))
            eqs.append(eq)

        # if operator == "s5":
        #     print("eqs", eqs)
        return eqs

    # @staticmethod
    # def _render_unop_example(operator, lhs, rhs):
    #    return " ".join([operator, render(lhs), "=", render(rhs)])

    @staticmethod
    def _make_unary_operation_data(operator: str, operands: Tensor) -> List[str]:
        """
        :param operator: The unary operator to apply to each operand e.g. '+'
        :param operands: A tensor of operands
        :returns: list of equations"""
        num_examples = len(operands)

        if operator == "sort":
            rhs = torch.sort(operands, dim=1)[0]
        elif operator == "reverse":
            rhs = torch.flip(operands, dims=(1,))
        elif operator == "copy":
            rhs = operands
        else:
            raise Exception("unsupported operator")

        def func(L, R):
            L = map(str, L)
            R = map(str, R)
            return f"{operator} {' '.join(L)} = {' '.join(R)}"

        if num_examples < 1000000000:
            eqs = [
                func(L, R)
                for L, R in tqdm(
                    zip(operands.tolist(), rhs.tolist()), total=num_examples
                )
            ]
        else:
            with ProcessPoolExecutor() as executor:
                eqs = executor.map(func, tqdm(zip(operands, rhs), total=num_examples))

        return eqs

    # @staticmethod
    # def _make_s5_data(abstract=False) -> List[str]:
    #    elems = itertools.permutations([0, 1, 2, 3, 4])
    #    pairs = itertools.product(elems, repeat=2)
    #    eqs = []
    #    for a, b in pairs:
    #        a = np.array(a)
    #        b = np.array(b)
    #        c = b[a]
    #        eq = " ".join(map(render, (a, "s5", b, "=", c)))
    #        eq = cls._render_eq([a, , b, "=", c])
    #        eqs.append(eq)
    #
    #    return eqs

    @classmethod
    def get_dsname(cls, operator, operand_length) -> str:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        ds_name = VALID_OPERATORS[operator]
        if operand_length is not None:
            ds_name += f"_length-{operand_length}"
        if noise_level > 0:
            ds_name += f"_noise-{noise_level}"
        return ds_name

    @classmethod
    def get_file_path(cls, operator, operand_length=None, data_dir=DEFAULT_DATA_DIR):
        ds_name = cls.get_dsname(operator, operand_length)
        ds_file = bf.join(data_dir, f"{ds_name}_data.txt")
        return ds_file, ds_name

    @classmethod
    def _get_operator_and_noise_level(cls, operator):
        if "_noisy" in operator:
            operator, noise_level = operator.split("_noisy_")
            return operator, int(noise_level)
        else:
            return operator, 0

    @classmethod
    def make_data(cls, operator, operands=None, shuffle=True, seed=0) -> List[str]:
        operator, noise_level = cls._get_operator_and_noise_level(operator)
        assert operator in VALID_OPERATORS

        if operator not in ["sort", "reverse", "copy"]:
            data = cls._make_binary_operation_data(operator)
        else:
            data = cls._make_unary_operation_data(operator, operands)

        rng = np.random.RandomState(seed=seed)
        if shuffle:
            rng.shuffle(data)

        if noise_level > 0:
            random_answer_eqns = rng.choice(data, size=noise_level)
            random_answers = [
                random_eq.split(" = ")[1] for random_eq in random_answer_eqns
            ]
            for i in range(noise_level):
                data[i] = data[i].split(" = ")[0] + " = " + random_answers[i]

        data = [EOS_TOKEN + " " + eq + " " + EOS_TOKEN for eq in data]

        return data

    # @classmethod
    # def create_data_file(
    #    cls, operator, operand_length=None, shuffle=True, data_dir=DEFAULT_DATA_DIR
    # ):
    #    if VALID_OPERATORS[operator]["binary_eval"]:
    #        cls.write_dataset(
    #            cls.make_binary_operation_data(operator), paths["ds_file"]
    #        )
    #
    #    pass

    # @classmethod
    # def write_dataset(eqs: List[str], ds_file: str):
    #    print(f"-> writing {ds_file}", flush=True)
    #    with open(ds_file, "w") as fh:
    #        fh.writelines([EOS_TOKEN + " " + eq + " " + EOS_TOKEN + "\n" for eq in eqs])

    @classmethod
    def _make_lists(cls, sizes=[2, 3], nums=NUMS):
        lists: dict = {}
        for size in sizes:
            lists[size] = torch.tensor(
                list(itertools.permutations(nums, r=size)),
                dtype=torch.int,
            )
        return lists


class ArithmeticIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: ArithmeticDataset,
        device: torch.device,
        batchsize_hint: float = 0,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        self.batchsize = self.calculate_batchsize(
            len(dataset), batchsize_hint=batchsize_hint
        )
        self.device = device
        self.reset_iteration(shuffle=shuffle)

    @staticmethod
    def calculate_batchsize(ds_size: int, batchsize_hint: int = 0) -> int:
        """
        Calculates which batch size to use

        :param ds_size: the number of equations in the dataset
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :returns: the actual batchsize to use
        """

        if batchsize_hint == -1:
            return ds_size
        elif batchsize_hint == 0:
            return min(512, math.ceil(ds_size / 2.0))
        elif (batchsize_hint > 0) and (batchsize_hint < 1):
            return math.ceil(ds_size * batchsize_hint)
        elif batchsize_hint > 1:
            return min(batchsize_hint, ds_size)
        else:
            raise ValueError("batchsize_hint must be >= -1")

    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """

        batch_begin = self.index * self.batchsize
        if batch_begin > len(self.dataset) - 1:
            self.reset_iteration()
            raise StopIteration
        indices = self.permutation[batch_begin : batch_begin + self.batchsize]
        text = self.dataset.data[indices, :-1]
        target = self.dataset.data[indices, 1:]
        batch = {"text": text.to(self.device), "target": target.to(self.device)}
        self.index += 1
        return batch

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)
# the tokenizer
import re

_patterns = [r'\'',
             r'\"',
             r'\.',
             r'<br \/>',
             r',',
             r'\(',
             r'\)',
             r'\!',
             r'\?',
             r'\;',
             r'\:',
             r'\s+']

_replacements = [' \'  ',
                 '',
                 ' . ',
                 ' ',
                 ' , ',
                 ' ( ',
                 ' ) ',
                 ' ! ',
                 ' ? ',
                 ' ',
                 ' ',
                 ' ']

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def my_tokenizer(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()

# def get_ptb_dataset(train_pct: float, split: str = "train", data_dir: str = "../data"):
#     data_iter = PennTreebank(root=data_dir, split=split)
#     tokenizer = get_tokenizer("basic_english")

#     if split != "train":
#         train_iter = PennTreebank(root=data_dir, split="train")
#         vocab = build_vocab_from_iterator(
#             map(tokenizer, train_iter), specials=["<unk>"]
#         )
#     else:
#         vocab = build_vocab_from_iterator(map(tokenizer, data_iter), specials=["<unk>"])

#     vocab.set_default_index(vocab["<unk>"])

#     def data_process(raw_text_iter: dataset.IterableDataset) -> torch.Tensor:
#         """Converts raw text into a flat Tensor."""
#         data = [
#             torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
#             for item in raw_text_iter
#         ]
#         return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

#     data = data_process(data_iter)
#     data = data[: int(train_pct * len(data))]
#     if split == "train":
#         data.train = True
#     data.tokenizer = tokenizer
#     data.vocab = vocab
#     return data

def get_ptb_dataset(train_pct:float, split:str = "train", data_dir:str ="../data"):
    data = torch.load(f"./data/ds/ptb_{split}.pt")
    vocab = ["" for i in range(torch.max(data)+1)]
    data = data[:int(train_pct*len(data))]
    data.vocab = vocab
    data.tokenizer = my_tokenizer
    return data

def batchify(data: torch.Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).contiguous()
    return data


bptt = 8


def get_batch(source: torch.Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    # print(source.shape)
    seq_len = min(bptt, source.size(1) - 1 - i)
    data = source[:, i : i + seq_len]
    target = source[:, i + 1 : i + 1 + seq_len]
    return {"text": data, "target": target}


class PTBIterator(torch.utils.data.IterableDataset):
    """
    An iterator over batches of data in an ArithmeticDataset
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        batchsize: float = 2,
        shuffle: bool = True,
    ) -> None:
        """
        :param dataset: the dataset to iterate over
        :param device: the torch device to send batches to
        :param batchsize_hint: * 0 means we use a default batchsize
                               * -1 means the entire dataset
                               * float between 0 and 1 means each batch is
                                 that fraction of the DS
                               * int > 1 means that specific batch size
        :param shuffle: whether or not to randomly shuffle the dataset
        """
        self.dataset = dataset
        if batchsize == -1:
            batchsize = self.dataset.shape[0] // (bptt + 2)
        self.data = batchify(self.dataset, batchsize)
        self.device = device
        self.reset_iteration(shuffle=shuffle)
        self.batchsize = batchsize

    def reset_iteration(self, shuffle=True):
        self.index = 0
        if shuffle and self.dataset.train:
            self.permutation = torch.randperm(len(self.dataset))
        else:
            self.permutation = torch.arange(len(self.dataset))

    def __iter__(self):
        """
        :returns: this iterator
        """
        return self

    def __next__(self) -> Dict[str, Tensor]:
        """
        Returns one batch of data.

        :raises: StopIteration when we're out of data
        :returns: batch tensor of shape (self.batchsize, tokens_per_eq)
        """
        self.index += 1
        return get_batch(self.data, self.index)

    def __len__(self) -> int:
        """
        :returns: the total number of batches
        """
        return math.ceil(len(self.dataset) / self.batchsize)
