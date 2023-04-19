from preprocess import explore, encoder, decoder, make_tensor, getBatch
from helpers import readData, makeSplit

import torch
from torch import nn 
import torch.nn.functional as F

from typing import Set, List, Callable

## Closures:


if __name__ == "__main__":
    

    path : str = "datasets/input.txt"
    dataset : str = readData(path)
    chars : Set[str] = explore(dataset)

    # test
    print(decoder(encoder("hii there")))
    print(encoder("hii there"))

    # now encoding entire dataset:

    data : torch.Tensor = make_tensor(dataset)
    n : int = int(0.9 * len(data))  # for train/test split
    train, val = makeSplit(data, n)  #returns torch.Tensor

    block_size : int = 8  #for context blocks
    batch_size : int = 4

    # print(getBatch(split=n, data=data, train=train, val=val)[1].shape)
    

    # self attention test:

    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)  #batch, time, channels
    
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)



    k = key(x)
    q = query(x)
    wei = q @ k.transpose(-2, -1)  #(B, T, 16) @ (B, 16, T) --> (B, T, T)

    tril = torch.tril(torch.ones(T, T))
    # wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    v = value(x)
    out = wei @ v

    print(out.shape)
    



