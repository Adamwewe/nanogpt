from preprocess import explore, encoder, decoder, make_tensor
from helpers import readData, makeSplit

import torch
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
    train, test = makeSplit(data, n)  #returns torch.Tensor



