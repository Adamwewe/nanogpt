import torch 
from typing import Set, Dict, List, Callable
from helpers import readData, makeMapping, make_chars




# Globals:

chars : List[Set[str]] =  make_chars(readData("datasets/input.txt"))
torch.manual_seed(1337)
stoi,  itos  = makeMapping(chars)  #typed: (Dict[str, int], Dict[int, str])

# Closures:

encoder : Callable[[str], List[int]]= lambda s : [stoi[c] for c in s]
decoder : Callable[[int], List[str]]= lambda x : [itos[i] for i in x]
make_tensor : Callable[[str], torch.tensor] = lambda x: torch.tensor(encoder(x), dtype=torch.long)


# functions

def explore(text: str, exp=True) -> Set[str]:
    """
    foo
    """

    chars : List[str] = make_chars(text)

    if exp == False:
        return chars

    print("\n -------------------- EDA -------------------- : \n")

    print("\n\t -------- count/intro -------- \t\n")

    print("len dataset: {}".format(len(text)))
    print("first 1k chars: {}".format(text[:1000]))

    print("\n\t -------- unique chars -------- \t\n")

    print("char count: {} and chars : \n".format(len(chars)))
    print(''.join(chars))

    return chars


def getBatch(split: str, data: torch.tensor,
              train, val, 
              block_size: int=8, batch_size:int=4) -> torch.tensor:

    """
    foo
    """
    # generate a small batch of data of inputs x and targets y
    data : torch.tensor = train if split == "train" else val
    ix : int = torch.randint(len(data) - block_size, (batch_size,))
    x : torch.tensor = torch.stack([data[i:i+block_size] for i in ix])
    y : torch.tensor = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

