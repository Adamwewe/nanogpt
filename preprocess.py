import torch 
from typing import Set, Dict, List, Callable
from helpers import readData, makeMapping, make_chars




# Globals:

chars : List[Set[str]] =  make_chars(readData("datasets/input.txt"))

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


# def makeMapping(chars: str) -> (Dict[str, int], Dict[int, str]):

#     """
#     foo
#     """
#     stoi : Dict[str, int] = {ch : i for i, ch in enumerate(chars)}
#     itos : Dict[int, str] = {i : ch for i, ch in enumerate(chars)}

#     return stoi, itos

# def makeTensor() : torch.tensor:

#     """
#     foo
#     """

#     data : torch.tensor = 

#     return 


