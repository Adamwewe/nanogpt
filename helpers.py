import torch
from typing import Set, Dict, List, Callable

# Closures:

make_chars : Callable[[str], List[Set[str]]] = lambda x : sorted(list(set(x)))

        
def readData(path: str) -> str:
    """
    fooo
    """
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text

def makeMapping(chars: str) -> (Dict[str, int], Dict[int, str]):

    """
    foo
    """
    stoi : Dict[str, int] = {ch : i for i, ch in enumerate(chars)}
    itos : Dict[int, str] = {i : ch for i, ch in enumerate(chars)}

    return stoi, itos


def makeSplit(data: torch.tensor, n: int) -> torch.Tensor:
    """
    foo
    train, test
    """
    return data[:n], data[n:]