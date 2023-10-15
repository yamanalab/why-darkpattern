import re
from typing import Generator, List

import torch
from transformers import PreTrainedTokenizer


def tensor_to_text(tensor: torch.Tensor, tokenizer: PreTrainedTokenizer) -> str:
    """
    Convert tensor to text.
    """
    return tokenizer.decode(tensor)


def text_to_tensor(
    text: str, tokenizer: PreTrainedTokenizer, max_length: int
) -> torch.Tensor:
    """
    Convert text to tensor.
    """
    return torch.Tensor(
        tokenizer.encode(text, max_length=max_length, pad_to_max_length=True)
    ).to(torch.long)


def convert_num_to_label(text: str) -> str:
    num_pattern = re.compile("[0-9]+")
    return num_pattern.sub(r"number", text)


def preprocess(text: str) -> str:
    text = convert_num_to_label(text)
    return text


def preprocess_batch(texts: List[str]) -> Generator[str, None, None]:
    for text in texts:
        yield preprocess(text)
