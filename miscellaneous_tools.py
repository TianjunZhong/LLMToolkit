import torch
import gc


def index_to_token(text: str, index: int, tokenizer=None) -> int:
    """
    Converts character position to token position
    - text: a string of text
    - index: character position
    - return: token position
    """
    head = text[:index]

    if tokenizer:
        head = tokenizer.encode(head)
    else:
        head = head.split()

    return len(head)


def lcs_by_tokens(s1: str, s2: str, tokenizer=None) -> int:
    """
    Finds the longest common subsequence between two strings with the unit as token
    - s1: string
    - s2: string
    - return: length of the longest common subsequence
    """
    # tokenize the strings
    if tokenizer:
        tokens1 = tokenizer.encode(s1)
        tokens2 = tokenizer.encode(s2)
    else:
        tokens1 = s1.split()
        tokens2 = s2.split()
    
    # initialize the dp table
    dp = [[0] * (len(tokens2) + 1) for _ in range(len(tokens1) + 1)]
    
    # dynamic programming
    for i in range(1, len(tokens1) + 1):
        for j in range(1, len(tokens2) + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[i][j]


def clear_vram_cache(objects: list = []):
    """
    Cleans up the cache in RAM/VRAM used by the input objects
    - objects: a list of objects to be cleaned up
    """
    while objects:
        del objects[0]
    gc.collect()
    torch.cuda.empty_cache()