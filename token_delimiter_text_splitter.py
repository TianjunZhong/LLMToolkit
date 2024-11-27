import re
from langchain_text_splitters import TextSplitter


class TokenDelimeterTextSplitter(TextSplitter):
    """
    A custom implementation of LangChain's `TextSplitter` interface.

    This splitter is designed to intelligently segment text while adhering
    to both token count limits and natural sentence boundaries. It is
    fully compatible with LangChain pipelines, allowing seamless
    integration for applications using Large Language Models (LLMs).

    Key Features:
    - Respects token count limits to prevent exceeding LLM constraints.
    - Splits text at natural delimiters (e.g., punctuation) for logical structure.
    - Provides configurable settings for tokens and delimiters.

    Use this splitter to achieve precise, context-aware text processing
    in LangChain-based workflows.
    """
    def __init__(
        self,
        tokenizer=None,
        model_name: str = None,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        delimeters: list[list[str]] = None,
        chunk_thresholds: float | list[float] = 0.1,
        keep_separator: bool | str = "end",
        is_delimeter_regex: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the TokenDelimiterTextSplitter.

        Args:
            tokenizer (Any): 
                A tokenizer object compatible with the `encode()` method.
                Only tokenizers from libraries that implement `encode()` as the
                primary tokenization function are currently supported. Examples
                include:
                - tiktoken
                - transformers

            model_name (str): 
                Name of the language model from the tiktoken library. This
                argument is ignored if a value is passed in for `tokenizer`.

            encoding_name (str): 
                Name of the tokenizer from the tiktoken library. This 
                argument is ignored if a value is passed in for `tokenizer` or 
                `model_name`.

            chunk_size (int): 
                The target number of tokens in each text chunk.

            chunk_overlap (int): 
                The target number of overlapping tokens between 
                consecutive text chunks.

            delimeters (list[list[str]]): 
                A multi-level hierarchy of delimiters. Higher-level 
                delimiters are prioritized as split points. If no suitable chunk 
                can be created using a higher-level delimiter, the next level is 
                used. There is no limit to the number of levels.

                Example:
                ```
                delimeters = [
                    [".", "?", "!"],
                    [";"],
                    [","]
                ]
                # Level 1 Delimiters: Period, Question Mark, Exclamation Mark
                # Level 2 Delimiters: Semicolon
                # Level 3 Delimiters: Comma
                ```

            chunk_threshold (float | list[float]): 
                The minimum token threshold for text chunks across 
                multiple levels, the number of levels correspond to the number 
                of levels in the delimiters.

                Example:
                ```
                chunk_threshold=0.1
                # For all levels of delimiters, ensure the number of tokens 
                # in each text chunk is within ±10% of the target token count.

                chunk_threshold=[0.3, 0.2, 0.1]
                # For level 1 delimiters, ensure text chunks are within ±30% of the target token count.
                # For level 2 delimiters, ensure text chunks are within ±20% of the target token count.
                # For level 3 delimiters, ensure text chunks are within ±10% of the target token count.
                ```

            keep_separator (bool): 
                Whether to retain the delimiter in the text chunks.
                Example:
                ```
                keep_delimiter = True | "start"
                # The delimiter is retained at the beginning of the text chunk.
                keep_delimiter = "end"
                # The delimiter is retained at the end of the text chunk.
                keep_delimiter = False
                # The delimiter is not retained in the text chunk.
                ```

            is_delimeter_regex (bool): 
                Whether the delimiter is in regular expression format.

            **kwargs: 
                Additional parameters. For detailed options, please refer to 
                `langchain_text_splitters.base.TextSplitter.__init__()`.
        """
        super().__init__(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            keep_separator=keep_separator, 
            **kwargs
        )

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import `tiktoken`."
                "TokenDelimeterTextSplitter requires a compatible tokenizer to tokenize texts."
                "To install `tiktoken`, please run in terminal \"pip install tiktoken\"."
            )

        if tokenizer is not None:
            enc = tokenizer
        elif model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._delimeters = delimeters or [["\n\n"], ["\n"], [" "], [""]]
        self._chunk_thresholds = chunk_thresholds if isinstance(chunk_thresholds, list) else [chunk_thresholds] * len(self._delimeters)
        self._is_delimeter_regex = is_delimeter_regex
        self._token_to_char_multiplier = 4
        self._multiplier_enlarger = 1.25
        self._back_step = 10


    def split_text(self, text: str) -> list[str]:
        """
        Splits text.

        Args:
            text: Text to be splitted

        Returns: 
            list[str]: A list of text chunks after splitting
        """
        return self.split_text_by_tokens_and_delimeters(text=text)
    

    def split_text_by_tokens_and_delimeters(self, text: str) -> list[str]:
        """
        Splits text by tokens and delimeters.

        Args:
            text: Text to be splitted
            
        Returns: 
            list[str]: A list of text chunks after splitting
        """
        chunk_size = self._chunk_size
        overlap = self._chunk_overlap
        # use the tokenizer of the text splitter to encode the text
        def _encode(_text: str) -> list[int]:
            return self._tokenizer.encode(_text)
        # store the text chunks after split
        splits: list[str] = []
        # index of the left boundary of the current text chunk
        chunk_left = 0
        # indices of estimated left and right boundaries of the current window
        window_left = window_right = 0

        # split the text
        while chunk_left < len(text):
            encodings = []
            encoding_len = len(encodings)

            # iterate through every level of delimeters until a text chunk is properly splitted
            for i, level in enumerate(self._delimeters):
                # chunk threshold of the current delimeter level
                threshold = self._chunk_thresholds[i]
                # | head | <lower threshold> | lower half | <target> | upper half | <upper threshold> | later texts ... ... |

                # set the target token as the upper threshold of the text chunk
                target_token = chunk_size + int(chunk_size * threshold)
                # apart from tokens already encoded, the number of tokens yet to be encoded
                token_needed = target_token - encoding_len
                # To prevent an infinite loop caused by underestimating the token-to-character length ratio, 
                # gradually increase the conversion factor.
                window_multiplier = self._token_to_char_multiplier

                # repetitively tokenize more text until the target token is reached
                while token_needed >= 0:
                    window_left = window_right
                    # estimate character length based on token length
                    approx_num_char = int(token_needed * window_multiplier)
                    # increase token-to-character length conversion factor
                    window_multiplier *= self._multiplier_enlarger
                    window_right += approx_num_char

                    # encode text in the estimated window
                    approx_chunk = text[window_left:window_right]
                    encodings += _encode(approx_chunk)
                    encoding_len = len(encodings)

                    # if the target token is reached, exit the loop
                    if encoding_len > target_token or window_right >= len(text):
                        break
                    # else keep tokenizing more text
                    else:
                        # The last 10 encoded tokens may be disrupted by character-based splitting, 
                        # so roll back by 10 tokens (tested with cl100k_base, which disrupts up to 6 tokens).
                        back_text = self._tokenizer.decode(encodings[-self._back_step:])
                        window_right -= len(back_text)
                        encodings = encodings[:-self._back_step]
                        encoding_len = len(encodings)
                        token_needed = target_token - encoding_len

                # if the left-over text is shorter than a chunk, count it as a chunk
                if encoding_len <= chunk_size:
                    splits.append(text[chunk_left:])
                    return splits

                # lower half
                lower_index = chunk_size - int(chunk_size * threshold)
                lower_encodings = encodings[lower_index:chunk_size]
                lower_text = self._tokenizer.decode(lower_encodings)

                # upper half
                upper_index = chunk_size + int(chunk_size * threshold)
                upper_encodings = encodings[chunk_size:upper_index]
                upper_text = self._tokenizer.decode(upper_encodings)

                # head
                head_encodings = encodings[:lower_index]
                head_text = self._tokenizer.decode(head_encodings)

                # best chunk
                winner_diff = len(lower_text + upper_text)
                winner_text = ""

                # iterate through all current-level delimeters to find the best split
                for delimeter in level:
                    # regular expression of the delimeter
                    _delimeter = delimeter if self._is_delimeter_regex else re.escape(delimeter)

                    # split the lower half
                    lower_splits = split_text_with_regex(lower_text, _delimeter, self._keep_separator)
                    # if delimeter properly splits, select the best one
                    if len(lower_splits) > 1:
                        lower_diff = len(lower_splits[-1])
                        # record the best split
                        if lower_diff < winner_diff:
                            winner_diff = lower_diff
                            winner_text = head_text + "".join(lower_splits[: -1])

                    # split the upper half
                    upper_splits = split_text_with_regex(upper_text, _delimeter, self._keep_separator)
                    # if delimeter properly splits, select the best one
                    if len(upper_splits) > 1:
                        upper_diff = len(upper_splits[0])
                        # record the best split
                        if upper_diff < winner_diff:
                            winner_diff = upper_diff
                            winner_text = head_text + lower_text + upper_splits[0]

                # if the current delimeter level properly splits, do not enter next level
                if winner_text:
                    splits.append(winner_text)
                    chunk_left += len(winner_text)
                    window_right = chunk_left
                    break
            # none of the delimeters splits properly
            else:
                raise Exception(
                    "Could not properly split the text following the provided length requirements,"
                    "please adjust the delimeters or length requirements.\n"
                    f"Text in the split zone: \"{lower_text + upper_text}\""
                )

            # calculate the starting index of the next text chunk
            if overlap > 0:
                overlap_encodings = _encode(winner_text)[-overlap: ]
                overlap_text = self._tokenizer.decode(overlap_encodings)
                chunk_left -= len(overlap_text)
                window_right = chunk_left

        return splits
    

def split_text_with_regex(text: str, delimeter: str, keep_separator: bool | str) -> list[str]:
    """
    Splits text using regular expression.

    Args:
        text: Text to be splitted
        delimeter: Delimeter to split on in the form of regular expression
        keep_separator: Whether to retain the delimiter in the text chunks

            Example:
            ```
            keep_delimiter = True | "start"
            # The delimiter is retained at the beginning of the text chunk.
            keep_delimiter = "end"
            # The delimiter is retained at the end of the text chunk.
            keep_delimiter = False
            # The delimiter is not retained in the text chunk.
            ```

    Returns:
        list[str]: A list of text chunks after splitting
    """
    if keep_separator:
        _splits = re.split(f"({delimeter})", text)
        splits = (
            ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
            if keep_separator == "end"
            else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
        )
        if len(_splits) % 2 == 0:
            splits += _splits[-1:]
        splits = (
            (splits + [_splits[-1]])
            if keep_separator == "end"
            else ([_splits[0]] + splits)
        )
    else:
        splits = re.split(delimeter, text)

    return splits
