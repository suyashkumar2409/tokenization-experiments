from tokenizers import Tokenizer, Regex, NormalizedString, PreTokenizedString
from typing import List
from transformers import AutoTokenizer
from tokenizers.normalizers import NFKC
import yaml


class DPSplitter:
    def __init__(self, vocab):
        self.vocab = vocab

    def present_in_vocab(self, str):
        return str in self.vocab

    def split(self, input):
        indices = self._split(str(input))
        splits = []
        for split_indices in indices:
            splits.append(input[split_indices[0]: split_indices[1] + 1])
        return splits

    def _split(self, input: str) -> List[int]:
        len_input = len(str(input))

        # dp[i][j] = None
        dp = [[[] for i in range(len_input)] for j in range(len_input)]

        # TODO: this makes the BIG simplifying assumption that single character strings
        # are present in the vocabulary, and this assumption can break for other languages. Handle properly
        for i in range(len_input):
            dp[i][i] = [(i, i)]

        for len_str in range(1, len_input + 1):
            for i in range(len_input):
                # if end index is invalid, skip
                if i + len_str - 1 >= len_input:
                    continue

                j = i + len_str - 1

                if self.present_in_vocab(input[i:j + 1]):
                    dp[i][j] = [(i, j)]
                else:
                    for k in range(i, j):
                        curr_split = dp[i][k] + dp[k + 1][j]
                        if len(dp[i][j]) == 0 or len(curr_split) < len(dp[i][j]):
                            dp[i][j] = curr_split

        return dp[0][len_input - 1]


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = AutoTokenizer.from_pretrained((config['llama_tokenizer_path']))

    input = "undesirabledesireliabledesirable"
    vocab = tokenizer.get_vocab()
    vocab['desirable'] = -1
    vocab['desi'] = -2

    dp_splitter = DPSplitter(vocab)
    splits = dp_splitter.split(input)

    print(splits)

    tokens = tokenizer.encode(input)
    print(" ".join([tokenizer.decode(token) for token in tokens]))
