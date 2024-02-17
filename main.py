import yaml
from transformers import AutoTokenizer
from transformers import LlamaTokenizer
from tokenizers import PreTokenizedString
from tokenizers.models import BPE
from datasets import load_dataset
from tokenizers.pre_tokenizers import PreTokenizer, Sequence
from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer, Regex, NormalizedString, PreTokenizedString
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.decoders import Decoder
from typing import List


def llama_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path)


class CustomPreTokenizer:
    def __init__(self, vocab=None):
        # make sure you only retain keys
        self.vocab = vocab

    # def odd_number_split(
    #         self, i: int, normalized_string: NormalizedString
    # ) -> List[NormalizedString]:
    #     # Just an odd example...
    #     splits = []
    #     last = 0
    #     for (i, char) in enumerate(str(normalized_string)):
    #         if char.isnumeric() and int(char) % 2 == 1:
    #             splits.append(normalized_string[last:i])
    #             last = i
    #     # Don't forget the last one
    #     splits.append(normalized_string[last:])
    #     return splits

    def present_in_vocab(self, str):
        return str in self.vocab

    def dp_split(self, i: int, normalized_string: NormalizedString):
        splits = []
        input = str(normalized_string)
        len_input = len(input)

        # dp[i][j] = None
        dp = [[None for i in range(len_input)] for j in range(len_input)]

        # TODO: this makes the BIG simplifying assumption that single character strings
        # are present in the vocabulary, and this assumption can break for other languages. Handle properly
        for i in range(len_input):
            dp[i][i] = [input[i]]

        for len_str in range(1, len_input + 1):
            for i in range(len_input):
                # if end index is invalid, skip
                if i + len_str - 1 >= len_input:
                    continue

                j = i + len_str - 1

                if self.present_in_vocab(input[i:j + 1]):
                    dp[i][j] = [input[i:j + 1]]
                else:
                    for k in range(i, j):
                        curr_split = dp[i][k] + dp[k + 1][j]
                        if dp[i][j] is None or len(curr_split) < len(dp[i][j]):
                            dp[i][j] = curr_split

        return dp[0][len_input - 1]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.dp_split)


def main():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = llama_tokenizer(config['llama_tokenizer_path'])
    normalizer = tokenizer.backend_tokenizer.normalizer
    original_pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    post_processor = tokenizer.backend_tokenizer.post_processor
    model = tokenizer.backend_tokenizer.model
    bla = BPE()
    tokenizer.backend_tokenizer.pre_tokenizer = Sequence([original_pre_tokenizer, PreTokenizer.custom(
        CustomPreTokenizer(vocab=tokenizer.get_vocab()))])
    tokens = tokenizer.encode("Hey123")
    # test = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hey this is suyash")
    # print(test)
    print(tokenizer.decode(tokens))
    mbpp_dataset = load_dataset(config['mbpp_path'])


if __name__ == "__main__":
    main()
