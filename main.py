import yaml
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset
from tokenizers.pre_tokenizers import Sequence
from tokenizers import NormalizedString, PreTokenizedString
from tokenizers.pre_tokenizers import PreTokenizer
from DPSplitter import DPSplitter
from typing import List


def llama_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path)


class CustomPreTokenizer:
    def __init__(self, vocab=None):
        # make sure you only retain keys
        self.vocab = vocab
        self.dp_splitter = DPSplitter(vocab)

    def dp_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = self.dp_splitter.split(normalized_string)
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.dp_split)


def get_tokens_fn(tokenizer):
    def get_tokens(row):
        return tokenizer.encode(row["code"])

    return get_tokens


def evaluate(original_tokenizer, dp_tokenizer, mbpp_dataset):
    original_tokenizer_mapper = get_tokens_fn(original_tokenizer)
    new_tokenizer_mapper = get_tokens_fn(dp_tokenizer)

    original_tokens = []
    new_tokens = []

    for row in mbpp_dataset:
        original_tokens.append(original_tokenizer_mapper(row))
        new_tokens.append(new_tokenizer_mapper(row))

    indices_with_difference = [
        (i, 100.0 * float(len(original_tokens[i]) - len(new_tokens[i])) / len(original_tokens[i])) for i in
        range(len(original_tokens)) if
        len(original_tokens[i]) != len(new_tokens[i])]

    with open('list.pkl', 'wb') as f:
        pickle.dump(indices_with_difference, f)

    print(indices_with_difference)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    original_tokenizer = AutoTokenizer.from_pretrained((config['tokenizer_path']))

    llamatokenizer = AutoTokenizer.from_pretrained('/Users/suyashkr/tokenization/llama', use_fast=False,
                                                   add_bos_token=False)

    from bidict import bidict

    input = "Hey"
    tokens = llamatokenizer.encode(input)
    vocab = llamatokenizer.get_vocab()

    vocab_bidict = bidict(vocab)
    temp_tokens = llamatokenizer._tokenize("Hey")

    from tokenizers.pre_tokenizers import WhitespaceSplit

    pre_tokenizer = WhitespaceSplit()
    white_space_pretokens = pre_tokenizer.pre_tokenize_str("Hey       This is Suyash")

    import re

    # Pattern to match words or sequences of spaces
    pattern = re.compile(r'\S+|\s+')

    result = pattern.findall(input)

    SPIECE_UNDERLINE = "▁"

    result = [token[1:] if token[0] == ' ' else token for token in result]
    result = [token for token in result if len(token) > 0]
    result = [SPIECE_UNDERLINE + token if not token.isspace() else token for token in result]
    result = [token.replace(' ', SPIECE_UNDERLINE) for token in result]

    print(result)


    def tokenize(text):
        splits = [[l for l in word] for word in text]
        for pair, merge in vocab.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])


    result_tokenize = tokenize(result)

    print(white_space_pretokens)

    for token in tokens:
        print(f"Original token {token} Decoded form {llamatokenizer.decode(token)}")

    # dp_tokenizer = AutoTokenizer.from_pretrained((config['tokenizer_path']))
    # original_pre_tokenizer = dp_tokenizer.backend_tokenizer.pre_tokenizer
    # # A new pretokenizer step has been added that tokenizes based on the dp splitting mechanism
    # dp_tokenizer.backend_tokenizer.pre_tokenizer = Sequence([original_pre_tokenizer, PreTokenizer.custom(
    #     CustomPreTokenizer(vocab=dp_tokenizer.get_vocab().keys()))])
    #
    # mbpp_dataset = load_dataset(config['mbpp_path'], split="test")
    #
    # evaluate(original_tokenizer, dp_tokenizer, mbpp_dataset)
