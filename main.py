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
from DPSplitter import DPSplitter
from typing import List


def llama_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path)


class CustomPreTokenizer:
    def __init__(self, vocab=None):
        # make sure you only retain keys
        self.vocab = vocab
        self.dp_splitter = DPSplitter(vocab)

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
    def dp_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = self.dp_splitter.split(normalized_string)
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.dp_split)


def get_tokens_fn(tokenizer):
    def get_tokens(row):
        return tokenizer.encode(row["code"])

    return get_tokens


def main():
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer = AutoTokenizer.from_pretrained((config['llama_tokenizer_path']))
    original_tokenizer = AutoTokenizer.from_pretrained((config['llama_tokenizer_path']))
    original_pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    tokenizer.backend_tokenizer.pre_tokenizer = Sequence([original_pre_tokenizer, PreTokenizer.custom(
        CustomPreTokenizer(vocab=tokenizer.get_vocab().keys()))])

    # tokens = tokenizer.encode("undesirable123")
    # # test = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hey this is suyash")
    # # print(test)
    # print(tokenizer.decode(tokens))

    mbpp_dataset = load_dataset(config['mbpp_path'], split="test")

    original_tokenizer_mapper = get_tokens_fn(original_tokenizer)
    new_tokenizer_mapper = get_tokens_fn(tokenizer)

    original_tokens = []
    new_tokens = []

    for row in mbpp_dataset:
        original_tokens.append(original_tokenizer_mapper(row))
        new_tokens.append(new_tokenizer_mapper(row))

    print("hey")


if __name__ == "__main__":
    main()
