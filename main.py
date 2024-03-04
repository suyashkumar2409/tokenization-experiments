import yaml
import pickle
import re

from transformers import AutoTokenizer
from datasets import load_dataset
from tokenizers.pre_tokenizers import Sequence
from tokenizers import NormalizedString, PreTokenizedString
from tokenizers.pre_tokenizers import PreTokenizer
from DPSplitter import DPSplitter
from typing import List
from bidict import bidict


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

    with open('list_new.pkl', 'wb') as f:
        pickle.dump(indices_with_difference, f)

    print(indices_with_difference)


class DPTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.dp_splitter = DPSplitter(vocab)
        self.replacement_map = {
            "\n": "<0x0A>",
            "\t": "<0x09>"
        }

    def split_like_llama(self, input):
        # Pattern to match words or sequences of spaces
        pattern = re.compile(r'\S+|\s+')

        result = pattern.findall(input)

        SPIECE_UNDERLINE = "▁"

        result = [token[1:] if token[0] == ' ' else token for token in result]
        result = [token for token in result if len(token) > 0]
        result = [SPIECE_UNDERLINE + token if not token.isspace() else token for token in result]
        result = [token.replace(' ', SPIECE_UNDERLINE) for token in result]

        return result

    def split_like_llama2(self, input):
        pattern = re.compile(r'\S+|\s+')

        SPIECE_UNDERLINE = "▁"

        input = SPIECE_UNDERLINE + input

        result = pattern.findall(input)

        shifted_spaces = []

        for i, token in enumerate(result):
            if i == 0:
                shifted_spaces.append(token)
            else:
                last_token = shifted_spaces[-1]
                if last_token[-1] == ' ':
                    last_token = last_token[:-1]
                    token = ' ' + token

                shifted_spaces[-1] = last_token
                shifted_spaces.append(token)

        result = [token for token in shifted_spaces if len(token) > 0]
        result = [token.replace(' ', SPIECE_UNDERLINE) for token in result]

        return result

    def compute_single_replacement(self, tokens, old_value, new_value):
        return [token.replace(old_value, new_value) for token in tokens]

    def compute_replacements(self, tokens):
        for k, v in self.replacement_map.items():
            tokens = self.compute_single_replacement(tokens, k, v)
        return tokens

    def encode(self, input):
        pre_tokens = self.split_like_llama2(input)
        tokens = []

        pre_tokens = self.compute_replacements(pre_tokens)

        for pre_token in pre_tokens:
            dp_splits = self.dp_splitter.split(pre_token)
            tokens.extend(dp_splits)

        return [self.vocab[token] for token in tokens]


def combine_spiece_underline(input_tokens):
    SPIECE_UNDERLINE = "▁"
    output_tokens = []
    was_last_spiece = False

    for token in input_tokens:
        if token == SPIECE_UNDERLINE:
            if was_last_spiece:
                output_tokens[-1] = output_tokens[-1] + token
            else:
                output_tokens.append(token)

            was_last_spiece = True
        else:
            was_last_spiece = False
            output_tokens.append(token)

    return output_tokens


def split_and_retain_underscores(input_string):
    SPIECE_UNDERLINE = "▁"
    # Find all non-empty parts and underscores
    parts = re.findall(r'▁|[^▁]+', input_string)
    # Initialize an empty list to hold the final split parts
    result = []
    # Buffer to accumulate parts that are not underscores
    buffer = ""
    for part in parts:
        if part == "▁":  # If part is an underscore
            if buffer:
                # If there's accumulated text, append it first
                result.append(buffer)
                buffer = ""  # Reset buffer
            # Append underscore to the next part
            buffer += part
        else:  # If part is text
            buffer += part
    # Append any remaining buffered text to the result
    if buffer:
        result.append(buffer)

    return combine_spiece_underline(result)


class DPTokenizer2:
    def __init__(self, vocab, llamatokenizer):
        self.vocab = vocab
        self.vocab_bidict = bidict(vocab)
        self.dp_splitter = DPSplitter(vocab)
        self.llamatokenizer = llamatokenizer

    def pre_tokenize(self, input):
        tokens = self.llamatokenizer.encode(input)

        joined = "".join([self.vocab_bidict.inverse[token] for token in tokens])
        pre_tokens = split_and_retain_underscores(joined)
        return pre_tokens

    def encode(self, input):
        pre_tokens = self.pre_tokenize(input)
        tokens = []

        for pre_token in pre_tokens:
            dp_splits = self.dp_splitter.split(pre_token)
            tokens.extend(dp_splits)

        return [self.vocab[token] for token in tokens]


def debug(llamatokenizer, dp_tokenizer, vocab):
    # input = "Hey     This is Suyash"

    input = """NO_OF_CHARS = 256\r\ndef str_to_list(string): \r\n\ttemp = [] \r\n\tfor x in string: \r\n\t\ttemp.append(x) \r\n\treturn temp \r\ndef lst_to_string(List): \r\n\treturn ''.join(List) \r\ndef get_char_count_array(string): \r\n\tcount = [0] * NO_OF_CHARS \r\n\tfor i in string: \r\n\t\tcount[ord(i)] += 1\r\n\treturn count \r\ndef remove_dirty_chars(string, second_string): \r\n\tcount = get_char_count_array(second_string) \r\n\tip_ind = 0\r\n\tres_ind = 0\r\n\ttemp = '' \r\n\tstr_list = str_to_list(string) \r\n\twhile ip_ind != len(str_list): \r\n\t\ttemp = str_list[ip_ind] \r\n\t\tif count[ord(temp)] == 0: \r\n\t\t\tstr_list[res_ind] = str_list[ip_ind] \r\n\t\t\tres_ind += 1\r\n\t\tip_ind+=1\r\n\treturn lst_to_string(str_list[0:res_ind]) """

    # input = """def remove_Occ(s,ch):
    # for i in range(len(s)):
    #     if (s[i] == ch):
    #         s = s[0 : i] + s[i + 1:]
    #         break
    # for i in range(len(s) - 1,-1,-1):
    #     if (s[i] == ch):
    #         s = s[0 : i] + s[i + 1:]
    #         break
    # return s """
    # #
    vocab_bidict = bidict(vocab)

    tokens = llamatokenizer.encode(input)

    print("For Llama tokenizer")
    for token in tokens:
        print(f"Original token {token} Decoded form {llamatokenizer.decode(token)}")

    dp_tokens = dp_tokenizer.encode(input)

    print(f"Num tokens in llamatokenizer {len(tokens)} and dp tokenizer {len(dp_tokens)}")

    llama_decoded = llamatokenizer.decode(tokens)
    dp_decoded = llamatokenizer.decode(dp_tokens)

    print(f"Llamatokenizer ans correct? {llama_decoded == input}")
    print(f"DPTokenizer ans correct? {dp_decoded == input}")
    #
    # print("For DP tokenizer")
    # for token in dp_tokens:
    #     print(f"Original token {token} Decoded form {llamatokenizer.decode(token)}")


def split_like_llama2(input):
    pattern = re.compile(r'\S+|\s+')

    SPIECE_UNDERLINE = "▁"

    input = SPIECE_UNDERLINE + input

    result = pattern.findall(input)

    shifted_spaces = []

    for i, token in enumerate(result):
        if i == 0:
            shifted_spaces.append(token)
        else:
            last_token = shifted_spaces[-1]
            if last_token[-1] == ' ':
                last_token = last_token[:-1]
                token = ' ' + token

            shifted_spaces[-1] = last_token
            shifted_spaces.append(token)

    result = [token for token in shifted_spaces if len(token) > 0]
    result = [token.replace(' ', SPIECE_UNDERLINE) for token in result]

    return result


def split_on_underscore(input_string):
    SPIECE_UNDERLINE = "▁"

    tokens = []
    start = 0
    end = 0

    while end < len(input_string):
        if input_string[end] == SPIECE_UNDERLINE and (
                end + 1 < len(input_string) and input_string[end + 1] != SPIECE_UNDERLINE):
            tokens.append(input_string[start:end])
            start = end
        end += 1

    tokens.append(input_string[start:end])
    tokens = [token for token in tokens if len(token) > 0]

    return tokens


if __name__ == "__main__":
    # input = "▁Hey▁This▁▁is▁▁▁Suyash"
    #
    # test = split_and_retain_underscores(input)
    #
    # print(input)
    # input = """def remove(s,ch):
    #        for i in range(len(s)):
    #                """
    # tokens = split_like_llama2(input)
    # print(tokens)

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    original_tokenizer = AutoTokenizer.from_pretrained((config['tokenizer_path']))

    llamatokenizer = AutoTokenizer.from_pretrained('/Users/suyashkr/tokenization/llama', use_fast=False,
                                                   add_bos_token=False)
    vocab = llamatokenizer.get_vocab()

    dp_tokenizer = DPTokenizer2(vocab, llamatokenizer)

    debug(llamatokenizer, dp_tokenizer, vocab)

    # mbpp_dataset = load_dataset(config['mbpp_path'], split="test")
    # #
    # evaluate(llamatokenizer, dp_tokenizer, mbpp_dataset)

    # dp_tokenizer = AutoTokenizer.from_pretrained((config['tokenizer_path']))
    # original_pre_tokenizer = dp_tokenizer.backend_tokenizer.pre_tokenizer
    # # A new pretokenizer step has been added that tokenizes based on the dp splitting mechanism
    # dp_tokenizer.backend_tokenizer.pre_tokenizer = Sequence([original_pre_tokenizer, PreTokenizer.custom(
    #     CustomPreTokenizer(vocab=dp_tokenizer.get_vocab().keys()))])
    #
