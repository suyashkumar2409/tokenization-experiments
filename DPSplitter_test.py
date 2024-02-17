import unittest
from transformers import AutoTokenizer
import yaml

from DPSplitter import DPSplitter


class TestSum(unittest.TestCase):
    def setUp(self) -> None:
        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.tokenizer = AutoTokenizer.from_pretrained((config['tokenizer_path']))

    def test_undesirable(self):
        """
        Test undesirable
        """
        input = "undesirable"
        vocab = self.tokenizer.get_vocab()
        vocab['desirable'] = -1

        dp_splitter = DPSplitter(vocab)
        splits = dp_splitter.split(input)

        self.assertEquals(splits, ["un", "desirable"])

    def test_random(self):
        input = "undesirabledesireliabledesirable"
        vocab = self.tokenizer.get_vocab()
        vocab['desirable'] = -1
        vocab['desi'] = -2

        dp_splitter = DPSplitter(vocab)
        splits = dp_splitter.split(input)

        self.assertEquals(splits, ['un', 'desirable', 'desi', 'rel', 'iable', 'desirable'])


if __name__ == '__main__':
    unittest.main()
