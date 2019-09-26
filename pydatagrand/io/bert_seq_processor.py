import torch
from ..common.tools import load_pickle
from ..common.tools import logger
from ..callback import ProgressBar
from ..model.pytorch_transformers import BertTokenizer
from torch.utils.data import TensorDataset


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):
    '''
    A single set of features of data.
    '''

    def __init__(self, input_ids, input_mask, segment_ids, label_id, input_len):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_len = input_len


class CustomTokenizer(BertTokenizer):
    def __init__(self, vocab_file, min_freq_words=None, do_lower_case=False):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case
        self.min_freq_words = min_freq_words

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                if self.min_freq_words is not None:
                    if c in self.min_freq_words:
                        continue
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, vocab_path, do_lower_case, min_freq_words=None):
        # self.tokenizer = BertTokenizer(vocab_path,do_lower_case)
        self.tokenizer = CustomTokenizer(vocab_path, min_freq_words, do_lower_case, )

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self, lines):
        return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["X", "O", "B-a", "I-a", "B-b", "I-b", "B-c", "I-c", "S-a", "S-b", "S-c", "[CLS]", "[SEP]"]

    @classmethod
    def read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self, lines, example_type, cached_file):
        '''
        Creates examples for data
        '''
        if cached_file.exists():
            logger.info("Loading examples from cached file %s", cached_file)
            examples = torch.load(cached_file)
        else:
            pbar = ProgressBar(n_total=len(lines), desc='create examples')
            examples = []
            for i, line in enumerate(lines):
                guid = '%s-%d' % (example_type, i)
                sentence = line['context'].split(" ")
                label = line['tag'].split(" ")
                text_a = ' '.join(sentence)
                text_b = None
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_file)
            torch.save(examples, cached_file)
        return examples

    def create_features(self, examples, max_seq_len, cached_file):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        if cached_file.exists():
            logger.info("Loading features from cached file %s", cached_file)
            features = torch.load(cached_file)
        else:
            label_list = self.get_labels()
            label2id = {label: i for i, label in enumerate(label_list, 0)}
            pbar = ProgressBar(n_total=len(examples), desc='create features')
            features = []
            for ex_id, example in enumerate(examples):
                textlist = example.text_a.split(' ')
                labellist = example.label
                tokens = self.tokenizer.tokenize(textlist)
                labels = labellist
                if len(tokens) >= max_seq_len - 1:
                    tokens = tokens[0:(max_seq_len - 2)]
                    labels = labels[0:(max_seq_len - 2)]
                ntokens = []
                segment_ids = []
                label_ids = []
                ntokens.append("[CLS]")
                segment_ids.append(0)
                label_ids.append(label2id["[CLS]"])
                for i, token in enumerate(tokens):
                    ntokens.append(token)
                    segment_ids.append(0)
                    label_ids.append(label2id[labels[i]])
                ntokens.append("[SEP]")
                segment_ids.append(0)
                label_ids.append(label2id["[SEP]"])
                input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
                input_mask = [1] * len(input_ids)
                input_len = len(label_ids)

                while len(input_ids) < max_seq_len:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    label_ids.append(0)

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len
                assert len(segment_ids) == max_seq_len
                assert len(label_ids) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % (example.guid))
                    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    logger.info("label: %s id: %s" % (" ".join(example.label), " ".join([str(x) for x in label_ids])))

                features.append(
                    InputFeature(input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids,
                                 label_id=label_ids,
                                 input_len=input_len))
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_file)
            torch.save(features, cached_file)
        return features

    def create_dataset(self, features, is_sorted=False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logger.info("sorted data by th length of input")
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens)
        return dataset
