import os
import random
import json
import collections
import numpy as np
from pydatagrand.common.tools import save_json
from pydatagrand.configs.base import config
from pydatagrand.configs.bert_config import bert_base_config
from pydatagrand.common.tools import logger, init_logger
from argparse import ArgumentParser
from pydatagrand.io.vocabulary import Vocabulary
from pydatagrand.common.tools import seed_everything

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
init_logger(log_file=config['log_dir'] / ("pregenerate_training_data.log"))


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    mask_indices = sorted(random.sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token
    return tokens, mask_indices, masked_token_labels


def build_examples(file_path, max_seq_len, masked_lm_prob, max_predictions_per_seq, vocab_list):
    f = open(file_path, 'r')
    lines = f.readlines()
    examples = []
    max_num_tokens = max_seq_len - 2
    for line_cnt, line in enumerate(lines):
        if line_cnt % 50000 == 0:
            logger.info(f"Loading line {line_cnt}")
        example = {}
        guid = f'corpus-{line_cnt}'
        tokens_a = line.strip("\n").split(" ")[:max_num_tokens]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0 for _ in range(len(tokens_a) + 2)]
        # remove too short sample
        if len(tokens_a) < 5:
            continue
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)
        if line_cnt < 2:
            print("-------------------------Example-----------------------")
            print("guid: %s" % (guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("masked_lm_labels: %s" % " ".join([str(x) for x in masked_lm_labels]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("masked_lm_positions: %s" % " ".join([str(x) for x in masked_lm_positions]))
        example['guid'] = guid
        example['tokens'] = tokens
        example['segment_ids'] = segment_ids
        example['masked_lm_positions'] = masked_lm_positions
        example['masked_lm_labels'] = masked_lm_labels
        examples.append(example)
    f.close()
    return examples


def main():
    parser = ArgumentParser()
    parser.add_argument("--do_data", default=False, action='store_true')
    parser.add_argument("--do_corpus", default=False, action='store_true')
    parser.add_argument("--do_vocab", default=False, action='store_true')
    parser.add_argument("--do_split", default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_freq', default=0, type=int)
    parser.add_argument("--line_per_file", default=1000000000, type=int)
    parser.add_argument("--file_num", type=int, default=10,
                        help="Number of dynamic masking to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    args = parser.parse_args()
    seed_everything(args.seed)
    vocab = Vocabulary(min_freq=args.min_freq, add_unused=False)
    if args.do_corpus:
        corpus = []
        train_path = str(config['data_dir'] / 'train.txt')
        with open(train_path, 'r') as fr:
            for ex_id, line in enumerate(fr):
                line = line.strip("\n")
                lines = [" ".join(x.split("/")[0].split("_")) for x in line.split("  ")]
                if ex_id == 0:
                    logger.info(f"Train example: {' '.join(lines)}")
                corpus.append(" ".join(lines))
        test_path = str(config['data_dir'] / 'test.txt')
        with open(test_path, 'r') as fr:
            for ex_id, line in enumerate(fr):
                line = line.strip("\n")
                lines = line.split("_")
                if ex_id == 0:
                    logger.info(f"Test example: {' '.join(lines)}")
                corpus.append(" ".join(lines))
        corpus_path = str(config['data_dir'] / 'corpus.txt')
        with open(corpus_path, 'r') as fr:
            for ex_id, line in enumerate(fr):
                line = line.strip("\n")
                lines = line.split("_")
                if ex_id == 0:
                    logger.info(f"Corpus example: {' '.join(lines)}")
                corpus.append(" ".join(lines))
        corpus = list(set(corpus))
        logger.info(f"corpus size: {len(corpus)}")
        random_order = list(range(len(corpus)))
        np.random.shuffle(random_order)
        corpus = [corpus[i] for i in random_order]
        new_corpus_path = config['data_dir'] / "corpus/corpus.txt"
        if not new_corpus_path.exists():
            new_corpus_path.parent.mkdir(exist_ok=True)
        with open(new_corpus_path, 'w') as fr:
            for line in corpus:
                fr.write(line + "\n")

    if args.do_split:
        new_corpus_path = config['data_dir'] / "corpus/corpus.txt"
        split_save_path = config['data_dir'] / "corpus/train"
        if not split_save_path.exists():
            split_save_path.mkdir(exist_ok=True)
        line_per_file = args.line_per_file
        command = f'split -a 4 -l {line_per_file} -d {new_corpus_path} {split_save_path}/shard_'
        os.system(f"{command}")

    if args.do_vocab:
        vocab.read_data(data_path=config['data_dir'] / "corpus/train")
        vocab.build_vocab()
        vocab.save(file_path=config['data_dir'] / 'corpus/vocab_mapping.pkl')
        vocab.save_bert_vocab(file_path=config['checkpoint_dir'] / 'vocab.txt')
        logger.info(f"vocab size: {len(vocab)}")
        bert_base_config['vocab_size'] = len(vocab)
        save_json(data=bert_base_config, file_path=config['checkpoint_dir'] / 'config.json')

    if args.do_data:
        vocab_list = vocab.load_bert_vocab(config['checkpoint_dir'] / 'vocab.txt')
        data_path = config['data_dir'] / "corpus/train"
        files = sorted([f for f in data_path.iterdir() if f.exists() and "." not in str(f)])
        logger.info("--- pregenerate training data parameters ---")
        logger.info(f'max_seq_len: {args.max_seq_len}')
        logger.info(f"max_predictions_per_seq: {args.max_predictions_per_seq}")
        logger.info(f"masked_lm_prob: {args.masked_lm_prob}")
        logger.info(f"seed: {args.seed}")
        logger.info(f"file num : {args.file_num}")
        for idx in range(args.file_num):
            logger.info(f"pregenetate file_{idx}.json")
            save_filename = data_path / f"file_{idx}.json"
            num_instances = 0
            with save_filename.open('w') as fw:
                for file_idx in range(len(files)):
                    file_path = files[file_idx]
                    file_examples = build_examples(file_path, max_seq_len=args.max_seq_len,
                                                   masked_lm_prob=args.masked_lm_prob,
                                                   max_predictions_per_seq=args.max_predictions_per_seq,
                                                   vocab_list=vocab_list)
                    file_examples = [json.dumps(instance) for instance in file_examples]
                    for instance in file_examples:
                        fw.write(instance + '\n')
                        num_instances += 1
            metrics_file = data_path / f"file_{idx}_metrics.json"
            print(f"num_instances: {num_instances}")
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()
