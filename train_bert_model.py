import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from tempfile import TemporaryDirectory
from pydatagrand.common.tools import logger, init_logger
from pydatagrand.configs.base import config
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pydatagrand.common.tools import AverageMeter
from pydatagrand.train.metrics import LMAccuracy
from pydatagrand.model.pytorch_transformers.modeling_bert import BertForMaskedLM, BertConfig
from pydatagrand.model.pytorch_transformers.file_utils import CONFIG_NAME
from pydatagrand.model.pytorch_transformers.tokenization_bert import BertTokenizer
from pydatagrand.model.pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pydatagrand.common.tools import seed_everything

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids")
init_logger(log_file=config['log_dir'] / ("train_bert_model.log"))


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1
    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, file_id, tokenizer, reduce_memory=False):
        self.tokenizer = tokenizer
        self.file_id = file_id
        data_file = training_path / f"file_{self.file_id}.json"
        metrics_file = training_path / f"file_{self.file_id}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        logger.info(f"Loading training examples for {str(data_file)}")
        with data_file.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logger.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)))


def main():
    parser = ArgumentParser()
    parser.add_argument("--file_num", type=int, default=10,
                        help="Number of pregenerate file")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs to train for")
    parser.add_argument('--num_eval_steps', default=2000)
    parser.add_argument('--num_save_steps', default=5000)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size", default=18, type=int,
                        help="Total batch size for training.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

    pregenerated_data = config['data_dir'] / "corpus/train"
    assert pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by prepare_lm_data_mask.py!"

    samples_per_epoch = 0
    for i in range(args.file_num):
        data_file = pregenerated_data / f"file_{i}.json"
        metrics_file = pregenerated_data / f"file_{i}_metrics.json"
        if data_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch += metrics['num_training_examples']
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            break
    logger.info(f"samples_per_epoch: {samples_per_epoch}")
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        f"device: {device} , distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    seed_everything(args.seed)
    tokenizer = BertTokenizer(vocab_file=config['checkpoint_dir'] / 'vocab.txt')
    total_train_examples = samples_per_epoch * args.epochs

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    args.warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)

    # Prepare model
    with open(str(config['checkpoint_dir'] / 'config.json'), "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    print(json_config)
    bert_config = BertConfig.from_json_file(str(config['checkpoint_dir'] / 'config.json'))
    model = BertForMaskedLM(config=bert_config)
    # model = BertForMaskedLM.from_pretrained(config['checkpoint_dir'] / 'checkpoint-580000')
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    global_step = 0
    metric = LMAccuracy()
    tr_acc = AverageMeter()
    tr_loss = AverageMeter()

    train_logs = {}
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_examples}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = {num_train_optimization_steps}")
    logger.info(f"  warmup_steps = {args.warmup_steps}")

    seed_everything(args.seed)  # Added here for reproducibility
    for epoch in range(args.epochs):
        for idx in range(args.file_num):
            epoch_dataset = PregeneratedDataset(file_id=idx, training_path=pregenerated_data, tokenizer=tokenizer,
                                                reduce_memory=args.reduce_memory)
            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
            model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids = batch
                outputs = model(input_ids=input_ids, token_type_ids=segment_ids,
                                attention_mask=input_mask, masked_lm_labels=lm_label_ids)
                pred_output = outputs[1]
                loss = outputs[0]
                metric(logits=pred_output.view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                nb_tr_steps += 1
                tr_acc.update(metric.value(), n=input_ids.size(0))
                tr_loss.update(loss.item(), n=1)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    lr_scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step % args.num_eval_steps == 0:
                    train_logs['loss'] = tr_loss.avg
                    train_logs['acc'] = tr_acc.avg
                    show_info = f'\n[Training]:[{epoch}/{args.epochs}]{global_step}/{num_train_optimization_steps} ' + "-".join(
                        [f' {key}: {value:.4f} ' for key, value in train_logs.items()])
                    logger.info(show_info)
                    tr_acc.reset()
                    tr_loss.reset()

                if global_step % args.num_save_steps == 0:
                    if args.local_rank in [-1, 0] and args.num_save_steps > 0:
                        # Save model checkpoint
                        output_dir = config['checkpoint_dir'] / f'lm-checkpoint-{global_step}'
                        if not output_dir.exists():
                            output_dir.mkdir()
                        # save model
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(str(output_dir))
                        torch.save(args, str(output_dir / 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        # save config
                        output_config_file = output_dir / CONFIG_NAME
                        with open(str(output_config_file), 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                        # save vocab
                        tokenizer.save_vocabulary(output_dir)


if __name__ == '__main__':
    main()
