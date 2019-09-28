import torch
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pydatagrand.train.ner_span_trainer import Trainer
from torch.utils.data import DataLoader
from pydatagrand.io.bert_span_processor import BertProcessor
from pydatagrand.common.tools import init_logger, logger
from pydatagrand.common.tools import seed_everything
from pydatagrand.configs.base import config
from pydatagrand.model.nn.bert_lstm_span import BERTLSTMSpan
from pydatagrand.callback import ModelCheckpoint
from pydatagrand.callback import TrainingMonitor
from pydatagrand.callback import BertAdam
from pydatagrand.callback import BERTReduceLROnPlateau
from torch.utils.data import RandomSampler, SequentialSampler
from pydatagrand.callback import Lookahead
from pydatagrand.train.losses import SpanLoss
warnings.filterwarnings("ignore")


def run_train(args):
    processor = BertProcessor(vocab_path=args.pretrain_model / 'vocab.txt', do_lower_case=args.do_lower_case)
    processor.tokenizer.save_vocabulary(str(args.model_path))
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}_train_fold_{args.fold}.pkl")
    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_file=config[
                                                               'data_dir'] / f"cached_train_span_examples")
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_file=config[
                                                               'data_dir'] / "cached_train_span_features_{}".format(
                                                   args.train_max_seq_len))
    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)
    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = processor.get_dev(config['data_dir'] / f'{args.data_name}_valid_fold_{args.fold}.pkl')
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_file=config[
                                                               'data_dir'] / f"cached_valid_span_examples")
    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_file=config[
                                                               'data_dir'] / "cached_valid_span_features_{}".format(
                                                   args.eval_max_seq_len))

    logger.info("initializing model")
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BERTLSTMSpan.from_pretrained(args.resume_path, label2id=label2id, soft_label=args.soft_label)
    else:
        model = BERTLSTMSpan.from_pretrained(args.pretrain_model, label2id=label2id, soft_label=args.soft_label)
    model = model.to(args.device)
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    bert_param_optimizer = list(model.bert.named_parameters())
    lstm_param_optimizer = list(model.bilstm.named_parameters())
    start_fc_param_optimizer = list(model.start_fc.named_parameters())
    end_fc_param_optimizer = list(model.end_fc.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.0005},
        {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.0005},
        {'params': [p for n, p in start_fc_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01,
         'lr': 0.0005},
        {'params': [p for n, p in start_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.0005},
        {'params': [p for n, p in end_fc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         'lr': 0.0005},
        {'params': [p for n, p in end_fc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': 0.0005}
    ]
    if args.optimizer == 'adam':
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                             warmup=args.warmup_proportion, t_total=t_total)
    else:
        base_optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                                  warmup=args.warmup_proportion, t_total=t_total)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    lr_scheduler = BERTReduceLROnPlateau(optimizer, lr=args.learning_rate, mode=args.mode, factor=0.5, patience=5,
                                         verbose=1, epsilon=1e-8, cooldown=0, min_lr=0, eps=1e-8)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)
    model_checkpoint = ModelCheckpoint(checkpoint_dir=args.model_path,
                                       mode=args.mode,
                                       monitor=args.monitor,
                                       arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(n_gpu=args.n_gpu,
                      model=model,
                      logger=logger,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      label2id=label2id,
                      criterion = SpanLoss(),
                      training_monitor=train_monitor,
                      fp16=args.fp16,
                      resume_path=args.resume_path,
                      grad_clip=args.grad_clip,
                      model_checkpoint=model_checkpoint,
                      gradient_accumulation_steps=args.gradient_accumulation_steps)
    trainer.train(train_data=train_dataloader, valid_data=valid_features, epochs=args.epochs, seed=args.seed)


def run_test(args):
    from pydatagrand.callback import ProgressBar
    from pydatagrand.train.ner_utils import bert_extract_item
    from pydatagrand.common.tools import save_pickle

    processor = BertProcessor(args.model_path / 'vocab.txt', do_lower_case=args.do_lower_case)
    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    model = BERTLSTMSpan.from_pretrained(args.resume_path, label2id=label2id, soft_label=args.soft_label)
    model.to(args.device)
    max_seq_len = args.eval_max_seq_len
    tokenizer = processor.tokenizer
    test_data = []
    with open(str(config['data_dir'] / 'test.txt'), 'r') as fr:
        for line in fr:
            line = line.strip("\n")
            test_data.append(line)
    test_result_path = config['result_dir'] / f'{args.arch}_test_submit.txt'
    fw = open(str(test_result_path), 'w')
    pbar = ProgressBar(n_total=len(test_data), desc='Testing')
    pred_logits = []
    for step, line in enumerate(test_data):
        token_a = line.split("_")
        tokens = tokenizer.tokenize(token_a)
        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            start_logits, end_logits = model(input_ids, segment_ids, input_mask, start_point=None)
            R = bert_extract_item(start_logits, end_logits)
            pred_logits.append([start_logits, end_logits])
            if len(R) == 0:
                record = "_".join(token_a) + "/o"
            else:
                labels = []
                for (s_l, i, j) in R:
                    lb = id2label[s_l]
                    tokens = token_a[i:j + 1]
                    labels.append("_".join(tokens) + f"/{lb}")
                record = "  ".join(labels)
        fw.write(record + "\n")
        pbar(step=step)
    fw.close()
    save_pickle(pred_logits, file_path=str(config['result_dir'] / f'{args.arch}_test_logits.pkl'))


def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert_lstm_span', type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--data_name', default='datagrand', type=str)
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'lookahead'])
    parser.add_argument('--markup', default='bios', type=str, choices=['bio', 'bios'])
    parser.add_argument('--checkpoint', default=900000, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument("--epochs", default=50.0, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--mode", default='max', type=str)
    parser.add_argument("--monitor", default='valid_f1', type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument('--eval_batch_size', default=48, type=int)
    parser.add_argument("--train_max_seq_len", default=128, type=int)
    parser.add_argument("--eval_max_seq_len", default=512, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    args = parser.parse_args()

    args.pretrain_model = config['checkpoint_dir'] / f'lm-checkpoint-{args.checkpoint}'
    args.device = torch.device(f"cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.arch = args.arch + f"_{args.markup}_fold_{args.fold}"
    if args.optimizer == 'lookahead':
        args.arch += "_lah"
    args.model_path = config['checkpoint_dir'] / args.arch
    args.model_path.mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    init_logger(log_file=config['log_dir'] / f"{args.arch}.log")
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
