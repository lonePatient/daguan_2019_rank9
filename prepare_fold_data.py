import argparse
from collections import Counter
from pydatagrand.configs.base import config
from pydatagrand.common.tools import logger
from pydatagrand.common.tools import init_logger
from pydatagrand.common.tools import save_pickle
from sklearn.model_selection import StratifiedKFold

def data_aug1(data):
    new_data = []
    i = 0
    for line in data:
        tags = [x.split("-")[1] for x in line['tag'].split(" ") if "-" in x]
        tags = list(set(tags))
        if ('b' in tags or 'a' in tags) and 'c' in tags:
            c_ = []
            t_ = []
            context = line['context'].split(" ")
            raw_tags = line['tag'].split(" ")
            for c, t in zip(context, raw_tags):
                if 'c' in t:
                    continue
                c_.append(c)
                t_.append(t)
            if i <= 5:
                logger.info("--------- data aug1 -----------")
                logger.info(f"raw: {line['context']}")
                logger.info(f'new: {" ".join(c_)}')
                logger.info(f"raw_tag: {line['tag']}")
                logger.info(f'tag: {" ".join(t_)}')
                i += 1
            new_data.append({"context": " ".join(c_),
                             "tag": " ".join(t_),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
        else:
            continue
    logger.info(f"data aug size: {len(new_data)}")
    return new_data

def data_aug2(data):
    new_data = []
    i = 0
    for line in data:
        tags = [x.split("-")[1] for x in line['tag'].split(" ") if "-" in x]
        tags = list(set(tags))
        if 'b' in tags and 'a' in tags:
            c_ = []
            t_ = []
            context = line['context'].split(" ")
            raw_tags = line['tag'].split(" ")
            for c, t in zip(context, raw_tags):
                if 'c' in t or 'b' in t:
                    continue
                c_.append(c)
                t_.append(t)
            if i <= 2:
                logger.info("--------- data aug2 -----------")
                logger.info(f"raw: {line['context']}")
                logger.info(f'new: {" ".join(c_)}')
                logger.info(f"raw_tag: {line['tag']}")
                logger.info(f'tag: {" ".join(t_)}')
                i += 1
            new_data.append({"context": " ".join(c_),
                             "tag": " ".join(t_),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
        else:
            continue
    logger.info(f"data2 aug size: {len(new_data)}")
    return new_data

def data_aug3(data):
    new_data = []
    i = 0
    for line in data:
        tags = [x.split("-")[1] for x in line['tag'].split(" ") if "-" in x]
        tags = list(set(tags))
        if 'b' in tags and 'a' in tags and 'c' in tags:
            c_1 = []
            t_1 = []
            c_2 = []
            t_2 = []
            context = line['context'].split(" ")
            raw_tags = line['tag'].split(" ")
            for c, t in zip(context, raw_tags):
                if 'a' in t :
                    continue
                c_1.append(c)
                t_1.append(t)

            for c, t in zip(context, raw_tags):
                if 'b' in t:
                    continue
                c_2.append(c)
                t_2.append(t)
            if i <= 2:
                logger.info("--------- data aug3 -----------")
                logger.info(f"raw: {line['context']}")
                logger.info(f'new: {" ".join(c_1)}')
                logger.info(f"raw_tag: {line['tag']}")
                logger.info(f'tag: {" ".join(t_1)}')
                i += 1
            new_data.append({"context": " ".join(c_1),
                             "tag": " ".join(t_1),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
            new_data.append({"context": " ".join(c_2),
                             "tag": " ".join(t_2),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
        else:
            continue
    logger.info(f"data3 aug size: {len(new_data)}")
    return new_data

def make_folds(args):
    train = []
    train_path = config['data_dir'] / 'train.txt'
    with open(str(train_path), 'r') as fr:
        idx = 0
        for line in fr:
            json_d = {}
            line = line.strip("\n")
            context = []
            tags = []
            lines = line.split("  ")
            for seg in lines:
                segs = seg.split("/")
                seg_text = segs[0].split("_")
                seg_label = segs[1]
                context.extend(seg_text)
                if seg_label == 'o':
                    tags.extend(["O"] * len(seg_text))
                elif len(seg_text) == 1:
                    tags.extend([f"S-{seg_label}"])
                else:
                    head_label = f"B-{seg_label}"
                    tags.extend([head_label])
                    tags.extend([f"I-{seg_label}"] * (len(seg_text) - 1))
            json_d['id'] = idx
            json_d['context'] = " ".join(context)
            json_d['tag'] = " ".join(tags)
            json_d['raw_context'] = line
            la = [x.split("-")[1] for x in tags if '-' in x]
            la = list(set(la))
            if len(la) == 0:
                y = 0
            elif len(la) == 3:
                y = 4
            elif len(la) == 2:
                if 'a' in la and 'b' in la:
                    y = 1
                if 'a' in la and 'c' in la:
                    y = 2
                if 'b' in la and 'c' in la:
                    y = 3
            elif len(la) == 1:
                if la[0] == 'a':
                    y = 5
                if la[0] == 'b':
                    y = 6
                if la[0] == 'c':
                    y = 7
            else:
                raise ValueError("tag is error")
            json_d['y'] = y
            idx += 1
            train.append(json_d)

    y_counter = Counter()
    y_counter.update([x['y'] for x in train])
    print(y_counter)
    X = train
    y = [d['y'] for d in train]
    sss = StratifiedKFold(n_splits=args.folds, random_state=args.seed, shuffle=True)
    for fold, (train_index, test_index) in enumerate(sss.split(X, y)):
        logger.info(f'fold-{fold} info:')
        logger.info(f'raw train data size: {len(train_index)}')
        logger.info(f'raw valid data size: {len(test_index)}')
        X_train = [X[i] for i in train_index]
        if args.do_aug:
            new_data1 = data_aug1(X_train)
            new_data2 = data_aug2(X_train)
            new_data3 = data_aug3(X_train)
            X_train.extend(new_data1)
            X_train.extend(new_data2)
            X_train.extend(new_data3)
            logger.info(f"After data augmentation, train data size: {len(X_train)}")
        X_test = [X[i] for i in test_index]
        train_file_name = f'{args.data_name}_train_fold_{fold}.pkl'
        dev_file_name = f'{args.data_name}_valid_fold_{fold}.pkl'
        save_pickle(X_train, file_path=config['data_dir'] / train_file_name)
        save_pickle(X_test, file_path=config['data_dir'] / dev_file_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--do_aug', action='store_true')
    parser.add_argument('--data_name',default='datagrand',type=str)
    args = parser.parse_args()
    init_logger(log_file=config['log_dir'] / 'prepare_fold_data.log')
    make_folds(args)

if __name__ == "__main__":
    main()
