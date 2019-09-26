import argparse
import numpy as np
from pydatagrand.configs.base import config
from pydatagrand.common.tools import load_pickle
from pydatagrand.train.ner_utils import get_entities
from pydatagrand.common.tools import seed_everything
from collections import Counter
from glob import glob
from datetime import datetime

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    def get_labels(self):
        return ["X", "O", "B-a", "I-a", "B-b", "I-b", "B-c", "I-c", "S-a", "S-b", "S-c", "[CLS]", "[SEP]"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",default='ner',type=str)
    parser.add_argument("--do_test",action='store_true')
    parser.add_argument("--do_eval",action='store_true')
    parser.add_argument('--seed',default=42,type=str)
    args = parser.parse_args()

    seed_everything(seed=args.seed)
    dt = str(datetime.today()).split(" ")[0]
    test_path = config['data_dir'] / 'test.txt'
    test_result_path =  config['result'] / f'{dt}_submit_test.txt'
    processors = {"ner": NerProcessor}
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list, 0)}
    test_data = []
    with open(str(test_path), 'r') as fr:
        for line in fr:
            line = line.strip("\n")
            test_data.append(line)
    fw = open(str(test_result_path), 'w')
    cv_test_pred = []
    for file in glob(f"{str(config['result']/ '*.pkl')}"):
        data = load_pickle(file)
        cv_test_pred.append(data)
    vote_pred = []
    for i in range(len(test_data)):
        t = [np.array([x[i]]).T for x in cv_test_pred]
        t2 = np.concatenate(t, axis=1)
        t3 = []
        for line in t2:
            c = Counter()
            c.update(line)
            t3.append(c.most_common(1)[0][0])
        vote_pred.append(t3)
    for tag,line in zip(vote_pred,test_data):
        token_a = line.split("_")
        label_entities = get_entities(tag, id2label)
        if len(label_entities) == 0:
            record = "_".join(token_a) + "/o"
        else:
            labels = []
            label_entities = sorted(label_entities, key=lambda x: x[1])
            o_s = 0
            for i, entity in enumerate(label_entities):
                begin = entity[1]
                end = entity[2]
                tag = entity[0]
                if begin != o_s:
                    labels.append("_".join(token_a[o_s:begin]) + "/o")
                labels.append("_".join(token_a[begin:end + 1]) + f"/{tag}")
                o_s = end + 1
                if i == len(label_entities) - 1:
                    if o_s <= len(token_a) - 1:
                        labels.append("_".join(token_a[o_s:]) + "/o")
            record = "  ".join(labels)
        fw.write(record + "\n")
    fw.close()

if __name__ == "__main__":
    main()