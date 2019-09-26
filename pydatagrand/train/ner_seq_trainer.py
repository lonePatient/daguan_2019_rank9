import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device,prepare_device
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from .ner_utils import SeqEntityScore
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    def __init__(self, model, n_gpu, logger, optimizer, lr_scheduler,
                label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 fp16=None, resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.fp16 = fp16
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device ,_ = prepare_device(n_gpu)
        self.id2label = {y: x for x, y in label2id.items()}
        self.entity_score = SeqEntityScore(self.id2label)
        self.start_epoch = 1
        self.global_step = 0
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if resume_path:
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            resume_dict = torch.load(resume_path / 'checkpoint_info.bin')
            best = resume_dict['epoch']
            self.start_epoch = resume_dict['epoch']
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")

    def save_info(self, epoch, best):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def valid_epoch(self, data_loader):
        pbar = ProgressBar(n_total=len(data_loader), desc='Evaluating')
        self.entity_score.reset()
        valid_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, input_lens = batch
            input_lens = input_lens.cpu().detach().numpy().tolist()
            self.model.eval()
            with torch.no_grad():
                features, loss = self.model.forward_loss(input_ids, segment_ids, input_mask, label_ids, input_lens)
                tags, _ = self.model.crf._obtain_labels(features, self.id2label, input_lens)
            valid_loss.update(val=loss.item(), n=input_ids.size(0))
            pbar(step=step, info={"loss": loss.item()})
            label_ids = label_ids.to('cpu').numpy().tolist()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == self.label2id['[SEP]']:
                        self.entity_score.update(pred_paths=[temp_2], label_paths=[temp_1])
                        break
                    else:
                        temp_1.append(self.id2label[label_ids[i][j]])
                        temp_2.append(tags[i][j])
            valid_info, class_info = self.entity_score.result()
            info = {f'valid_{key}': value for key, value in valid_info.items()}
            info['valid_loss'] = valid_loss.avg
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            return info, class_info

    def train_epoch(self, data_loader):
        pbar = ProgressBar(n_total=len(data_loader), desc='Training')
        tr_loss = AverageMeter()
        for step, batch in enumerate(data_loader):
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, input_lens = batch
            input_lens = input_lens.cpu().detach().numpy().tolist()
            _, loss = self.model.forward_loss(input_ids, segment_ids, input_mask, label_ids, input_lens)
            if len(self.n_gpu.split(",")) >= 2:
                loss = loss.mean()
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            tr_loss.update(loss.item(), n=1)
            pbar(step=step, info={'loss': loss.item()})
        info = {'loss': tr_loss.avg}
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return info

    def train(self, train_data, valid_data, epochs, seed):
        seed_everything(seed)
        for epoch in range(self.start_epoch, self.start_epoch + int(epochs)):
            self.logger.info(f"Epoch {epoch}/{int(epochs)}")
            train_log = self.train_epoch(train_data)
            valid_log, class_info = self.valid_epoch(valid_data)

            logs = dict(train_log, **valid_log)
            show_info = f'Epoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
            self.logger.info(show_info)
            self.logger.info("The entity scores of valid data : ")
            for key, value in class_info.items():
                info = f'Entity: {key} - ' + "-".join([f' {key_}: {value_:.4f} ' for key_, value_ in value.items()])
                self.logger.info(info)

            if hasattr(self.lr_scheduler,'epoch_step'):
                self.lr_scheduler.epoch_step(metrics=logs[self.model_checkpoint.monitor], epoch=epoch)
            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs)

            # save model
            if self.model_checkpoint:
                state = self.save_info(epoch, best=logs[self.model_checkpoint.monitor])
                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor], state=state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
