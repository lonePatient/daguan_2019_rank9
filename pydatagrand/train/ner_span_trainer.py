import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import model_device,prepare_device
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from .ner_utils import SpanEntityScore
from .ner_utils import bert_extract_item
from torch.nn.utils import clip_grad_norm_

class Trainer(object):
    def __init__(self, model, n_gpu, logger, criterion, optimizer, lr_scheduler,
                 label2id, gradient_accumulation_steps, grad_clip=0.0,early_stopping=None,
                 fp16=None, resume_path=None, training_monitor=None, model_checkpoint=None):

        self.n_gpu = n_gpu
        self.model = model
        self.logger = logger
        self.fp16 = fp16
        self.criterion = criterion
        self.optimizer = optimizer
        self.label2id = label2id
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model)
        self.device, _ = prepare_device(n_gpu)
        self.id2label = {y: x for x, y in label2id.items()}
        self.entity_score = SpanEntityScore(self.id2label)
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

    def valid_epoch(self, data_features):
        pbar = ProgressBar(n_total=len(data_features), desc='Evaluating')
        self.entity_score.reset()
        valid_loss = AverageMeter()
        for step, f in enumerate(data_features):
            input_lens = f.input_len
            input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(self.device)
            input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(self.device)
            segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(self.device)
            start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(self.device)
            end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(self.device)
            subjects = f.subjects

            self.model.eval()
            with torch.no_grad():
                start_logits, end_logits = self.model(input_ids, segment_ids, input_mask, start_point=None)
                start_loss = self.criterion(start_logits.view(-1, 4), start_ids.view(-1), input_mask)
                end_loss = self.criterion(end_logits.view(-1, 4), end_ids.view(-1), input_mask)
                loss = start_loss + end_loss
            valid_loss.update(val=loss.item(), n=input_ids.size(0))
            R = bert_extract_item(start_logits, end_logits)
            T = subjects
            self.entity_score.update(true_subject=T, pred_subject=R)
            pbar(step=step, info={"loss": loss.item()})
        print(' ')
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
            input_ids, input_mask, segment_ids, start_ids, end_ids, input_lens = batch
            start_logits, end_logits = self.model(input_ids, segment_ids, input_mask, start_point=start_ids)
            start_loss = self.criterion(start_logits.view(-1, len(self.id2label)), start_ids.view(-1), input_mask)
            end_loss = self.criterion(end_logits.view(-1, len(self.id2label)), end_ids.view(-1), input_mask)
            loss = start_loss + end_loss

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
