import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import LayerNorm
from ..pytorch_transformers.modeling_bert import BertPreTrainedModel
from ..pytorch_transformers.modeling_bert import BertModel
from ..layers.linears import PoolerEndLogits, PoolerStartLogits


class BERTLSTMSpan(BertPreTrainedModel):
    def __init__(self, config, label2id, num_layers=2, lstm_dropout=0.35, soft_label=False):
        super(BERTLSTMSpan, self).__init__(config)
        self.soft_label = soft_label
        self.num_labels = len(label2id)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              batch_first=True,
                              num_layers=num_layers,
                              dropout=lstm_dropout,
                              bidirectional=True)
        self.layer_norm = LayerNorm(config.hidden_size)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_point=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.layer_norm(sequence_output)
        ps1 = self.start_fc(sequence_output)
        if start_point is not None:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                start_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                start_logits.zero_()
                start_logits = start_logits.to(self.device)
                start_logits.scatter_(2, start_point.unsqueeze(2), 1)
            else:
                start_logits = start_point.unsqueeze(2).float()

        else:
            start_logits = F.softmax(ps1, -1)
            if not self.soft_label:
                start_logits = torch.argmax(start_logits, -1).unsqueeze(2).float()
        ps2 = self.end_fc(sequence_output, start_logits)
        return ps1, ps2

    def unfreeze(self, start_layer=6, end_layer=12):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer):
            set_trainable(self.bert.encoder.layer[i], True)
