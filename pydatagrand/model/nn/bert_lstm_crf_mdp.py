import torch.nn as nn
from ..layers import CRF
from ..layers import LayerNorm
from ..pytorch_transformers.modeling_bert import BertPreTrainedModel
from ..pytorch_transformers.modeling_bert import BertModel


class BERTLSTMCRFMDP(BertPreTrainedModel):
    def __init__(self, config, label2id, device, num_layers=2, lstm_dropout=0.35, mdp_n=5, mdp_p=0.5):
        super(BERTLSTMCRFMDP, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(label2id))
        self.init_weights()

        self.bilstm = nn.LSTM(input_size=config.hidden_size,
                              hidden_size=config.hidden_size // 2,
                              batch_first=True,
                              num_layers=num_layers,
                              dropout=lstm_dropout,
                              bidirectional=True)
        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropouts = nn.ModuleList([
            nn.Dropout(mdp_p) for _ in range(mdp_n)
        ])
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device, is_bert=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.bilstm(sequence_output)
        sequence_output = self.layer_norm(sequence_output)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(sequence_output))
            else:
                logits += self.classifier(dropout(sequence_output))
        return logits / len(self.dropouts)

    def forward_loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        features = self.forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        if labels is not None:
            return features, self.crf.calculate_loss(features, tag_list=labels, lengths=input_lens)
        else:
            return features, None

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
