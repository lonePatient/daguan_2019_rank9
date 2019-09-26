import torch
import torch.nn as nn
import torch.nn.functional as F


def nn_init(nn_module, method='xavier'):
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)


def init_weight(weight, method):
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


class PretrainedEmbedding(nn.Module):
    def __init__(self, embedding_matrix, requires_grad=False):
        super(PretrainedEmbedding, self).__init__()
        embed_size = embedding_matrix.shape[1]
        max_features = embedding_matrix.shape[0]
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = requires_grad

    def forward(self, indices):
        return self.embedding(indices)


class ProjSumEmbedding(nn.Module):

    def __init__(self, embedding_matrices, output_size):
        super(ProjSumEmbedding, self).__init__()
        assert len(embedding_matrices) > 0

        self.embedding_count = len(embedding_matrices)
        self.output_size = output_size
        self.embedding_projectors = nn.ModuleList()
        for embedding_matrix in embedding_matrices:
            embedding_dim = embedding_matrix.shape[1]
            projection = nn.Linear(embedding_dim, self.output_size)
            nn_init(projection)

            self.embedding_projectors.append(nn.Sequential(
                PretrainedEmbedding(embedding_matrix),
                projection
            ))

    def forward(self, x):
        projected = [embedding_projector(x) for embedding_projector in self.embedding_projectors]
        return F.relu(sum(projected))
