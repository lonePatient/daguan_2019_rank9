
import numpy as np


class LoadEmbedding(object):
    '''
    word_index:{word:id}
    '''

    def __init__(self, max_features, word_index):
        self.max_features = max_features
        self.word_index = word_index

    def load_glove(self, embedding_path):
        '''
        embedding_path = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        '''

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')[:300]

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path))
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        nb_words = min(self.max_features, len(self.word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            # ALLmight
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_vector = embeddings_index.get(word.capitalize())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def load_fasttext(self, embedding_path):
        '''
        embedding_path = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        '''

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path) if len(o) > 100)
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        nb_words = min(self.max_features, len(self.word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def load_para(self, embedding_path):
        '''
        embedding_path = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        '''

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(embedding_path, encoding="utf8", errors='ignore') if len(o) > 100)
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        # word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(self.word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def load_custom_embedding(self, embedding_path):
        '''
        embedding_path = '../input/embeddings/word2vec.bin'
        '''

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.strip("\n").split(" ")) for o in open(embedding_path, 'r') if o.strip("\n").split(" ")[0]!='')
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        # word_index = tokenizer.word_index
        nb_words = min(self.max_features, len(self.word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in self.word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix
