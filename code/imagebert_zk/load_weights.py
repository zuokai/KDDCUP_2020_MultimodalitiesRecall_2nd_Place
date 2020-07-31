import numpy as np
import os
import gc
import io

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype=np.float16)


def load_embed(path, dim=300, word_index=None):
    embedding_index = {}
    with io.open(path, mode="r", encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            word, arr = l[0], l[1:]
            if len(arr) != dim:
                print("[!] l = {}".format(l))
                continue
            if word_index and word not in word_index:
                continue
            word, arr = get_coefs(word, arr)
            embedding_index[word] = arr
    return embedding_index


def build_matrix(path, word_index=None, max_features=None, dim=300):
    embedding_index = load_embed(path, dim=dim, word_index=word_index)
    max_features = len(word_index) + 1 if max_features is None else max_features
    embedding_matrix = np.zeros((max_features + 1, dim))
    unknown_words = []

    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                #                 print(word)
                unknown_words.append(word)
    return embedding_matrix, unknown_words


def load_word_embed(word_embed_glove="word2vec/glove.840B.300d.txt",
                    word_embed_crawl="word2vec/crawl-300d-2M.vec",
                    save_filename="./word_embedding_matrix",
                    word_index=None):
    """
    (30524, 300) 7590
    (30524, 300) 7218
    """
    if os.path.exists(save_filename + ".npy"):
        word_embedding_matrix = np.load(save_filename + ".npy").astype("float32")
    else:
        word_embedding_matrix, tx_unk = build_matrix(word_embed_glove, word_index=word_index, dim=300)

        print(word_embedding_matrix.shape, len(tx_unk))

        word_embedding_matrix_v2, tx_unk = build_matrix(word_embed_crawl, word_index=word_index, dim=300)

        print(word_embedding_matrix_v2.shape, len(tx_unk))

        word_embedding_matrix = np.concatenate([word_embedding_matrix, word_embedding_matrix_v2], axis=1)

        gc.collect()
        np.save(save_filename, word_embedding_matrix)
    return word_embedding_matrix


#word_embedding_matrix = load_word_embed(word_index=word_index)