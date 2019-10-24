import gensim
import numpy as np

from gensim.models import Word2Vec

def createEmbeddingMatrix(modelPath, vocabLength, corpus):
    model = Word2Vec.load(modelPath)
    wordIndex = set(model.wv.vocab)

    embedding_matrix = np.zeros((vocabLength, 200))
    for word, index in corpus.items():
        if word in wordIndex:
            embedding_vector = model.wv[word]
            embedding_matrix[index] = embedding_vector

    return embedding_matrix