import gensim
import numpy as np

from gensim.models import Word2Vec, KeyedVectors

def createEmbeddingMatrix(modelPath, vocabLength, corpus):
    model = KeyedVectors.load_word2vec_format(modelPath, binary=True)
    model.init_sims(replace=True)
    wordIndex = set(model.wv.vocab)

    embedding_matrix = np.zeros((vocabLength, 200))
    for word, index in corpus.items():
        if index < vocabLength:
            if word in wordIndex:
                embedding_vector = model[word]
                embedding_matrix[index] = embedding_vector

    return embedding_matrix

def getEmbeddingValue(modelPath, corpus):
    model = KeyedVectors.load_word2vec_format(modelPath, binary=True)
    model.init_sims(replace=True)
    wordIndex = set(model.wv.vocab)

    outputVector = []
    for sentences in corpus:
        tmpVector = []
        for s in sentences.split(" "):
            if s in wordIndex:
                vector = model[s]
                tmpVector.append(np.mean(vector))
            else:
                tmpVector.append(0)
        outputVector.append(tmpVector)
    
    return outputVector