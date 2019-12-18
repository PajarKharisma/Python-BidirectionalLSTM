import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Vocabulary:

    def __init__(self, maxFeatures=0, data=None, path=''):
        self.__maxFeatures = maxFeatures
        self.__data = data
        self.__path = path
        self.__wordTokenizer = Tokenizer(num_words=maxFeatures)
    
    def prepareVocabulary(self):
        self.__wordTokenizer.fit_on_texts(self.__data)
        datajson = self.__wordTokenizer.word_index
        with open(self.__path, 'w') as fp:
            json.dump(datajson, fp, sort_keys=True, indent=4, cls=NumpyEncoder)

    def getVocab(self, path):
        with open(path) as json_file:
            self.input_word_index = json.load(json_file)
        
        return self.input_word_index

    def transformSentencesToId(self):
        return self.__wordTokenizer.texts_to_sequences(self.__data)

    def transformSentencesToId2(self, sentences, vocabLength, path):
        with open(path) as json_file:
            self.input_word_index = json.load(json_file)

        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))

            for t, word in enumerate(words):
                if word in self.input_word_index:
                    val = self.input_word_index[word]
                    if val < vocabLength:
                        vector[t] = self.input_word_index[word]
                else:
                    pass
                
            vectors.append(vector)
            
        return vectors


    def transformSentencesToOneHot(self, dataInt):
        return [to_categorical(data) for data in dataInt]