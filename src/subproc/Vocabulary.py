import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer

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
    def __init__(self, maxFeatures, data, path):
        self.__maxFeatures = maxFeatures
        self.__data = data
        self.__path = path
    
    def prepareVocabulary(self):
        vectorizer = CountVectorizer(max_features=self.__maxFeatures)
        vectorizer.fit(self.__data)
        datajson = vectorizer.vocabulary_

        with open(self.__path, 'w') as fp:
            json.dump(datajson, fp, sort_keys=True, indent=4, cls=NumpyEncoder)

    def getVocab(self, path):
        with open(path) as json_file:
            self.input_word_index = json.load(json_file)
        
        return self.input_word_index

    def transformSentencesToId(self, sentences, path):
        with open(path) as json_file:
            self.input_word_index = json.load(json_file)

        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))

            for t, word in enumerate(words):
                if word in self.input_word_index:
                    vector[t] = self.input_word_index[word]
                else:
                    pass
                
            vectors.append(vector)
            
        return vectors