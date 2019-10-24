import sys
import os
sys.path.append("../")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import modules.PreProcess as pp
import json
from subproc.Vocabulary import *
from keras.preprocessing import sequence
from keras.models import load_model

vocabPath = '../../vocabulary/analysis.json'

def main():
    # tweet = 'itu kan kata ahok kafir..umat islam anti ahok karena ahok itu kafr ditambah ahok itu cina yg suka makan babi & bir'
    tweet = 'Sylvi: bagaimana gurbernur melakukan kekerasan perempuan? Buktinya banyak ibu2 mau foto bareng #DebatFinalPilkadaJKT'
    # tweet = input('masukan tweet ujaran kebencian : ')
    data = []
    data.append(tweet)
    data = pp.getResult(data)
    
    TOP_WORDS = 500
    vocab = Vocabulary(TOP_WORDS, vocabPath)
    x = vocab.TransformSentencesToId2(data, vocabPath)
    x = sequence.pad_sequences(x, maxlen=30)

    model = load_model('lstm_model.h5')
    y = model.predict(x, verbose=0)

    print('data tweet : ', tweet, '\n')
    print('data preprocess : ', data, '\n')
    print('vektor input : ', x, '\n')
    print('output : ', y, '\n')

if __name__ == "__main__":
    main()