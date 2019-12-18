import sys
import os
sys.path.append("../")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import modules.PreProcess as pp
import model.ListOfModels as lm
import json
from subproc.Vocabulary import *
from keras.preprocessing import sequence
from keras.models import load_model
import keras_metrics as km

NUM_OF_ATTRIBUTES = 2500
vocabPath = '../../vocabulary/corpus.json'

def main():
    # tweet = 'itu kan kata ahok kafir..umat islam anti ahok karena ahok itu kafr ditambah ahok itu cina yg suka makan babi & bir'
    # tweet = 'Sylvi: bagaimana gurbernur melakukan kekerasan perempuan? Buktinya banyak ibu2 mau foto bareng #DebatFinalPilkadaJKT'
    # tweet = input('masukan tweet ujaran kebencian : ')
    # data = [
    #     'apakah perlu semua berita ahok harus menggunakan penista agama? misalnya penista agama sedang kampanye di jakarta pusat',
    #     'kemarin kan ada berita tentang penyakit meningitis dari babi, maksud gw apa nyambungnya sama ahok',
    #     'huu sylvi tak tahu apa - apa asal ngoceh keliatan bloonnya',
    #     'anies anda sadis topengmu terbuka lebar malam ini biar warga dki yang menilai apa yang anda katakan adalah pengecut',
    #     'ahok memelihara anjing',
    #     'Mereka Siang MALAM Maki2 Ulama Kita, Nongkrongi Medsos Ulama Bela PENISTA Agama yg SOMBONGNYA Minta AMPUN.',
    #     'program paslon no. 1 ini programnya ngambang yaiyalah pakkk gak ada program diaa cma ada celaan'
    # ]
    data = ['Shame on you silvy !! Ga malu fitnah?? Tuh rasain pak ahok panas, skak mat #DebatFinalPilkadaJKT']
    cleanData = pp.getResult(data)
    vocab = Vocabulary()
    x = vocab.transformSentencesToId2(cleanData, NUM_OF_ATTRIBUTES, vocabPath)
    x = sequence.pad_sequences(x, maxlen=21)

    for d in data:
        print(d)
    model = load_model('lstm_model.h5')
    y = model.predict(np.array(x), verbose=0)
    # print('vektor input : ', xx, '\n')
    print(y, '\n')
    # print('='*20)

if __name__ == "__main__":
    main()