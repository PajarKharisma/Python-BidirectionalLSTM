import io
import time
import gensim
import multiprocessing
import numpy as np

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
from datetime import timedelta

corpusPath = '../../vocabulary/w2v/idwiki-latest-pages-articles.xml.bz2'
outputCorpus = '../../vocabulary/w2v/id-wiki.txt'
modelPath = '../../vocabulary/w2v/CBOW/idwiki_word2vec_200.model'

def word2vecfunc():
    # datas = [row.split(' ') for row in datas]
    model = Word2Vec(words, min_count=1, size= 100, workers=3, window=10, sg = 0)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(modelPath)
    print(model.wv.most_similar('jokowi'))

def extractCorpus():
    start_time = time.time()
    print('Streaming wiki...')
    id_wiki = gensim.corpora.WikiCorpus(corpusPath, lemmatize=False, dictionary={})
    article_count = 0

    with io.open(outputCorpus, 'w', encoding="utf-8") as wiki_txt:
        for text in id_wiki.get_texts():

            wiki_txt.write(" ".join(map(str, text)) + '\n')
            article_count += 1

            if article_count % 10000 == 0:
                print('{} articles processed'.format(article_count))

        print('total: {} articles'.format(article_count))

    finish_time = time.time()
    print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

def createModel():
    start_time = time.time()
    print('Training Word2Vec Model...')
    sentences = LineSentence(outputCorpus)
    id_w2v = Word2Vec(sentences, min_count=1, size= 200, workers=multiprocessing.cpu_count()-1, window=10, sg = 0)
    # id_w2v = Word2Vec(sentences, size=200, workers=multiprocessing.cpu_count()-1)
    id_w2v.save(modelPath)
    finish_time = time.time()

    print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

def checkModel():
    model = Word2Vec.load(modelPath)
    # cosmul = model.wv.most_similar(positive=['prabowo', 'presiden'], negative=['jokowi'])
    # for i in cosmul:
    #     print(i)
    words = set(model.wv.vocab)
    # print(words)
    # print('Vocabulary size: %d' % len(words))
    # vector = model.wv['debatfinal']
    # print(np.mean(vector))
    word = 'prabowo'
    if word in words:
        vector = model.wv[word]
        print(vector)
        print(len(vector))
    else:
        print('not exist')

if __name__ == "__main__":
    checkModel()