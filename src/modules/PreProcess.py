import nltk
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import pandas as pd

slangwordsPath = '../../vocabulary/slangwords.csv'

REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\@)|(\\\)|(\.)|(\{)|(\})|(\_)|(\^)|(\|)|(\&)|(\%)|(\<)|(\>)|(\*)|(\~)")
REPLACE_NO_SPACE = re.compile("(\#)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\@)|(\/)|(\\\)")
LINK = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

def getResult(reviews):
    reviews = [removeLink(line) for line in reviews]

    default_stop_words = nltk.corpus.stopwords.words('indonesian')
    stopwords = set(default_stop_words)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    swCorpus = readSlangwords()

    reviews = [LINK.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line.lower()) for line in reviews]
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [removeDigit(line) for line in reviews]
    reviews = [removeSlangwords(line, swCorpus) for line in reviews]
    # reviews = [stemming(line, stemmer) for line in reviews]
    reviews = [removeStopWords(line,stopwords) for line in reviews]
    reviews = [removeUnnecessary(line) for line in reviews]
    
    return reviews

def stemming(reviews, stemmer):
    katadasar = stemmer.stem(reviews)
    return katadasar

def removeStopWords(line, stopwords):
    words = []
    for word in line.split(" "):
        word = word.strip()
        if word not in stopwords and word != "" and word != "&":
            words.append(word)

    return " ".join(words)

def removeLink(s):
    corpus = ('https:', 'http:', '@', 'href', '#')
    l = s.split()
    i = 0
    while i < len(l):
        if l[i].startswith(corpus):
            del l[i]
        else:
            i+=1

    s = ' '.join(l)
    return s

def removeDigit(s):
    result = ''.join([i for i in s if not i.isdigit()])
    return result

def removeUnnecessary(s):
    corpus = ['u', 'n', 'rt']
    words = []
    for word in s.split(" "):
        if word not in corpus:
            words.append(word)

    return " ".join(words)

def getMaxPad(data):
    word_count = lambda sentence: len(word_tokenize(sentence))
    longest_sentence = max(data, key=word_count)
    length_long_sentence = len(word_tokenize(longest_sentence))
    return length_long_sentence

def removeSlangwords(s, corpus):
    words = []
    for word in s.split(" "):
        word = word.strip()
        if word in corpus:
            word = corpus[word]
        words.append(word)

    return " ".join(words)
        

def readSlangwords():
    slangwordsCorpus = {}
    df = pd.read_csv(slangwordsPath, error_bad_lines=False)
    for index, (key, value) in enumerate(zip(df['key'], df['value'])):
        slangwordsCorpus[key] = value

    return slangwordsCorpus