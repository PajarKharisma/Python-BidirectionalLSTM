import nltk
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\@)|(\\\)|(\.)")
REPLACE_NO_SPACE = re.compile("(\#)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\@)|(\/)|(\\\)")
LINK = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

def getResult(reviews):
    reviews = [removeLink(line) for line in reviews]

    default_stop_words = nltk.corpus.stopwords.words('indonesian-sentimen')
    stopwords = set(default_stop_words)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    # reviews = [stemming(line, stemmer) for line in reviews]
    reviews = [removeStopWords(line,stopwords) for line in reviews]
    
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
    list = s.split()
    i = 0
    while i < len(list):
        if list[i].startswith('https:') or list[i].startswith('@') or list[i].startswith('http:'):
            del list[i]
        else:
            i+=1

    s = ' '.join(list)
    return s