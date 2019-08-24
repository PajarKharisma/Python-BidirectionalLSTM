import nltk
from nltk.corpus import stopwords
import re

REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\@)|(\\\)|(\.)")
REPLACE_NO_SPACE = re.compile("(\#)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\@)|(\/)|(\\\)")

def getResult(reviews):
    default_stop_words = nltk.corpus.stopwords.words('indonesian-sentimen')
    stopwords = set(default_stop_words)

    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [RemoveStopWords(line,stopwords) for line in reviews]
    
    return reviews

def RemoveStopWords(line, stopwords):
    words = []
    for word in line.split(" "):
        word = word.strip()
        if word not in stopwords and word != "" and word != "&":
            words.append(word)

    return " ".join(words)