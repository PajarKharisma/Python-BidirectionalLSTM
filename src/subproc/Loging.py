def writeLog(logPath, logName, logFile):
    file = open(logPath+logName, "w", encoding='utf-8')
    file.write(logFile)
    file.close()

def summaryLog(method, numEpochs, numAttributes, numNeurons, sg):
    word2vec = ''
    if sg == 1:
        word2vec = 'Skipgram'
    else:
        word2vec = 'CBOW'
    result = 'Arsitektur Word2Vec : ' + word2vec +'\n'
    result += 'Metode : ' + method +'\n'
    result += 'Jumlah Epoch : ' + str(numEpochs) + '\n'
    result += 'Jumlah Atribut : ' + str(numAttributes) + '\n'
    result += 'Jumlah Neuron : ' + str(numNeurons) + '\n'
    return result