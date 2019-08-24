import csv

def csv2array(path):
    posData = []
    negData = []
    with open(path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            kelas = row[0]
            if kelas.lower() == 'non_hs':
                posData.append(row[1])
            else:
                negData.append(row[1])

    return posData, negData