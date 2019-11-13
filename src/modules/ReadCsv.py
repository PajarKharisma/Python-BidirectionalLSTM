import pandas as pd

def csv2array(path):
    posData = []
    negData = []
    df = pd.read_csv(path, error_bad_lines=False)
    for index, (label, tweet) in enumerate(zip(df['label'], df['tweet'])):
        if label.lower() == 'non_hs':
            posData.append(tweet)
        else:
            negData.append(tweet)

    return posData, negData

# def csv2array(path):
#     posData = []
#     negData = []
#     with open(path) as csvDataFile:
#         csvReader = csv.reader(csvDataFile)
#         for row in csvReader:
#             kelas = row[0]
#             if kelas.lower() == 'non_hs':
#                 posData.append(row[1])
#             else:
#                 negData.append(row[1])

#     return posData, negData