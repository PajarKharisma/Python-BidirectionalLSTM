import pandas as pd
df = pd.read_csv("input.csv", error_bad_lines=False)

posData = []
negData = []

for index, (label, tweet) in enumerate(zip(df['label'], df['tweet'])):
    # print(label, ' | ', tweet)
    if label.lower() == 'non_hs':
        posData.append(tweet)
    else:
        negData.append(tweet)

for pos in posData:
    print(pos)