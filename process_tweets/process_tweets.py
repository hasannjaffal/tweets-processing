import pandas as pd

df = pd.read_csv("training.1600000.processed.noemoticon.csv", header =None, encoding='ISO-8859-1')

df= df[[0,5]]
df.columns = ["sentiment", "tweet"]

df.to_csv('tweets.csv', index = False)