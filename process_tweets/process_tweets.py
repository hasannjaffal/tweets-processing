import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("training.1600000.processed.noemoticon.csv", header =None, encoding='ISO-8859-1')

df= df[[0,5]]
df.columns = ["sentiment", "tweet"]

df.to_csv('tweets.csv', index = False)

df = pd.read_csv('tweets.csv')
texts = df['tweet'].values 
labels = df['sentiment'].values 

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(texts)
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=50)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100)) 
