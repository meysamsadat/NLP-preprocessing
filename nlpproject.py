import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


df_train = pd.read_csv(r'C:\Users\meysam-sadat\PycharmProjects\nlp_project_iran\Train.csv')
df_test = pd.read_csv(r'C:\Users\meysam-sadat\PycharmProjects\nlp_project_iran\Test.csv')
df_valid = pd.read_csv(r'C:\Users\meysam-sadat\PycharmProjects\nlp_project_iran\Valid.csv')
pd.set_option('display.max_rows',1000)

df_train.describe()
df_train['text'][1]




df_train['text'] = df_train['text'].str.replace('<br /><br />',' ')
#converting all words into lower case
df_train['text'] = df_train['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
#removing all the punctuation
df_train['text'] = df_train['text'].str.replace('[^\w\s]',' ')
#removing stop words
stop_words = stopwords.words('english')
df_train['text'] = df_train['text'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
#lemmatization
df_train['text'] = df_train['text'].apply(lambda x: ' '.join([ Word(word).lemmatize() for word in x.split()]))


sns.countplot(df_train['label'])
df_train.groupby('label').count()
df_train.dtypes

text = df_train.text.str.cat()
wordcloud = WordCloud(background_color='white').generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
df_possitive = df_train[df_train['label'] == 1]
df_negetive = df_train[df_train['label'] == 0]

possitive_str = df_possitive.text.str.cat()
negetive_str = df_negetive.text.str.cat()

wordcloud_positive = WordCloud(background_color='white').generate(possitive_str)
wordcloud_negetive = WordCloud(background_color='white').generate(negetive_str)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud_positive,interpolation='bilinear')
plt.axis('off')

plt.figure(figsize=(10,10))
plt.imshow(wordcloud_negetive,interpolation='bilinear')
plt.axis('off')

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.imshow(wordcloud_negetive,interpolation='bilinear')
ax1.axis('off')
ax1.set_title('Reviews with Negative points',fontsize=20)

cp = sns.color_palette()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

emptyline = []
for row in df_train['text']:
    vs = analyzer.polarity_scores(row)
    emptyline.append(vs)
df_sentiment = pd.DataFrame(emptyline)
df_sentiment = df_sentiment[['compound','neg','neu','pos']]
df_train_analysed = pd.concat([df_train.reset_index(drop=True),df_sentiment],axis=1)
df_train_analysed = df_train_analysed[df_train_analysed['compound'] > 0.9]
sns.countplot(df_train_analysed['label'])
