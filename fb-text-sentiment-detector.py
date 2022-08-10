from gettext import translation
import heapq
from facebook_scraper import get_posts
from textblob import TextBlob
import csv
import emoji
import pandas as pd
import matplotlib.pyplot as plot
import re
import numpy as np
from translate import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize

gs = Translator(from_lang='es', to_lang='en')

#----------------------------- EXTRACCION DE DATOS DE COMENTARIOS EN UN .CSV---------------------------->
with open('data.csv', 'w', newline="", encoding="utf-8")as file:
    writer = csv.writer(file)

    # CABECERA
    writer.writerow(['Date', 'Day', 'Post_id', 'Post_link', 'Commenter_id',
                    'Commenter', 'Comment_link', 'Comment_text'])

    for post in get_posts(group='radiounotunja', pages=1, cookies="cookies.txt", options={'comments': True}):
        text_comments = post['comments_full']
        for comment in text_comments:
            # ELIMINAMOS COMENTARIOS VACIOS Y STICKES
            if(comment['comment_text'] != comment['commenter_name']):
                text = comment['comment_text']
                allchars = [str for str in text]
                emoji_list = emoji.EMOJI_DATA
                # ELIMINAMOS EMOJIS
                clean_text = ' '.join(
                    [str for str in text.split() if not any(i in str for i in emoji_list)])
                if(not pd.isnull(clean_text) and clean_text != ''):
                    data = [post['time'].strftime("%b %d %Y"), post['time'].strftime("%b %d"), post['post_id'], post['post_url'], comment['comment_id'], comment['commenter_name'],
                            comment['comment_url'], clean_text]
                    writer.writerow(data)

data = pd.read_csv('data.csv')
print("=================================DATAFRAME===================================")
print(data)

# -------------------------------------ANALISIS DE POPULARIDAD---------------------------------------->
popularity_list = []

for comment in data['Comment_text']:
    if (not pd.isnull(comment) and comment != 'nan'):
        translation = gs.translate(comment)
        analysis = TextBlob(translation)
        analysis = analysis.sentiment
        popularity = analysis.polarity
        popularity_list.append(popularity)

#---------------------------------------FIGURA POPULARIDAD--------------------------------------->
plot.figure(figsize=(9, 4))
plot.scatter(data['Day'], popularity_list)
plot.title('Post popularity')
plot.xlabel('Comments')
plot.ylabel('Sentiment')
plot.show()

# -------------------------------------CANTIDAD COMENTARIOS-------------------------------------------->
print("=================================CANTIDAD COMENTARIOS===================================")
print('Numero de comentarios: '+str(data.shape[0]))

# =====================DISTRIBUCION TEMPORAL DE LOS COMENTARIOS POR FECHA==============================>
df_temp = data
df_temp = df_temp.groupby(df_temp['Day']).size().reset_index(name='Comments')


plot.figure(figsize=(9, 4))
plot.plot(df_temp['Day'], df_temp['Comments'])
plot.title('Comments Quantity per Date')
plot.xlabel('Date')
plot.ylabel('Comments')
plot.show()

# =========================LIMPIEZA Y TOKENIZACION DE LOS DATOS=======================================>
corpus = []

for text in data['Comment_text']:
    corpus.append(text)

# Limpieza datos
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()
    # Elimina los signos de puntuación y espacios.
    corpus[i] = re.sub(r'[^\w\s]', '', corpus[i])
    # ^\w reconoce caracteres que no son alfanuméricos y \s reconoce espacios
    # Estos caracteres los substituye por '', es decir, los elimina.
print("==============================CORPUS=====================================")
print(corpus)

# ============================ANALISIS DE SENTIMIENTOS EN PALABRAS FRECUENTES===================================>


def sentiment_analisis(words):
    print("=================================SENTIMIENTOS PALABRAS FRECUENTES===================================")
    negative_words = pd.read_table('negative-words.txt')
    positive_words = pd.read_table('positive-words.txt')
    positive = 0
    negative = 0
    neutral = 0

    for word in words:
        translation = gs.translate(word)
        if translation.lower() in np.array(negative_words):
            negative = negative + 1
        else:
            if translation.lower() in np.array(positive_words):
                positive = positive + 1
            else:
                neutral = neutral + 1

    sentiments = [negative, positive, neutral]
    tags = ["Negative", "Positive", "Neutral"]
    plot.pie(sentiments, labels=tags, autopct="%0.1f %%")
    plot.title('Sentiments frec words')
    plot.show()

# =================================PALABRAS FRECUENTES===================================>
wordfreg = {}
for sentence in corpus:
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreg.keys():
            wordfreg[token] = 1
        else:
            wordfreg[token] += 1
most_freg = heapq.nlargest(200, wordfreg, key=wordfreg.get)
print("=================================PALABRAS FRECUENTES===================================")
print(most_freg)
sentiment_analisis(most_freg)


analyzer = SentimentIntensityAnalyzer()
 
pos_count = 0
pos_correct = 0
 
with open("positive-words.txt","r") as f:
  for line in f.read().split('\n'):
    vs = analyzer.polarity_scores(line)
    if vs['compound'] > 0:
      pos_correct += 1
    pos_count +=1
 
neg_count = 0
neg_correct = 0
 
with open("negative-words.txt","r") as f:
  for line in f.read().split('\n'):
    vs = analyzer.polarity_scores(line)
    if vs['compound'] <= 0:
      neg_correct += 1
    neg_count +=1
print("=================================PRUEBA VADER===================================")
print("Precisión Positiva = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Precisión Negativa = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

pos_count = 0
pos_correct = 0
 
with open("positive-words.txt","r") as f:
  for line in f.read().split('\n'):
    analysis = TextBlob(line)
    # print(line)
    try:
      if analysis.sentiment.polarity > 0:
        pos_correct += 1
      pos_count +=1
    except:
    #Mostramos este mensaje en caso de que se presente algún problema
      print ("El elemento no está presente")
 
neg_count = 0
neg_correct = 0
 
with open("negative-words.txt","r") as f:
  for line in f.read().split('\n'):
    analysis = TextBlob(line)
    # print(line)
    try:
      if analysis.sentiment.polarity <= 0:
        neg_correct += 1
      neg_count +=1
    except:
      print('el elemento no esta presente')
 
print("=================================PRUEBA TextBlob===================================")
print("Precisión positiva = {}% via {} ejemplos".format(pos_correct/pos_count*100.0, pos_count))
print("Precisión negativa = {}% via {} ejemplos".format(neg_correct/neg_count*100.0, neg_count))


print("=================================ANALISIS DE SENTIMIENTOS POR COMENTARIO===================================")

for comment in corpus:
    translation = gs.translate(comment)
    scores = analyzer.polarity_scores(translation)
    print(comment)
    print(translation)
    for key in scores:
        print(key, ': ', scores[key])

print("================================INFORMACION DEL SISTEMA Y MODULOS NECESARIOS=======================================")
# from sinfo import sinfo
# sinfo()
