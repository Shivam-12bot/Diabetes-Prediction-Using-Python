Text Analytics on online content.


# Objective

The objective of this assignment is to extract textual data articles from the given URL and perform text analysis to compute variables that are explained below. 

# Data Extraction

Input.xlsx

For each of the articles, given in the input.xlsx file, extract the article text and save the extracted article in a text file with URL_ID as its file name.

While extracting text, please make sure your program extracts only the article title and the article text. It should not extract the website header, footer, or anything other than the article text. 

import pandas as pd
df = pd.read_excel('input.xlsx')
df.head(10)

We would derive the text content from each URL
from newspaper import Article
import nltk
url = 'https://insights.blackcoffer.com/ai-and-its-impact-on-the-fashion-industry/'
article = Article(url, language="en")
article.download() 
article.parse() 
article.nlp() 
print("Article Title:") 
print(article.title) #prints the title of the article
print("\n") 
print("Article Text:") 
print(article.text) #prints the entire text of the article
print("\n") 
print("Article Summary:") 
print(article.summary) #prints the summary of the article
print("\n") 
print("Article Keywords:")
print(article.keywords) #prints the keywords of the article
def get_text(url):
    
    url1 = url
    article = Article(url1, language="en")
    
    article.download() 
    article.parse() 
    article.nlp()
    
    return article.text
for i in range(0,len(df)):
    df['URL_ID'][i] = get_text(df['URL'][i])
df.rename({'URL_ID':'Text'},axis=1,inplace=True)
df.head(10)

Let's do some Text preprocessing and cleaning
import re
from nltk.corpus import stopwords
def transform(text):

    review = re.sub('[^a-zA-Z0-9]', ' ',text)  # except small and capital letters and numeric remove everythong.
    review = review.lower()                    # lower it.
    review = review.split()
    
    review = [word for word in review if not word in stopwords.words('english')]   # remove stopwords.
    review = ' '.join(review)
    return review


df['Transform_Text'] = df['Text'].apply(transform)
# Data Analysis

For each of the extracted texts from the article, perform textual analysis and compute variables.

I am looking for these variables in the analysis document:

POSITIVE SCORE

NEGATIVE SCORE

POLARITY SCORE

SUBJECTIVITY SCORE

AVG SENTENCE LENGTH

PERCENTAGE OF COMPLEX WORDS

FOG INDEX

AVG NUMBER OF WORDS PER SENTENCE

COMPLEX WORD COUNT

WORD COUNT

SYLLABLE PER WORD

PERSONAL PRONOUNS

AVG WORD LENGTH

# word count in each text row.

df['word_counts'] = df['Transform_Text'].apply(lambda x: len(str(x).split()))    

import nltk
len(nltk.sent_tokenize(df['Text'][0]))  # checking length function 
import numpy as np
df['average number of words per sentence'] = np.nan

for i in range(0,len(df)):
    
    df['average number of words per sentence'][i] = df['word_counts'][i]/len(nltk.sent_tokenize(df['Text'][i]))
df.head(10)

# Average Word Length


Average Word Length is calculated by the formula:
    
( Sum of the total number of characters in each word ) / ( Total number of words )


def char_count(x):
    s = x.split()
    x = ''.join(s)
    return len(x)      # counting the total number of characters in each text data.
df['chara_count'] = df['Transform_Text'].apply(lambda x: char_count(x))

to check for stopwords in each text.

from nltk.corpus import stopwords

df['stopwords'] = df['Text'].apply(lambda x: [t for t in x.split() if t  in stopwords.words('english')])

df['average word length'] = np.nan

for i in range(0,len(df)):
    
    df['average word length'][i] = df['chara_count'][i]/df['word_counts'][i]
df.head()

# SYLLABLE COUNT

We count the number of Syllables in each word of the text by counting the vowels present in each word.
h = df.head()

def syllable_count(x):
    v = []
    d = {}
    for i in x:
        if i in "aeiou":
            v.append(i)
            d[i] = d.get(i,0)+1     # checking purpose
            
    k = []
    for i in d:
        k.append(d[i])
    print(d)
    print(v)  
    print(k)
    print(np.sum(k))
        
    
g = 'bore i am gone to london in england britian uk'

syllable_count(g)

def syllable_count(x):
    v = []
    d = {}
    for i in x:
        if i in "aeiou":
            v.append(i)
            d[i] = d.get(i,0)+1
            
    k = []
    for i in d:
        k.append(d[i])

    return np.sum(k)

g = h['Transform_Text'][1]

syllable_count(g)

df['syllable count'] = df['Transform_Text'].apply(lambda x: syllable_count(x))
df.head()

# COMPLEX Word Count

Complex words are words in the text that contain more than two Syllables.

from collections import  Counter

def complex_word_count(x):
    
    syllable = 'aeiou'
    
    t = x.split()
    
    v = []
    
    for i in t:
        words = i.split()
        c=Counter()
        
        for word in words:
            c.update(set(word))

        n = 0
        for a in c.most_common():
            if a[0] in syllable:
                if a[1] >= 2:
                    n += 1
                
        m = 0
        p = []
        for a in c.most_common():
            if a[0] in syllable:
                p.append(a[0])
        if len(p) >= 2:
            m += 1
        
        if n >= 1 or m >= 1:
            v.append(i)
            
    return len(v) 

g = h['Transform_Text'][1]

complex_word_count(g)

df['complex_count'] = np.nan

df['complex_count'] = df['Transform_Text'].apply(lambda x: complex_word_count(x))
df.head()

# Analysis of Readability

Analysis of Readability is calculated using the Gunning Fox index formula described below.

Average Sentence Length      =  the number of words / the number of sentences

Percentage of Complex words  =  the number of complex words / the number of words 

Fog Index                    =  0.4 * (Average Sentence Length + Percentage of Complex words)


df['sentence length'] = np.nan
df['Average Sentence Length'] = np.nan
df['Percentage of Complex words'] = np.nan
df['Fog Index'] = np.nan


for i in range(0,len(df)):
    
    df['sentence length'][i]  =   len(nltk.sent_tokenize(df['Text'][i]))
    df['Average Sentence Length'][i] = df['word_counts'][i]/df['sentence length'][i]
    df['Percentage of Complex words'][i] = df['complex_count'][i]/df['word_counts'][i] 
    df['Fog Index'][i] = 0.4 * (df['Average Sentence Length'][i] + df['Percentage of Complex words'][i])
df.head()

# SENTIMENT ANALYSIS

Sentimental analysis is the process of determining whether a piece of writing is positive, negative or neutral.

The Master Dictionary (found here) is used for creating a dictionary of Positive and Negative words. We add only those words in the dictionary if they are not found in the Stop Words Lists. Use this url if above does not work https://sraf.nd.edu/textual-analysis/resources/ 


sentiment = pd.read_csv('sentiment dict.csv')
dfs = sentiment[['Word','Negative','Positive']]
dfs
f = ['ZYGOTIC','BAD','DONE','EXCELLENT','WORSE']

negative = 0
positive = 0

for i in dfs['Word']:
    if i in f:
        if dfs[dfs['Word']==i].Negative.any() == True:
            negative += 1
        if dfs[dfs['Word']==i].Positive.any() == True:                # CHECKING
            positive += 1
            
print(negative),
print(positive)

We need to lower the word column in dfs to be used for sentiment score for the text data.

dfs = dfs.dropna()
dfs.isnull().sum()
w = 'the good man'
w.split()
dfs['word_lower'] = np.nan
import warnings
warnings.filterwarnings('ignore')

for i in range(len(dfs)):
        dfs['word_lower'][i] = dfs['Word'][i].lower()
for i in range(50742,len(dfs)):
        dfs['word_lower'][i] = dfs['Word'][i].lower()
dfs['word_lower'].dtype
dfs.head()

# Positive Score

Positive Score: This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.


# Calculate the positive score for text.

def positive(x):
    
    s = x.split()
    
    positive = 0
    
    for i in dfs['word_lower']:
        if i in s:
            if dfs[dfs['word_lower']==i].Positive.any() == True:
                positive += 1
            
    return positive
df['positive_score'] = np.nan

for i in range(len(df)):
    df['positive_score'][i] = positive(df['Transform_Text'][i])
df.head()

def positive_word(x):
    
    s = x.split()
    
    positive_word = []
    
    for i in dfs['word_lower']:
        if i in s:
            if dfs[dfs['word_lower']==i].Positive.any() == True:   # checking which words are positive
                positive_word.append(i)
            
    print(positive_word)
df['positive_word'] = np.nan

for i in range(1):
    df['positive_word'][i] = positive_word(df['Transform_Text'][i])
df.drop('positive_word',axis=1,inplace=True)

# NEGATIVE Score

Negative Score: This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. We multiply the score with -1 so that the score is a positive number.

def negative_score(x):
    
    s = x.split()
    
    negative = 0
    
    for i in dfs['word_lower']:
        if i in s:
            if dfs[dfs['word_lower']==i].Negative.any() == True:
                negative += 1
            
    return negative
df['negative_score'] = np.nan

for i in range(len(df)):
    df['negative_score'][i] = negative_score(df['Transform_Text'][i])

df.head()

# Polarity Score


Polarity Score: This is the score that determines if a given text is positive or negative in nature.

It is calculated by using the formula: 

Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)

Range is from -1 to +1

df['Polarity Score'] = np.nan

for i in range(len(df)):
    df['Polarity Score'][i] = (df['positive_score'][i]-df['negative_score'][i])/ ((df['positive_score'][i] + df['negative_score'][i]) + 0.000001)
df.head()

# SUBJECTIVITY SCORE

Subjectivity Score: This is the score that determines if a given text is objective or subjective. 

It is calculated by using the formula: 

Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)

Range is from 0 to +1

from textblob import TextBlob
blob = TextBlob(df['Transform_Text'][1])
blob.sentiment
TextBlob(df['Transform_Text'][1]).sentiment[1]
df['subjectivity'] = np.nan

for i in range(len(df)):
    df['subjectivity'][i] = TextBlob(df['Transform_Text'][i]).sentiment[1]
df.head()

# PERSONAL PRONOUNS 

To calculate Personal Pronouns mentioned in the text, we use regex to find the counts of the words - “I,” “we,” “my,” “ours,” and “us”. 
import spacy
nlp = spacy.load('en_core_web_sm')
x = 'he is the my father'
y = nlp(x)

for noun in y.noun_chunks:
    print(noun)
doc = nlp('he is the my father')
for token in doc:
    if token.pos_ == 'PRON':
        print(token)
df['PERSONAL PRONOUNS'] = np.nan
doc = nlp(df['Text'][1])
tok = []
for token in doc:
    if token.pos_ == 'PRON':
        tok.append(token)
        
tok
df['PERSONAL PRONOUNS'][1] = tok
df.head()

df['PERSONAL PRONOUNS'] = np.nan

for i in range(len(df)):
    doc = nlp(df['Text'][i])
    tok = []
    for token in doc:
        if token.pos_ == 'PRON':
            tok.append(token)
        
    df['PERSONAL PRONOUNS'][i] = tok
df.head()
df['PERSONAL PRONOUNS'][2]

submit = df[['URL','positive_score','negative_score','Polarity Score','subjectivity','Average Sentence Length','Percentage of Complex words',
            'Fog Index','average number of words per sentence','complex_count','word_counts','syllable count','PERSONAL PRONOUNS','average word length']]
submit.head()

file_name = "Output Data Structure.xlsx"

submit.to_excel(file_name)
