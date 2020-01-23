from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.mwe import MWETokenizer
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from dask import bag
from string import punctuation
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import nltk
import re
import pymongo


client = pymongo.MongoClient()
db = client.taprecommendations
db_reviews = db.reviews

global review_df
review_df = pd.DataFrame(db_reviews.find())


drop_list = [
    "Gordon Biersch Brewery Restaurant", 
    "Gordon Biersch Brewery & Restaurant - Midtown",
    "Gordon Biersch Brewery & Restaurant - Buckhead",
    "Fat Head's Brewery & Saloon",
    "Boston Beer Works - Fenway",
    "10 Barrel Brewing Co.",
    'Pizza Port Bressi Ranch', 
    'Pizza Port Carlsbad',
    'Pizza Port Ocean Beach',
    'Stone Brewing Co. - Richmond',
    'Surly Brewing Co. Beer Hall'
]

for brewery in drop_list: 
    beer_topics = review_df[~(review_df.brewery == brewery)].reset_index(drop=True)

review_df = review_df[review_df.review_count > 50].reset_index(drop=True)


def clean_reviews():
    # Replace punctuation and whitespace with empty string
    global review_df
    review_text = review_df['review'].map(
            lambda x: re.sub(
                f'[{re.escape(punctuation)}0-9\s]', ' ', x
            )
        ).str.strip(
        ).str.lower(
    )

    # Perform sentiment analysis assisted with Dask multiprocessing
    def compute_sentiment(x):
        sentiment = TextBlob(x['text']).sentiment
        x['polarity'] = sentiment.polarity
        x['subjectivity'] = sentiment.subjectivity
        return x

    b = bag.from_sequence(({'text': text} for text in review_text))
    mapped = b.map(compute_sentiment)
    sentiment_df = pd.DataFrame.from_records(mapped.compute())

    review_df[['polarity', 'subjectivity']] = sentiment_df[['polarity', 'subjectivity']]

    # Remove any excessively positive reviews, and any reviews that are
    # both excessively subjective and positive.  Cuts the corpus down by around
    # 5000 reviews out of 140,000
    review_df = review_df[
        (review_df.polarity < .70)
    ][
        (review_df.subjectivity <= 0.50) | (review_df.polarity <= .60)
    ].reset_index(drop=True)

    review_df['review_pp1'] = review_text

def tokenize():
    """
      Pull phrases from the corpus.  For example, to distinguish 
      between orange color and orange flavor, or to determine the strength
      of the flavor (light_citrus) or the carbonation (strong_carbonation)
    """
    global review_df
    from phrases import phrase_map, phrases, synonym_map
    phrase_tokenizer = MWETokenizer(phrases)
    stop_word_list = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()    

    def process(x):
        return [
            phrase_map.get(word, word) for word in phrase_tokenizer.tokenize(
                [
                    lemmatizer.lemmatize(synonym_map.get(word, word))
                    for word in x
                    if word not in stop_word_list
                ]
            )
        ]

    b = bag.from_sequence(review_df['review_pp1'].str.split())
    mapped = b.map(process)
    review_df['review_pp1'] = pd.Series(mapped.compute()).str.join(' ')

PHRASE_MODEL_LOC = './models/phrases.model'

def build_phrase_model():
    global review_df
    ### Trigram phrase model.  Fed back into the phrases for MWETokenizer
    bigram = Phrases(review_df.review_pp1, min_count=1, threshold=1)
    bigram_phraser = Phraser(bigram)
    trigram = Phrases(bigram_phraser[review_df.review_pp1])
    trigram.save(PHRASE_MODEL_LOC)
    return trigram

def get_phrase_model():
    return Phrases.load(PHRASE_MODEL_LOC)

def check_vocab(model, filter, score_threshold=300):
    ### Lookup a word and see if it's found in any common phrases.
    return [
        (str(phrase)[2:-1], score)
        for phrase, score in model.vocab.items()
        if str(phrase).find(filter) > -1 and score > score_threshold
    ]

def build_w2v_model():
    ### Useful to find synonyms for the phrases.synonym_map
    model = Word2Vec(sentences=review_df.review_pp1, workers=16)
    model.save('./models/w2v_reviews.model')
    return model


def model_topics():
    # Decompose the corpus with NMF
    global review_df
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(review_df.review_pp1.to_numpy())
    nmf = NMF(n_components=60)
    topics = nmf.fit_transform(X)
    names = vectorizer.get_feature_names()

    top_words = [
        sorted(
            zip(names, importance), key=lambda x: x[1], reverse=True
        )[:20]
        for importance in nmf.components_
    ]
    #Generate topic names from the top words
    topic_names = ['_'.join([word for word, _ in words][:3]) for words in top_words]

    #Concatenate the topics to the dataframe
    review_df = pd.concat(
        [
            review_df, pd.DataFrame(dict(zip(topic_names, topic)) for topic in topics)
        ], axis=1
    )
    review_df['score'] = review_df['score'].astype(float)
    #Aggregate each reviews topics into beer topics
    beer_topics = review_df.drop(
            columns=['rDev', 'review', 'review_count', 'polarity', 'subjectivity', 'review_pp1']
        ).groupby(
            ['brewery', 'beer', 'abv', 'city', 'style'],
            as_index=False
        ).agg({
            **{name: 'sum' for name in topic_names}, **{'score': 'mean'}
    })
    db.topics.drop()
    db.topics.insert_many(
        beer_topics.to_dict(orient='records')
    )

import sys
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'shell':
        import IPython
        IPython.embed()
        sys.exit()
    print('Cleaning Reviews...')
    clean_reviews()
    print('Tokenizing Reviews into Phrases...')
    tokenize()
    print('Building Phrase Model...')
    build_phrase_model()
    print('Building Word2Vec Model...')
    build_w2v_model()
    print('Performing NMF Topic Modeling...')
    model_topics()