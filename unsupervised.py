import pymongo
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


client = pymongo.MongoClient()
db = client.taprecommendations
beer_topics = pd.DataFrame.from_records(db.topics.find())

topic_df = beer_topics.drop(columns=['brewery', 'beer', '_id', 'abv', 'city', 'style'])

clus1 = KMeans(n_clusters=20, n_init=3, random_state=1).fit(topic_df)
clus2 = KMeans(n_clusters=20, n_init=3, random_state=20).fit(topic_df)
clus3 = KMeans(n_clusters=20, n_init=3, random_state=30).fit(topic_df)
clus4 = KMeans(n_clusters=20, n_init=3, random_state=40).fit(topic_df)
mix_clus = KMeans(n_clusters=10, n_init=3, random_state=50).fit(topic_df)
beer_topics['cluster1'] = clus1.predict(topic_df)
beer_topics['cluster2'] = clus2.predict(topic_df)
beer_topics['cluster3'] = clus3.predict(topic_df)
beer_topics['cluster4'] = clus4.predict(topic_df)
beer_topics['mix_cluster'] = mix_clus.predict(topic_df)

db.topic_clusters.drop()
db.topic_clusters.insert_many(beer_topics.to_dict(orient='records'))