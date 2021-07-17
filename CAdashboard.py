import streamlit as st
import numpy as np
import pandas as pd
import os

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#_____________________________________________________Functions___________________________________________________________________________________________
def getDF(file):
    return pd.read_csv(file)


def k_means_compile(df, col):


    chal = []

    for challenge in df[col]:
        if str(challenge) != "nan":
            sent = str(challenge).replace("\xa0", "")
            chal.append(sent)

    sentence_embeddings = model.encode(chal)
    optimal_k = silhouette(sentence_embeddings)

    return kmean_cluster(optimal_k, chal, sentence_embeddings)
    



def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances

def silhouette(corpus):
    from sklearn.metrics import silhouette_score
    sil = []
    kmax = 10


    for k in range(2, kmax+1):
      kmeans = KMeans(n_clusters = k).fit_predict(corpus)
      labels = kmeans
      sil.append(silhouette_score(corpus, labels, metric = 'euclidean'))
    print(sil)
    return sil.index(max(sil)) + 2

def kmean_cluster(num_clusters, corpus, embeddings):
    clustering_model = KMeans(n_clusters=num_clusters)
    
    y_pred = clustering_model.fit_predict(embeddings)
    distances = clustering_model.fit_transform(embeddings)
    cluster_assignment = clustering_model.labels_
    
    centroids = clustering_model.cluster_centers_
    
    
    clustered_sentences = [[] for i in range(num_clusters)]
    
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    
    reflection = []
    cluster2 = []
    for i, cluster in enumerate(clustered_sentences):
        for sent in cluster:
            reflection.append(sent)
            cluster2.append("cluster " + str(i))
    distances = np.min(distances, axis=1)

    df2 = {"student_reflection": reflection, "cluster #": cluster2, "distance": distances}
    df_output = pd.DataFrame().from_dict(df2)
    return df_output
    


#_____________________________________________________Web App Code___________________________________________________________________________________________
st.sidebar.title('Course Analytics Dashboard')

uploaded_file = st.sidebar.file_uploader("Upload student reflection")


compile = False

if uploaded_file is not None:
    df = getDF(uploaded_file)
    selected_col = st.sidebar.selectbox("Choose data column to cluster:", list(df.columns.values))
    compile = st.sidebar.button("Compile")

if compile and uploaded_file is not None:
    st.write("Outputting clusters for: \"" + selected_col + "\"." )
    with st.spinner('Compiling...'):
        st.dataframe(k_means_compile(df, selected_col),width=500, height=500)
        st.success('Done!')
    