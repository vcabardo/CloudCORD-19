# Resources
# Pyspark k-means example: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/kmeans_example.py
# EMR setup help(mentions how to get a notebook): https://towardsdatascience.com/data-science-at-scale-with-pyspark-on-amazon-emr-cluster-622a0f4534ed
# Example of k-means not using built in function: https://github.com/apache/spark/blob/master/examples/src/main/python/kmeans.py

import argparse
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


with SparkSession.builder.appName("Cluster research papers").getOrCreate() as spark:
    dataset = spark.read.format("libsvm").csv(data_source)

    # Figure out best K
    silhouetteScores = []
    for K in range(2,11):
        #compute model with K
        kmeans = KMeans().setK(K).setSeed(1)
        model = kmeans.fit(dataset)
        predictions = model.transform(dataset)

        #evaluate K and add to list
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        silhouetteScores.append(silhouette)

    fig, ax = plt.subplots(1,1, figsize =(10,8))
    ax.plot(range(2,11),silhouetteScores)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')

    #Create final kmeans model - assignment says to create 8 clusters
    kmeans = KMeans().setK(8).setSeed(1)
    model = kmeans.fit(dataset)
    predictions = model.transform(dataset)

    #Print result
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    spark.stop()