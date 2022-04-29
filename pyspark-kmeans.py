# Resources
# Pyspark k-means example: https://github.com/apache/spark/blob/master/examples/src/main/python/ml/kmeans_example.py
# EMR setup help(mentions how to get a notebook): https://towardsdatascience.com/data-science-at-scale-with-pyspark-on-amazon-emr-cluster-622a0f4534ed
# Example of k-means not using built in function: https://github.com/apache/spark/blob/master/examples/src/main/python/kmeans.py
# How to use matplotlib on EMR: https://aws.amazon.com/de/blogs/big-data/install-python-libraries-on-a-running-cluster-with-emr-notebooks/
# Vectorization: https://spark.apache.org/docs/latest/ml-features

import argparse
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import lower, regexp_replace, col

# sc.install_pypi_package("six==1.16.0")
# sc.install_pypi_package("matplotlib", "https://pypi.org/simple")
# import matplotlib.pyplot as plt

with SparkSession.builder.appName("Cluster research papers").getOrCreate() as spark:
    dataset = spark.read.format("csv") \
      .option("sep", ",") \
      .option("inferSchema", "true") \
      .option("header", "true") \
      .load("s3://ccassignment2/cleaned.csv")

    column_name = "abstract"

    #preprocessing: make all letters lowercase and remove special characters from the documents
    lowercase_df = dataset.select(lower(col("abstract"), lower(col("title")).alias("abstract"))
    no_punct_df = lowercase_df.select((regexp_replace(col("abstract"), "[^a-z0-9\\s]", "")).alias("abstract"))

    #preprocessing: tokenize
    tokenizer = Tokenizer(inputCol="abstract", outputCol="words")
    wordsData = tokenizer.transform(no_punct_df)

    #preprocessing: remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
    removed = remover.transform(wordsData)

    #apply TF
    hashingTF = HashingTF(inputCol="filteredWords", outputCol="rawFeatures", numFeatures=1000)
    featurizedData = hashingTF.transform(removed)

    #apply IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    result = idfModel.transform(featurizedData)

    #print to check if it works
    result.select("filteredWords", "features").show()

    # Figure out best K
#     silhouetteScores = []
#     for K in range(7,8):
#         #compute model with K
#         kmeans = KMeans().setK(K).setSeed(1)
#         model = kmeans.fit(result)
#         predictions = model.transform(result)

#         #evaluate K and add to list
#         evaluator = ClusteringEvaluator()
#         silhouette = evaluator.evaluate(predictions)
#         silhouetteScores.append(silhouette)

#     fig, ax = plt.subplots(1,1, figsize =(10,8))
#     ax.plot(range(7,8),silhouetteScores)
#     ax.set_xlabel('Number of Clusters')
#     ax.set_ylabel('Silhouette Score')
#     plt.show()

    #Create final kmeans model - assignment says to create 8 clusters
    kmeans = KMeans().setK(8).setSeed(1)
    model = kmeans.fit(result)
    predictions = model.transform(result)
    predictions.show()

    #Print result
#     centers = model.clusterCenters()
#     print("Cluster Centers: ")
#     for center in centers:
#         print(center)
#         print(type(center))


    spark.stop()
