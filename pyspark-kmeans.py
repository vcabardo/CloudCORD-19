
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from pyspark.sql.functions import lower, regexp_replace, col, collect_list

with SparkSession.builder.appName("Cluster research papers").getOrCreate() as spark:
    dataset = spark.read.format("csv") \
      .option("sep", ",") \
      .option("inferSchema", "true") \
      .option("header", "true") \
      .load("s3://ccassignment2/cleaned.csv") #IMPORTANT CHANGE THIS TO THE LOCATION OF YOUR CSV FILE

    column_name = "abstract"

    #preprocessing: make all letters lowercase and remove special characters from the documents
    lowercase_df = dataset.select(lower(col("abstract")).alias("abstract"))
    no_punct_df = lowercase_df.select((regexp_replace(col("abstract"), "[^a-z0-9\\s]", "")).alias("abstract"))

    #preprocessing: tokenize
    tokenizer = Tokenizer(inputCol="abstract", outputCol="words")
    wordsData = tokenizer.transform(no_punct_df)

    #preprocessing: remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
    removed = remover.transform(wordsData)

    cv = CountVectorizer(inputCol="filteredWords", outputCol="rawFeatures", vocabSize=400)
    cvmodel = cv.fit(removed)
    featurizedData = cvmodel.transform(removed)

    #apply IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    result = idfModel.transform(featurizedData)

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

    predictions.groupBy('prediction').count().show()
    topWords = {}
    for cluster in range(8):
        print("Cluster: ", cluster)
        counts = predictions.filter(col('prediction') == cluster).select("features").collect()
        frequencies = dict(zip(cvmodel.vocabulary, counts[0]['features'].values))

        i = 0
        from operator import itemgetter
        for key, value in sorted(frequencies.items(), key=itemgetter(1), reverse=True):
            print(key, value, )

            if i == 0:
                topWords[key] = value

            i = i + 1
            if i >= 10:
                break

    print(topWords)
    print(sorted(topWords, key=topWords.get, reverse=True)[:3])

    spark.stop()
