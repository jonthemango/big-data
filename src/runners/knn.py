# Add this for dumb relative imports
import sys
sys.path.append(".")


import importlib
from src import utils
from src.preprocessing import step2_feature_engineering as feature_eng
from src.evaluators import multiple_evaluator

from pyspark.rdd import RDD
from pyspark.sql import DataFrame, SparkSession, Row
from pyspark.sql.functions import udf,col,count,sum
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

import json
import copy as cp

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from numpy.linalg import norm
from numpy.lib.scimath import sqrt

def driver(takeSample=False):
    print("***\n\n\nStarting KNN")

    # Pre-process features
    data_df, features = feature_eng.preprocess_features2(takeSample=takeSample)

    # Assemble all features in a vector using Vector Assembler
    # map it to new column named features
    vector_assembler = VectorAssembler(inputCols=features, outputCol="features")

    data_df = vector_assembler.transform(data_df) # data_df => SK_ID_CURR, ..., .transform(vector_assembler) =>  "features": Vector<float>

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.8, 0.2])

    # Cross Join train_vector and test_vector
    train_vector = trainingData.select("SK_ID_CURR","features", "TARGET").withColumnRenamed("SK_ID_CURR","train_SK_ID_CURR").withColumnRenamed("features","train_features")
    test_vector = testData.select("SK_ID_CURR","features").withColumnRenamed("SK_ID_CURR","test_SK_ID_CURR").withColumnRenamed("features","test_features")
    cross_join = test_vector.crossJoin(train_vector)
    cross_join.show(20)

    def distance(row):
        '''
        Euclidean Distance Function
        '''
        train_features = row['train_features']
        test_features = row['test_features']
        train_SK_ID_CURR = row['train_SK_ID_CURR']
        test_SK_ID_CURR = row['test_SK_ID_CURR']
        target = row['TARGET']

        # euclid distance
        s = 0
        for i in range(len(train_features)):
            s += (train_features[i]-test_features[i])**2
        distance = float(sqrt(s))

        row = Row(
            train_features=train_features,
            test_features=test_features,
            train_SK_ID_CURR=train_SK_ID_CURR,
            test_SK_ID_CURR=test_SK_ID_CURR,
            TARGET=target,
            distance=distance
        )
        return row

    # Convert to DF
    cross_join = cross_join.rdd.map(distance).toDF()
    cross_join.show()

    # Group by test_SK_ID_CURR
    grouped = cross_join.rdd.map(lambda x: (x.test_SK_ID_CURR, x)).groupByKey()

    def choose_k_and_predict(row):
        '''
        Choose the 5 smallest using heapq and return distances and prediction based on those
        '''
        k = 5
        print(row)
        import heapq
        closest_k = heapq.nsmallest(k, row[1], key=lambda ele: ele.distance)
        distances = []
        target_sum = 0
        for ele in closest_k:
            distances.append((ele.train_SK_ID_CURR, ele.TARGET, ele.distance))
            target_sum += ele.TARGET
        if target_sum/k < 0.5:
            prediction = 0
        else:
            prediction = 1
        return (row[0], prediction, row[1].TARGET, distances)

    # Choose k smallest and predict
    prediction_df = grouped.map(choose_k_and_predict).toDF().withColumnRenamed('_1', 'train_SK_ID_CURR').withColumnRenamed('_2', 'rawPrediction').withColumnRenamed('_3', "TARGET").withColumnRenamed('_4', 'features')
    prediction_df = prediction_df.withColumn('prediction', prediction_df['rawPrediction'])
    prediction_df.show(10)
    return multiple_evaluator(prediction_df)



if __name__ == '__main__':
    driver()
