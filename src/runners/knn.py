# Add this for dumb relative imports
import sys
sys.path.append(".")


import importlib
from src import utils
from src.preprocessing import step2_feature_engineering as feature_eng

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
    data_df, features = feature_eng.preprocess_features(takeSample=takeSample)

    # Assemble all features in a vector using Vector Assembler
    # map it to new column named features
    vector_assembler = VectorAssembler(
        inputCols=features, outputCol="features")

    data_df = vector_assembler.transform(data_df) # data_df => SK_ID_CURR, ..., .transform(vector_assembler) =>  "features": Vector<float>
    print("***\n\n\nVector Assembler Made")
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])
    print("***\n\n\nData Split KNN")
    train_vector = trainingData.select("SK_ID_CURR","features", "TARGET").withColumnRenamed("SK_ID_CURR","train_SK_ID_CURR").withColumnRenamed("features","train_features")
    test_vector = testData.select("SK_ID_CURR","features").withColumnRenamed("SK_ID_CURR","test_SK_ID_CURR").withColumnRenamed("features","test_features")
    train_vector.show(10)
    test_vector.show(10)
    print("***\n\n\nVectors Renamed")
    cross_join = test_vector.crossJoin(train_vector)
    cross_join.show(20)
    print("***\n\n\nCross Joined")

    def distance(row):
        print(row)
        train_features = row['train_features']
        test_features = row['test_features']
        train_SK_ID_CURR = row['train_SK_ID_CURR']
        test_SK_ID_CURR = row['test_SK_ID_CURR']
        target = row['TARGET']
        row = Row(
            train_features=train_features,
            test_features=test_features,
            train_SK_ID_CURR=train_SK_ID_CURR,
            test_SK_ID_CURR=test_SK_ID_CURR,
            TARGET=target,
            distance=float(norm(train_features[1]-test_features[1]))
        )
        return row

    print("***\n\n\nDistance Function Defined")
    spark = utils.init_spark()
    cross_join = cross_join.rdd.map(distance).toDF()
    print(cross_join.take(10))
    cross_join.show()
    cross_join = cross_join.groupBy('test_SK_ID_CURR').agg(count("TARGET"), sum("TARGET"))
    cross_join.show()
    cross_join = cross_join.withColumn('prediction', cross_join["sum(TARGET)"] / cross_join["count(TARGET)"])
    cross_join.show()



if __name__ == '__main__':
    driver()
