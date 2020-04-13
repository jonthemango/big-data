# Add this for dumb relative imports
import sys
sys.path.append(".")


import importlib
from src import utils
from src.preprocessing import step2_feature_engineering as feature_eng
from src.evaluators import multiple_evaluator
from src.preprocessing import step3_undersampling as sampling

from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

import matplotlib.pyplot as plt


def driver(takeSample=False):

    # Pre-process features
    data_df, features = feature_eng.preprocess_features2(takeSample=takeSample)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.8, 0.2])
    trainingData = sampling.undersample(trainingData,class_ratio=0.5)


    # Assemble all features in a vector using Vector Assembler
    # map it to new column named features
    vector_assembler = VectorAssembler(
        inputCols=features, outputCol="features")

    # Train a RandomForest model.
    rf = RandomForestClassifier(
        labelCol="TARGET", featuresCol="features", numTrees=100, maxDepth=10)

    # Chain vector_assembler and forest in a Pipeline
    pipeline = Pipeline(stages=[vector_assembler, rf])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    pred = predictions.select('SK_ID_CURR','TARGET', 'rawPrediction','prediction','probability')

    pred.show()

    return multiple_evaluator(predictions)




if __name__ == '__main__':
    driver()
