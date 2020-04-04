# Add this for dumb relative imports
import sys
sys.path.append(".")

from src.preprocessing import step2_feature_engineering as feature_eng

from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
from src import utils

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LinearSVC


def driver():
    data_df, features = feature_eng.preprocess_features()
    data_df.cache()
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])

    # Assemble all features in a vector using Vector Assembler
    # map it to new column named features
    vector_assembler = VectorAssembler(
        inputCols=features, outputCol="features")

    lsvc = LinearSVC(labelCol='TARGET', maxIter=10, regParam=0.1)

    # Chain vector_assembler and lsvc in a Pipeline
    pipeline = Pipeline(stages=[vector_assembler, lsvc])

    # Fit the model
    lsvcModel = pipeline.fit(trainingData)

    predictions = lsvcModel.transform(testData)
    predictions.select('TARGET', 'rawPrediction', 'prediction').show(150)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                              labelCol="TARGET", metricName='areaUnderROC')

    areaUnderRoc = evaluator.evaluate(predictions)
    print(f"Area Under ROC = {areaUnderRoc}")


if __name__ == '__main__':
    driver()
