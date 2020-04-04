# Add this for dumb relative imports
import sys
sys.path.append(".")

from preprocessing import step2_feature_engineering as feature_eng

from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
import utils

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def driver():
    data_df, features = feature_eng.preprocess_features()
    data_df.cache()
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])

    # create the trainer and set its parameters
    nb = NaiveBayes(labelCol='TARGET', featuresCol='OCCUPATION_TYPE',
                    smoothing=1.0, modelType="multinomial")

    # Chain vector_assembler and forest in a Pipeline

    # Train model.  This also runs the indexers.
    model = nb.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    predictions.select('TARGET', 'rawPrediction', 'prediction','probability').show(20)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                              labelCol="TARGET", metricName='areaUnderROC')

    areaUnderRoc = evaluator.evaluate(predictions)
    print(f"Area Under ROC = {areaUnderRoc}")


if __name__ == '__main__':
    driver()
