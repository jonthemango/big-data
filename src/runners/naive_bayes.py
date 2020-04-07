# Add this for dumb relative imports
import sys
sys.path.append(".")

from src.preprocessing import step2_feature_engineering as feature_eng
from src.evaluators import multiple_evaluator
from src.preprocessing import step3_undersampling as sampling

from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
from src import utils

from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator


def driver(takeSample=False):
    data_df, features = feature_eng.preprocess_features(takeSample=takeSample)
    data_df.cache()
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])
    trainingData = sampling.undersample(trainingData,class_ratio=0.6)

    # create the trainer and set its parameters
    nb = NaiveBayes(labelCol='TARGET', featuresCol='OCCUPATION_TYPE',
                    smoothing=1.0, modelType="multinomial")


    # Train model.  This also runs the indexers.
    model = nb.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    predictions.select('TARGET', 'rawPrediction', 'prediction','probability').show(20)

    return multiple_evaluator(predictions)


if __name__ == '__main__':
    driver()
