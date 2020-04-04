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
    '''
    OCUPATION_TYPE: 1 | 5 | 9
    1, 1,0
    5, 1,0
    9, 1,0
    '''
    # create the trainer and set its parameters

    vector_assembler = VectorAssembler(
        inputCols=features, outputCol="features")
    nb = NaiveBayes(labelCol='TARGET', featuresCol='features',
                    smoothing=1.0, modelType="multinomial")

    # Chain vector_assembler and forest in a Pipeline
    pipeline = Pipeline(stages=[vector_assembler, nb])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    predictions.select('TARGET', 'rawPrediction', 'prediction','probability').show(20)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                              labelCol="TARGET", metricName='areaUnderROC')

    areaUnderRoc = evaluator.evaluate(predictions)
    print(f"Area Under ROC = {areaUnderRoc}")


driver()
