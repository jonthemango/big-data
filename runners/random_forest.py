from preprocessing import step2_feature_engineering as feature_eng

from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
import utils

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def driver():

    # Pre-process features
    data_df, features = feature_eng.preprocess_features()
    data_df.cache()
    print("\n\n\n***Finished Pre-Processing Features")

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])  # 300*0.7 = 210
    print("\n\n\n***Split into Train and Test")
    print("\n\n\n***",trainingData.count(), len(trainingData.columns))
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
    print("\n\n\n***Trained Model")

    # Make predictions.
    predictions = model.transform(testData)
    predictions.select('TARGET', 'rawPrediction',
                       'prediction', 'probability').show(20)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol="TARGET", metricName='areaUnderROC')
    multi_class_evaluator = MulticlassClassificationEvaluator(
        predictionCol="prediction", labelCol="TARGET", metricName="f1")

    print("\n\n\n***Evaluating")
    areaUnderRoc = evaluator.evaluate(predictions)
    print(f"Area Under ROC = {areaUnderRoc}")
    f1 = multi_class_evaluator.evaluate(predictions)
    print(f"F1 = {f1}")


driver()
