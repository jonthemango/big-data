from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
import utils

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoderEstimator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pyspark.sql.types as sparkTypes


root = utils.get_project_root_dir()


def cast_column_to_type(df, column_name, sql_type):
    new_df = (df.withColumn(column_name, df[column_name]
                            .cast(sql_type())))
    return new_df



def driver():
    filename = f'{root}data/application_train.csv'
    spark = utils.init_spark()
    data_df = spark.read.csv(filename, header=True)

    # Cast TARGET to int
    data_df = cast_column_to_type(data_df, 'TARGET', sparkTypes.IntegerType)

    # Selec only needed columns
    data_df = data_df.select('TARGET', 'SK_ID_CURR', 'OCCUPATION_TYPE')
    data_df = StringIndexer(inputCol="OCCUPATION_TYPE",outputCol="indexOCCUPATION_TYPE",handleInvalid="skip").fit(data_df).transform(data_df)


    encoder = OneHotEncoderEstimator(inputCols=['indexOCCUPATION_TYPE'], outputCols=['OCCUPATION_TYPE_VEC'])
    model = encoder.fit(data_df)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="TARGET", featuresCol="OCCUPATION_TYPE_VEC", numTrees=10)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[model, rf])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    predictions.show(4)

    # Select example rows to display.
    #predictions.select("TARGET", "SK_ID_CURR", "OCCUPATION_TYPE_VEC").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
        labelCol="TARGET", metricName='areaUnderROC')
    areaUnderRoc = evaluator.evaluate(predictions)
    print(f"Area Under ROC = {areaUnderRoc}")


driver()
