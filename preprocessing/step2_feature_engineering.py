from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
import utils

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pyspark.sql.types as sparkTypes


root = utils.get_project_root_dir()


'''
Transforms a df at a particular column_name by casting all values to the specified type, default is IntegerType
'''
def cast_column_to_type(df, column_name, sql_type=sparkTypes.IntegerType):
    new_df = (df.withColumn(column_name, df[column_name]
                            .cast(sql_type())))
    return new_df

'''
Transforms a df at a particular column_name by converting all unique categories to a vector as they get processed.
ie. If values in column are {a,b,b,c,d} => {<1.0,0,0>, <0,1.0,0>, ... } (Good for non-ordinal categories)
'''
def encode_using_one_hot(df, column_name):
    indexed_name = 'index_'+column_name
    vectored_name = 'vec_'+column_name

    df = StringIndexer(inputCol=column_name,outputCol=indexed_name,handleInvalid="skip").fit(df).transform(df)
    encoder = OneHotEncoderEstimator(inputCols=[indexed_name], outputCols=[vectored_name])
    model = encoder.fit(df)
    df = model.transform(df)
    df = df.drop(indexed_name)
    df = df.drop(column_name)
    df = df.withColumnRenamed(vectored_name, column_name)
    return df

'''
Transforms a df at a particular column_name by converting all unique categories to an index as they get processed.
ie. If values in column are {a,b,b,c,d} => {0.0,1.0,1.0,2.0,3.0} (Good for Binary)
'''
def encode_using_indexer(df, column_name):
    indexed_name = 'index_'+column_name
    df = StringIndexer(inputCol=column_name,outputCol=indexed_name,handleInvalid="skip").fit(df).transform(df)
    df = df.drop(column_name)
    df = df.withColumnRenamed(indexed_name, column_name)
    return df





def driver():
    # Set up
    filename = f'{root}data/application_train.csv'
    spark = utils.init_spark()
    data_df = spark.read.csv(filename, header=True)

    # List of Features
    features = ['FLAG_OWN_CAR','CODE_GENDER','AMT_GOODS_PRICE','DAYS_EMPLOYED','DAYS_BIRTH','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'AMT_CREDIT','FLAG_OWN_REALTY','FLAG_MOBIL','NAME_FAMILY_STATUS', 'NAME_TYPE_SUITE', 'NAME_EDUCATION_TYPE','NAME_CONTRACT_TYPE','NAME_INCOME_TYPE']
    
    # Feature Encoding

    # Cast TARGET to int
    data_df = cast_column_to_type(data_df, 'TARGET', sparkTypes.IntegerType)
    data_df = cast_column_to_type(data_df, 'AMT_CREDIT', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'AMT_GOODS_PRICE', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'DAYS_EMPLOYED', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'DAYS_BIRTH', sparkTypes.FloatType)

    ## Indexer
    data_df = encode_using_indexer(data_df, 'FLAG_OWN_CAR')
    data_df = encode_using_indexer(data_df, 'CODE_GENDER')
    data_df = encode_using_indexer(data_df, 'FLAG_OWN_REALTY')
    data_df = encode_using_indexer(data_df, 'FLAG_MOBIL')
    data_df = encode_using_indexer(data_df, 'NAME_CONTRACT_TYPE')
    data_df = encode_using_indexer(data_df, 'FLAG_DOCUMENT_2')
    data_df = encode_using_indexer(data_df, 'FLAG_DOCUMENT_3')
    data_df = encode_using_indexer(data_df, 'FLAG_DOCUMENT_4')

    ## One-Hot Encoding
    data_df = encode_using_one_hot(data_df, 'NAME_EDUCATION_TYPE')
    data_df = encode_using_one_hot(data_df, 'NAME_FAMILY_STATUS')
    data_df = encode_using_one_hot(data_df, 'NAME_TYPE_SUITE')
    data_df = encode_using_one_hot(data_df, 'NAME_INCOME_TYPE')
    data_df = encode_using_one_hot(data_df, 'OCCUPATION_TYPE')
    
    

    # Assemble all features in a vector using Vector Assembler
    # map it to new column named features
    vector_assembler = VectorAssembler(inputCols=features, outputCol="features")

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_df.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="TARGET", featuresCol="features", numTrees=10)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[vector_assembler, rf])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)
    predictions.show(4)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
        labelCol="TARGET", metricName='areaUnderROC')
    areaUnderRoc = evaluator.evaluate(predictions)
    print(f"Area Under ROC = {areaUnderRoc}")


driver()
