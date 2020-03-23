from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import json
import copy as cp
import utils

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import pyspark.sql.types as sparkTypes


root = utils.get_project_root_dir()


def cast_column_to_type(df, column_name, sql_type=sparkTypes.FloatType):
    '''
    Transforms a df at a particular column_name by casting all values to the specified type, default is IntegerType
    '''
    new_df = (df.withColumn(column_name, df[column_name]
                            .cast(sql_type())))
    return new_df


def encode_using_one_hot(df, column_name):
    '''
    Transforms a df at a particular column_name by converting all unique categories to a vector as they get processed.
    ie. If values in column are {a,b,b,c,d} => {<1.0,0,0>, <0,1.0,0>, ... } (Good for non-ordinal categories)
    '''
    indexed_name = 'index_'+column_name
    vectored_name = 'vec_'+column_name

    df = StringIndexer(inputCol=column_name, outputCol=indexed_name,
                       handleInvalid="skip").fit(df).transform(df)

    encoder = OneHotEncoderEstimator(
        inputCols=[indexed_name], outputCols=[vectored_name])
    model = encoder.fit(df)
    df = model.transform(df)
    df = df.drop(indexed_name)
    df = df.drop(column_name)
    df = df.withColumnRenamed(vectored_name, column_name)
    return df


def encode_using_indexer(df, column_name):
    '''
    Transforms a df at a particular column_name by converting all unique categories to an index as they get processed.
    ie. If values in column are {a,b,b,c,d} => {0.0,1.0,1.0,2.0,3.0} (Good for Binary)
    '''
    indexed_name = 'index_'+column_name
    df = StringIndexer(inputCol=column_name, outputCol=indexed_name,
                       handleInvalid="skip").fit(df).transform(df)
    df = df.drop(column_name)
    df = df.withColumnRenamed(indexed_name, column_name)
    return df


def encode_using_stats(df, column_names, sql_type=sparkTypes.FloatType):
    '''Returns an aggregated df of the  min, max, and mean of a column'''
    aggregations = []
    for column_name in column_names:
        df = cast_column_to_type(df, column_name, sql_type)
        aggregations.append(F.avg(column_name))
        aggregations.append(F.min(column_name))
        aggregations.append(F.max(column_name))

    agg_df = df.groupBy("SK_ID_CURR").agg(*aggregations) #.agg(aggregations)
    return agg_df, agg_df.columns


def preprocess_features():
    application_filename = f'{root}data/application_train.csv'
    bureau_filename = f'{root}data/bureau.csv'

    spark = utils.init_spark()
    data_df = spark.read.csv(application_filename, header=True)

    # Count of Applicant's Previous Loans
    previous_loans_df = spark.read.csv(bureau_filename, header=True)
    previous_loan_count = previous_loans_df.groupBy('SK_ID_CURR').count().na.fill(0, "count")  # [SK_ID_CURR, count]
    data_df = data_df.join(previous_loan_count,on="SK_ID_CURR")  # join on SK_ID_CURR

    # Numerical Data
    agg_df, _ = encode_using_stats(previous_loans_df, ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE'])
    data_df = data_df.join(agg_df, on="SK_ID_CURR")
    agg_df.show(10)

    # List of Features
    features = [
        'FLAG_OWN_CAR',
        'CODE_GENDER',
        'AMT_GOODS_PRICE',
        'DAYS_EMPLOYED',
        'DAYS_BIRTH',
        'FLAG_DOCUMENT_2',
        'FLAG_DOCUMENT_3',
        'FLAG_DOCUMENT_4',
        'AMT_CREDIT',
        'FLAG_OWN_REALTY',
        'FLAG_MOBIL',
        'NAME_FAMILY_STATUS',
        'NAME_TYPE_SUITE',
        'NAME_EDUCATION_TYPE',
        'NAME_CONTRACT_TYPE',
        'NAME_INCOME_TYPE',
        'count',
        'avg(CREDIT_DAY_OVERDUE)',
        'min(DAYS_CREDIT)',
        'max(DAYS_CREDIT)',
        'avg(DAYS_CREDIT)'
    ]

    # Feature Encoding

    # Cast TARGET to int
    data_df = cast_column_to_type(data_df, 'TARGET', sparkTypes.IntegerType)
    data_df = cast_column_to_type(data_df, 'AMT_CREDIT', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'AMT_GOODS_PRICE', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'DAYS_EMPLOYED', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'DAYS_BIRTH', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'avg(CREDIT_DAY_OVERDUE)', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'min(DAYS_CREDIT)', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'max(DAYS_CREDIT)', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'avg(DAYS_CREDIT)', sparkTypes.FloatType)

    # Indexer
    data_df = encode_using_indexer(data_df, 'FLAG_OWN_CAR')
    data_df = encode_using_indexer(data_df, 'CODE_GENDER')
    data_df = encode_using_indexer(data_df, 'FLAG_OWN_REALTY')
    data_df = encode_using_indexer(data_df, 'FLAG_MOBIL')
    data_df = encode_using_indexer(data_df, 'NAME_CONTRACT_TYPE')
    data_df = encode_using_indexer(data_df, 'FLAG_DOCUMENT_2')
    data_df = encode_using_indexer(data_df, 'FLAG_DOCUMENT_3')
    data_df = encode_using_indexer(data_df, 'FLAG_DOCUMENT_4')

    # One-Hot Encoding
    data_df = encode_using_one_hot(data_df, 'NAME_EDUCATION_TYPE')
    data_df = encode_using_one_hot(data_df, 'NAME_FAMILY_STATUS')
    data_df = encode_using_one_hot(data_df, 'NAME_TYPE_SUITE')
    data_df = encode_using_one_hot(data_df, 'NAME_INCOME_TYPE')
    data_df = encode_using_one_hot(data_df, 'OCCUPATION_TYPE')

    return data_df, features


if __name__ == "__main__":
    data_df, features = preprocess_features()
    data_df.show(10)
    print(features)
