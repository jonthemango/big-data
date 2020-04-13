# Add this for dumb relative imports
import sys
sys.path.append(".")

from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import json
import copy as cp
from src import utils

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
    '''Returns an aggregated df of the  min, max, and mean of a column for each SK_ID_CURR'''
    aggregations = []
    for column_name in column_names:
        df = cast_column_to_type(df, column_name, sql_type)
        aggregations.append(F.avg(column_name))
        aggregations.append(F.min(column_name))
        aggregations.append(F.max(column_name))

    agg_df = df.groupBy("SK_ID_CURR").agg(*aggregations) #.agg(aggregations)
    return agg_df, agg_df.columns

def get_previous_loan_status(bureau_df):
    previous_loan_count_df = (bureau_df
                              .groupBy('SK_ID_CURR')
                              .count().na.fill(0)
                              .withColumnRenamed('count', 'previous_loans_count')
                              )

    late_df = (bureau_df
               .where(bureau_df['CREDIT_DAY_OVERDUE'] > 0)
               .groupBy('SK_ID_CURR')
               .count()
               .withColumnRenamed('count', 'late_loans')
               )

    ontime_df = (bureau_df
                 .where(bureau_df['CREDIT_DAY_OVERDUE'] <= 0)
                 .groupBy('SK_ID_CURR')
                 .count()
                 .withColumnRenamed('count', 'ontime_loans')
                 )

    joined = previous_loan_count_df.join(late_df, on=['SK_ID_CURR'], how='outer').na.fill(0)
    joined = joined.join(ontime_df, on=['SK_ID_CURR'], how='outer').na.fill(0)
    joined = joined.withColumn('late_loan_ratio', joined['late_loans'] / joined['previous_loans_count'])
    return joined


def preprocess_features(takeSample=False):
    application_filename = f'{root}data/application_train.csv'
    bureau_filename = f'{root}data/bureau.csv'

    # Read CSV file
    spark = utils.init_spark()
    data_df = spark.read.csv(application_filename, header=True)
    previous_loans_df = spark.read.csv(bureau_filename, header=True) # X columns

    # Sample Data
    if takeSample:
        data_df = data_df.sample(0.001)
        sample_ids = data_df.select('SK_ID_CURR')

        #remove rows from bureau that dont refer to sample_ids
        previous_loans_df = previous_loans_df.join(sample_ids,on='SK_ID_CURR',how='inner')

    print(f"\n\n*---size of bureau.csv dataframe = {previous_loans_df.count()}\n\n")

    data_df.cache()
    previous_loans_df.cache()

    # Count of Applicant's Previous Loans (ontime vs late)
    payment_status_df = get_previous_loan_status(previous_loans_df)
    data_df = data_df.join(payment_status_df, on="SK_ID_CURR", how="inner")

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
        'NAME_TYPE_SUITE',
        'NAME_EDUCATION_TYPE',
        'NAME_CONTRACT_TYPE',
        'NAME_INCOME_TYPE',
        'avg(CREDIT_DAY_OVERDUE)',
        'min(DAYS_CREDIT)',
        'max(DAYS_CREDIT)',
        'avg(DAYS_CREDIT)',
        'late_loan_ratio',
        'OCCUPATION_TYPE',
        'NAME_FAMILY_STATUS'
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

def preprocess_features2(application_filename = f'{root}data/application_train.csv',takeSample=False):

    # Read CSV file
    spark = utils.init_spark()
    data_df = spark.read.csv(application_filename, header=True)

    # Compute Loan Repayment Ratio
    bureau_filename = f'{root}data/bureau.csv'
    previous_loans_df = spark.read.csv(bureau_filename, header=True) # X columns
    payment_status_df = get_previous_loan_status(previous_loans_df)
    data_df = data_df.join(payment_status_df, on="SK_ID_CURR", how="inner")

    data_df.cache()

    # List of Features
    # https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering/notebook#Feature-Engineering-Outcomes
    features = [
        "EXT_SOURCE_1",
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'late_loan_ratio',
        'avg(DAYS_CREDIT)',
        'credit_over_income'
        # 'previous_loan_count'
    ]

    # Cast columns to float
    if application_filename==f'{root}data/application_train.csv':
        data_df = cast_column_to_type(data_df, 'TARGET', sparkTypes.IntegerType)

    data_df = cast_column_to_type(data_df, 'EXT_SOURCE_1', sparkTypes.FloatType).na.fill(0.5, 'EXT_SOURCE_1')
    data_df = cast_column_to_type(data_df, 'EXT_SOURCE_2', sparkTypes.FloatType).na.fill(0.5, 'EXT_SOURCE_2')
    data_df = cast_column_to_type(data_df, 'EXT_SOURCE_3', sparkTypes.FloatType).na.fill(0.5, 'EXT_SOURCE_3')
    data_df = cast_column_to_type(data_df, 'AMT_INCOME_TOTAL', sparkTypes.FloatType)
    data_df = cast_column_to_type(data_df, 'AMT_CREDIT', sparkTypes.FloatType)

    # Combine two features into one AMT_CREDIT/AMT_INCOME_TOTAL as credit_over_loan
    data_df = data_df.withColumn('credit_over_income',data_df['AMT_CREDIT']/data_df['AMT_INCOME_TOTAL'])

    # Encode DAYS_CREDIT as avg(DAYS_CREDIT)
    agg_df, _ = encode_using_stats(previous_loans_df, ['DAYS_CREDIT'], sql_type=sparkTypes.FloatType)
    data_df = data_df.join(agg_df, on="SK_ID_CURR", how='inner')

    # Count number of previous_application
    '''count_of_previous = previous_loans_df.groupby('SK_ID_CURR').count().withColumnRenamed('count', 'previous_loan_count').select("SK_ID_CURR","previous_loan_count")
    data_df = data_df.join(count_of_previous, on="SK_ID_CURR")'''

    if application_filename==f'{root}data/application_train.csv':
        data_df = data_df.select(features+['TARGET', 'SK_ID_CURR'])
    else:
        data_df = data_df.select(features+['SK_ID_CURR'])

    describe = data_df.describe()
    describe.show()
    data_df.show()
    return data_df, features

if __name__ == "__main__":
    data_df, features = preprocess_features2()
    data_df.describe().show()
    ''''#data_df.show(100)
    def toCSVLine(data):
        return ','.join(str(d) for d in data)

    #lines = data_df.rdd.map(toCSVLine).reduce(lambda x,y : x+y+"\n")
    with open('out.csv', 'w') as f:
        #f.write(lines)
        pass'''

    data_df.show()

    print(features)
