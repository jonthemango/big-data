from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

def dump(rdd, fn="out"):
    if isinstance(rdd, DataFrame):
        rdd = rdd.rdd
    with open(fn, 'w+') as f:
        text = rdd.map(lambda row: str(row) + "\n").reduce(lambda row1,row2: row1+ row2)
        f.write(text)
    return True

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark