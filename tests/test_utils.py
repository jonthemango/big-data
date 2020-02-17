from utils import dump, init_spark
import os

def test_dump():
    spark = init_spark()
    sc = spark.sparkContext
    data = [1, 2, 3, 4, 5]
    distData = sc.parallelize(data)

    fn = "test_dump"
    dump(distData,fn)
    with open(fn, 'r') as f:
        assert len(f.readlines()) == len(data)
    os.remove(fn)
