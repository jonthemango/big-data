
def dump(rdd):
    from pyspark.sql import DataFrame
    if isinstance(rdd, DataFrame):
        rdd = rdd.rdd
    with open('out', 'w+') as f:
        text = rdd.map(lambda row: str(row) + "\n").reduce(lambda row1,row2: row1+ row2)
        f.write(text)
    return True

