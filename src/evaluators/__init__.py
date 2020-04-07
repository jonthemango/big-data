from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from collections import defaultdict

def multiple_evaluator(prediction_df):
    '''
    Evaluate all possible m
    prediction_df.prediction
    prediction_df.rawPrediction
    prediction_df.TARGET
    '''

    print('evaluating model...')

    m = calculate_TP_TN_FP_FN(prediction_df)  # m is a placeholder for metrics
    TP, TN, FP, FN = m['TP'], m['TN'], m['FP'], m['FN']

    m['precision'] = 0 if TP == 0 else TP/(TP+FP)
    m['recall'] = 0 if TP == 0 else TP/(TP+FN)
    m['accuracy'] = (TP+TN)/(TP+TN+FP+FN)
    if m['precision']+m['recall'] == 0:
        m['F1'] = 0
    else:
        m['F1'] = 2*(m['precision']*m['recall'])/(m['precision']+m['recall'])

    # Select (prediction, true label) and compute test error
    binary_class_metric = ["areaUnderROC", "areaUnderPR"]

    for metric_name in binary_class_metric:

        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction", labelCol="TARGET", metricName=metric_name)

        metric = evaluator.evaluate(prediction_df)
        m[metric_name] = metric

    return dict(m)


def calculate_TP_TN_FP_FN(prediction_df) -> dict:

    def identify(row):
        target = int(row['TARGET'])
        prediction = int(row['prediction'])

        dic = [["TN", "FP"], ["FN", "TP"]]
        ''' because of the following
            target------prediction
            0-----------------0         TN
            1-----------------1         TP
            1-----------------0         FN
            0-----------------1         FP
        '''
        return (dic[target][prediction], 1)

    tuple_count = prediction_df.rdd.map(identify).reduceByKey(lambda x, y: x+y).collect()
    # tuple_count = [('FN', 4), ('TN', 47)]  for examle

    result = defaultdict(int)
    for item in tuple_count:
        result[item[0]] = item[1]

    return result
