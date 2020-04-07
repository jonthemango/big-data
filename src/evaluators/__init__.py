from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

def multiple_evaluator(prediction_df):
    '''
    Evaluate all possible metrics
    prediction_df.prediction
    prediction_df.rawPrediction
    prediction_df.TARGET
    '''
    # Select (prediction, true label) and compute test error
    multi_class_metric = ["f1", "weightedPrecision", "weightedRecall", "accuracy"]
    binary_class_metric = ["areaUnderROC", "areaUnderPR"]

    retval = {}

    for metric_name in multi_class_metric:
        multi_class_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="TARGET", metricName=metric_name)
        metric = multi_class_evaluator.evaluate(prediction_df)
        print(f"{metric_name} = {metric}")
        retval[metric_name] = metric

    for metric_name in binary_class_metric:
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="TARGET", metricName=metric_name)
        metric = evaluator.evaluate(prediction_df)
        print(f"{metric_name} = {metric}")
        retval[metric_name] = metric


    print("Our F1 = ", 2*(retval['weightedPrecision']*retval['weightedRecall'])/(retval['weightedPrecision']+retval['weightedRecall']))
    print(retval)
    return retval

