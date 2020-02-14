from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json

'''
This script counts the number of missing values for each column (122) in the
training set and outputs a report to 'missing_values.json'. Keep in mind that
this can take 2-3 min. 
'''

filename = "../data/application_train.csv"

# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark




def find_missing_by_column(data, column):
    total_records = data.count()
    count_missing = data.where(f"{column} is null").count()
    percent = round(count_missing/total_records*100, 2)
    return (count_missing, percent)


def generate_dict_of_missing_values():
    spark = init_spark()
    data = spark.read.csv(filename, header=True)
    columns = data.columns
    report = {"columns": {}}
    iteration = 0
    for feature in columns:
        iteration += 1
        absolute, percent = find_missing_by_column(data, feature)
        report["columns"][feature] = {
            "missing values": absolute,
            "percentage": percent
        }

        # This is just to print progress given that the script take about 2-3 min to run
        progress = iteration/len(columns)*100
        if iteration % 5 == 0:
            print(f"progress: {round(progress,1)}%")

    return report

def generate_csv_from_dict(dictionary):
    text = "Feature Name, Missing Values, Percentage Missing\n"
    for key in dictionary["columns"]:
        text += key + "," + str(dictionary["columns"][key]["missing values"]) + "," + str(dictionary["columns"][key]["percentage"])+ "\n"
    with open('missing_values.csv', 'w') as file:
        file.write(text)


report = generate_dict_of_missing_values()
generate_csv_from_dict(report)
json_string = json.dumps(report, indent=True)
with open('missing_values.json', 'w') as file:
    file.write(json_string)
print(json_string)
