from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import json
import copy as cp
import utils

'''
This script counts the number of missing values for each column of each file in
the training set and outputs a report to 'missing_values.json'. Keep in mind
that this can take ~7min.
'''

data_directory = utils.get_project_root_dir() + 'data/'

filenames = [
    'bureau_balance.csv',
    'bureau.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'POS_CASH_balance.csv',
    'previous_application.csv',
    'application_train.csv',
]


json_output_file = f'{utils.get_project_root_dir()}preprocessing/missing_values.json'
csv_output_file = f'{utils.get_project_root_dir()}preprocessing/missing_values.csv'


def driver(include_unique_count=True):
    '''This is the full process of analysing columns in all data file'''
    print("Running step 1 column analysis driver")
    report: dict = analyse_columns_in_all_data_files(
        filenames, include_unique_count=True)

    json_report: str = json.dumps(report, indent=4)
    write2file(json_report, json_output_file)

    csv_report: str = generate_csv_from_json(json_output_file)
    write2file(csv_report, csv_output_file)

    print(f'View {json_output_file} and {csv_output_file} for result')


def sample_driver():
    '''This is the same as driver but only on the sample data (for testing purposes)'''

    filenames = [
        'sample/sample_bureau_balance.csv',
        'sample/sample_bureau.csv',
        'sample/sample_credit_card_balance.csv',
        'sample/sample_installments_payments.csv',
        'sample/sample_POS_CASH_balance.csv',
        'sample/sample_previous_application.csv',
        'sample/sample_application_train.csv',
    ]
    sample_json_output_file = f'{utils.get_project_root_dir()}preprocessing/sample_missing_values.json'
    sample_csv_output_file = f'{utils.get_project_root_dir()}preprocessing/sample_missing_values.csv'

    report: dict = analyse_columns_in_all_data_files(filenames)
    json_report: str = json.dumps(report, indent=4)
    write2file(json_report, sample_json_output_file)

    csv_report: str = generate_csv_from_json(json_output_file)
    write2file(csv_report, sample_csv_output_file)

    print(
        f'View {sample_json_output_file} and {sample_csv_output_file} for result')


def write2file(string, filename: str):
    ''' helper function, which overwrites contents of filename if exists'''
    with open(filename, 'w')as file:
        file.write(string)


def find_missing_by_column(data, column: str, include_unique_count=True):
    total_records = data.count()
    count_missing = data.where(f"{column} is null").count()
    percent = round(count_missing/total_records*100, 2)

    if include_unique_count:
        unique_count = data.select(column).distinct().count()
    else:
        unique_count = "N/A"
    return (count_missing, percent, unique_count)


def generate_dict_of_missing_values(filename: str, include_unique_count=True):
    spark = utils.init_spark()
    data = spark.read.csv(filename, header=True)
    columns = data.columns
    report = {"columns": {}, "complete_features": 0}
    iteration = 0

    for feature in columns:
        iteration += 1
        absolute, percent, unique_count = find_missing_by_column(
            data, feature, include_unique_count=True)
        report["columns"][feature] = {
            "unique_count": unique_count,
            "missing_values": absolute,
            "percentage": percent
        }
        if absolute == 0:
            report["complete_features"] += 1

        # printing progress given that script can take ~7min to run
        progress = iteration/len(columns)*100
        if iteration % 5 == 0:
            print(f"progress: {round(progress,1)}%")

    return report


def analyse_columns_in_all_data_files(filenames: list, include_unique_count=True):
    final_report = {}
    for file in filenames:
        file_path = f'{data_directory}{file}'
        print(f'generating report for file: {file}')
        final_report[file] = generate_dict_of_missing_values(
            file_path, include_unique_count)
    return final_report


def generate_csv_from_json(json_filename: str):

    csv_text: str = ',Filename, Column Name, Unique Categories, Missing Values, Percentage Missing\n'
    count = 0
    with open(json_filename) as report_file:

        report: dict = json.loads(report_file.read())
        filenames: list = sorted(report.keys())

        for file in filenames:

            columns = sorted(
                report[file]["columns"], key=lambda column: report[file]["columns"][column]["missing_values"])

            for column in columns:
                count += 1
                missing_values = str(
                    report[file]["columns"][column]["missing_values"])
                unique_count = str(
                    report[file]["columns"][column]["unique_count"]
                )

                percent = str(report[file]["columns"][column]["percentage"])
                csv_text += f'{count}, {file}, {column}, {unique_count} , {missing_values}, {percent}\n'

    return csv_text


if __name__ == '__main__':
    driver(include_unique_count=False)
