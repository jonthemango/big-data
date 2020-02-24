import preprocessing as prep
import utils
import json

project_root = utils.get_project_root_dir()


def test_json_to_csv_conversion():
    input_file = f'{project_root}tests/resources/test_missing_values.json'

    # this is what we want to obtain
    goal_file = f'{project_root}tests/resources/test_missing_values.csv'

    csv_report: str = prep.step1_column_analysis.generate_csv_from_json(
        input_file)

    with open(goal_file) as file:
        target: str = file.read()
        assert(csv_report == target)


def test_report_inmemory_generation():
    goal_file = f'{project_root}tests/resources/sample_missing_values.json'
    goal_data = {}

    with open(goal_file, 'r') as read_file:
        goal_data = json.load(read_file)

    filenames = [
        'sample/sample_bureau_balance.csv',
        'sample/sample_bureau.csv',
        'sample/sample_credit_card_balance.csv',
        'sample/sample_installments_payments.csv',
        'sample/sample_POS_CASH_balance.csv',
        'sample/sample_previous_application.csv',
        'sample/sample_application_train.csv',
    ]

    report: dict = prep.step1_column_analysis.analyse_columns_in_all_data_files(
        filenames)

    assert(report == goal_data)
