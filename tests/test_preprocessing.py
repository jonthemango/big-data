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
    goal_file = f'{project_root}tests/resources/test_missing_values.json'
    goal_data = {}

    with open(goal_file, 'r') as read_file:
        goal_data = json.load(read_file)

    filenames = [
        'bureau_balance.csv',
        'bureau.csv',
        'credit_card_balance.csv',
        'installments_payments.csv',
        'POS_CASH_balance.csv',
        'previous_application.csv',
        'application_train.csv',
    ]

    report: dict = prep.step1_column_analysis.analyse_columns_in_all_data_files(
        filenames)

    assert(report == goal_data)
