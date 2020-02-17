import preprocessing


def test_json_to_csv_conversion():
    input_file = 'tests/resources/test_missing_values.json'

    # this is what we want to obtain
    goal_file = 'tests/resources/test_missing_values.csv'

    csv_report: str = preprocessing.step1_column_analysis.generate_csv_from_json(
        input_file)

    with open(goal_file) as file:
        target: str = file.read()
        assert(csv_report == target)
