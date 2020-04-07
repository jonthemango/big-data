from src.runners import random_forest
from src.runners import decision_tree
from src.runners import linear_support_vector_machine
from src.runners import naive_bayes
from src.runners import knn
import sys
import pprint



algorithms = [
    random_forest,
    decision_tree,
    linear_support_vector_machine,
    naive_bayes,
    knn
]

if __name__ == '__main__':
    try:
        runner = sys.argv[1] # str
    except Exception as e:
        runner = 'random_forest'

    try:
        takeSample = sys.argv[2] == "sample"  # True if we write sample
    except Exception as e:
        takeSample = False

    print(f"Runner = {runner}", f"Take Sample = {takeSample}")
    report = {}
    if runner == 'all':
        for algo in algorithms:
            try:
                report[algo.__name__] = algo.driver(takeSample)
            except:
                print(f"{algo.__name__} failed to execute.")
    else:
        driver = getattr(globals()[runner], 'driver')
        report[runner] = driver(takeSample=takeSample)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(report)



