from src.runners import random_forest
from src.runners import decision_tree
from src.runners import linear_support_vector_machine
from src.runners import naive_bayes
from src.runners import knn
import sys


if __name__ == '__main__':
    try:
        runner = sys.argv[1] # str
    except Exception as e:
        runner = 'random_forest'

    try:
        takeSample = sys.argv[2] == "sample"  # True if we write sample
    except Exception as e:
        takeSample = False

    if runner == 'all':
            random_forest.driver(takeSample)
            decision_tree.driver(takeSample)
            linear_support_vector_machine.driver(takeSample)
            naive_bayes.driver(takeSample)
            knn.driver(takeSample)

    print(runner, takeSample)
    driver = getattr(globals()[runner], 'driver')
    driver(takeSample=takeSample)

