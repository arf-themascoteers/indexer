from evaluator import Evaluator

if __name__ == '__main__':
    algos = ["mi","sfs","lasso","fsdr"]
    algos = ["fsdr"]
    datasets = [(False, False), (True, False), (False, True)]
    datasets = [(False, True)]
    sizes = [2, 5, 10, 15, 20]
    sizes = [2, 5, 10]
    tasks = []
    for reduced_features, reduced_rows in datasets:
        for algorithm in algos:
            for size in sizes:
                tasks.append(
                    {
                        "reduced_features":reduced_features,
                        "reduced_rows":reduced_rows,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()