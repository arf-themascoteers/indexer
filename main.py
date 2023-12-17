from evaluator import Evaluator

if __name__ == '__main__':
    algorithms = ["mi","sfs","lasso","fsdr"]
    algorithms = ["fsdr"]
    datasets = ["original",
                "dataset_4200_21782.csv",
                "dataset_525_21782.csv",
                "dataset_66_21782.csv",
                "dataset_4200_871.csv",
                "dataset_525_871.csv"
                ]
    datasets = ["dataset_525_871.csv"]
    sizes = [2, 5, 10, 15, 20]
    sizes = [2, 5]
    tasks = []
    for dataset in datasets:
        for algorithm in algorithms:
            for size in sizes:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()