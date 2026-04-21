import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)

def run_one_method(method_name, task_name, args,
                   train_features, test_features,
                   train_labels_reg, test_labels_reg,
                   train_labels_classif, test_labels_classif,
                   verbose=True):
                   
    """
    Initialize, train, evaluate and print results for one method/task pair.
    Returns a dictionary of results.
    """

    # Initialize the method
    if method_name == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif method_name == "knn":
        method_obj = KNN(k=args.K, task_kind=task_name)

    elif method_name == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif method_name == "linear_regression":
        method_obj = LinearRegression()

    else:
        raise ValueError(f"Unknown method: {method_name}")

    if verbose:
        print(f"\n===== {method_name} | {task_name} =====")

    results = {
        "method": method_name,
        "task": task_name,
    }

    # Classification
    if task_name == "classification":
        if method_name == "linear_regression":
            if verbose:
                print("Skipped: linear regression cannot be used for classification.")
            return None

        t1 = time.time()
        preds_train = method_obj.fit(train_features, train_labels_classif)
        t2 = time.time()

        preds = method_obj.predict(test_features)
        t3 = time.time()

        acc_train = accuracy_fn(preds_train, train_labels_classif)
        f1_train = macrof1_fn(preds_train, train_labels_classif)
        acc_test = accuracy_fn(preds, test_labels_classif)
        f1_test = macrof1_fn(preds, test_labels_classif)

        results.update({
            "train_accuracy": acc_train,
            "train_f1": f1_train,
            "test_accuracy": acc_test,
            "test_f1": f1_test,
            "train_time": t2 - t1,
            "predict_time": t3 - t2,
        })

        if verbose:
            print(f"Train set: accuracy = {acc_train:.3f}% - F1-score = {f1_train:.6f}")
            print(f"Test set:  accuracy = {acc_test:.3f}% - F1-score = {f1_test:.6f}")
            print(f"Runtime:   fit = {t2 - t1:.6f}s - predict = {t3 - t2:.6f}s")

    # Regression
    elif task_name == "regression":
        if method_name == "logistic_regression":
            if verbose:
                print("Skipped: logistic regression cannot be used for regression.")
            return None

        if method_name == "dummy_classifier":
            if verbose:
                print("Skipped: dummy classifier is not used for regression.")
            return None

        t1 = time.time()
        preds_train = method_obj.fit(train_features, train_labels_reg)
        t2 = time.time()

        preds = method_obj.predict(test_features)
        t3 = time.time()

        train_mse = mse_fn(preds_train, train_labels_reg)
        test_mse = mse_fn(preds, test_labels_reg)

        results.update({
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_time": t2 - t1,
            "predict_time": t3 - t2,
        })

        if verbose:
            print(f"Train set: MSE = {train_mse:.6f}")
            print(f"Test set:  MSE = {test_mse:.6f}")
            print(f"Runtime:   fit = {t2 - t1:.6f}s - predict = {t3 - t2:.6f}s")

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return results

def plot_knn(k_values, acc_values, f1_values, mse_values, save_path="knn_vs_k_combined.png"):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    # ===== Classification =====
    axs[0].plot(k_values, acc_values, marker='o', label="Accuracy")
    axs[0].plot(k_values, f1_values, marker='s', label="Macro F1")
    axs[0].set_title("KNN Classification vs K")
    axs[0].set_xlabel("K")
    axs[0].set_ylabel("Score")
    axs[0].legend()
    axs[0].grid(True)

    # ===== Regression =====
    axs[1].plot(k_values, mse_values, marker='o', color='red', label="MSE")
    axs[1].set_title("KNN Regression vs K")
    axs[1].set_xlabel("K")
    axs[1].set_ylabel("MSE")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_summary_tables(results_list):
    """
    Prints formatted summary tables comparing all methods:
    shows classification metrics (accuracy, F1) and regression metric (MSE),
    along with training and prediction runtimes.
    """
    print("\n==============================")
    print("Classification summary")
    print("==============================")
    print("Method                Train Acc    Test Acc     Train F1     Test F1      Fit Time(s)   Pred Time(s)")
    for r in results_list:
        if r is None or r["task"] != "classification":
            continue
        print(f"{r['method']:<20} {r['train_accuracy']:>10.3f} {r['test_accuracy']:>11.3f} "
              f"{r['train_f1']:>11.6f} {r['test_f1']:>11.6f} "
              f"{r['train_time']:>12.6f} {r['predict_time']:>13.6f}")

    print("\n==============================")
    print("Regression summary")
    print("==============================")
    print("Method                Train MSE    Test MSE     Fit Time(s)   Pred Time(s)")
    for r in results_list:
        if r is None or r["task"] != "regression":
            continue
        print(f"{r['method']:<20} {r['train_mse']:>10.6f} {r['test_mse']:>11.6f} "
              f"{r['train_time']:>12.6f} {r['predict_time']:>13.6f}")
        



def main(args):
    """
    The main function of the script.

        Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """

    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'], feature_data['xtest'], feature_data['ytrainreg'],
        feature_data['ytestreg'], feature_data['ytrainclassif'], feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Shuffle before split
        indices = np.random.permutation(len(train_features))
        train_features = train_features[indices]
        train_labels_reg = train_labels_reg[indices]
        train_labels_classif = train_labels_classif[indices]

        # 80/20 split
        val_size = int(0.2 * len(train_features))

        test_features = train_features[-val_size:]
        test_labels_reg = train_labels_reg[-val_size:]
        test_labels_classif = train_labels_classif[-val_size:]

        train_features = train_features[:-val_size]
        train_labels_reg = train_labels_reg[:-val_size]
        train_labels_classif = train_labels_classif[:-val_size]

    ### WRITE YOUR CODE HERE to do any other data processing

    # Normalize
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1

    train_features = normalize_fn(train_features, mean, std)
    test_features = normalize_fn(test_features, mean, std)

    ## 3. Initialize the method

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "linear_regression":
        method_obj = LinearRegression()

    elif args.method == "all": #bonus feature
        # dummy placeholder so Step 4 doesn't crash
        method_obj = DummyClassifier(arg1=1, arg2=2)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method  (UNCHANGED)

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_classif)

        # Predict on unseen data
        preds = method_obj.predict(test_features)
       
        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_reg)
        
        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    if args.method == "all":

        results_list = []

        results_list.append(run_one_method("dummy_classifier", "classification", args,
            train_features, test_features, train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif))

        results_list.append(run_one_method("knn", "classification", args,
            train_features, test_features, train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif))

        results_list.append(run_one_method("knn", "regression", args,
            train_features, test_features, train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif))

        results_list.append(run_one_method("logistic_regression", "classification", args,
            train_features, test_features, train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif))

        results_list.append(run_one_method("linear_regression", "regression", args,
            train_features, test_features, train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif))

        print_summary_tables(results_list)

        # KNN plots
        k_values = [1, 3, 5, 7, 9, 11, 15]
        original_k = args.K

        acc_values, f1_values, mse_values = [], [], []

        for k in k_values:
            args.K = k
            r_class = run_one_method("knn", "classification", args,
                train_features, test_features, train_labels_reg, test_labels_reg,
                train_labels_classif, test_labels_classif, verbose=False)

            r_reg = run_one_method("knn", "regression", args,
                train_features, test_features, train_labels_reg, test_labels_reg,
                train_labels_classif, test_labels_classif, verbose=False)

            acc_values.append(r_class["test_accuracy"])
            f1_values.append(r_class["test_f1"])
            mse_values.append(r_reg["test_mse"])

        plot_knn(k_values, acc_values, f1_values, mse_values)

        args.K = original_k

        print("\nSaved plots:")
        print(" - knn_vs_k_combined.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="all",
        type=str,
        help="all / dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )

    args = parser.parse_args()
    main(args)
