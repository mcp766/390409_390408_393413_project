import argparse
import numpy as np

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
                   train_labels_classif, test_labels_classif):
    """
    Initialize, train, evaluate and print results for one method/task pair.
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

    print(f"\n===== {method_name} | {task_name} =====")

    # Classification
    if task_name == "classification":
        if method_name == "linear_regression":
            print("Skipped: linear regression cannot be used for classification.")
            return

        preds_train = method_obj.fit(train_features, train_labels_classif)
        preds = method_obj.predict(test_features)

        acc_train = accuracy_fn(preds_train, train_labels_classif)
        f1_train = macrof1_fn(preds_train, train_labels_classif)
        print(f"Train set: accuracy = {acc_train:.3f}% - F1-score = {f1_train:.6f}")

        acc_test = accuracy_fn(preds, test_labels_classif)
        f1_test = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc_test:.3f}% - F1-score = {f1_test:.6f}")

    # Regression
    elif task_name == "regression":
        if method_name == "logistic_regression":
            print("Skipped: logistic regression cannot be used for regression.")
            return

        preds_train = method_obj.fit(train_features, train_labels_reg)
        preds = method_obj.predict(test_features)

        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"Train set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {task_name}")
    
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
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Shuffle the training set before splitting into train/validation
        indices = np.random.permutation(len(train_features))
        train_features = train_features[indices]
        train_labels_reg = train_labels_reg[indices]
        train_labels_classif = train_labels_classif[indices]

        val_size = int(0.2 * len(train_features))
        test_features = train_features[-val_size:]
        test_labels_reg = train_labels_reg[-val_size:]
        test_labels_classif = train_labels_classif[-val_size:]
        train_features = train_features[:-val_size]
        train_labels_reg = train_labels_reg[:-val_size]
        train_labels_classif = train_labels_classif[:-val_size]

    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1 # Avoid division by zero for constant features
    train_features = normalize_fn(train_features, mean, std)
    test_features = normalize_fn(test_features, mean, std)


    ## 3. Initialize the method you want to use.

       ## 3. Run one method or all methods

    if args.method == "all":
        # Run all valid method/task combinations
        run_one_method(
            "dummy_classifier", "classification", args,
            train_features, test_features,
            train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif
        )

        run_one_method(
            "knn", "classification", args,
            train_features, test_features,
            train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif
        )

        run_one_method(
            "knn", "regression", args,
            train_features, test_features,
            train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif
        )

        run_one_method(
            "logistic_regression", "classification", args,
            train_features, test_features,
            train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif
        )

        run_one_method(
            "linear_regression", "regression", args,
            train_features, test_features,
            train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif
        )

    else:
        # Run only the user-specified method/task pair
        run_one_method(
            args.method, args.task, args,
            train_features, test_features,
            train_labels_reg, test_labels_reg,
            train_labels_classif, test_labels_classif
        )

    ## 4. Train and evaluate the method

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
