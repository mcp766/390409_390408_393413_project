import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        # KNN is a lazy learning algorithm -> fit simply stores the training samples and their labels.
        self.training_data = training_data
        self.training_labels = training_labels

        # Since the training data is now stored, we can directly call predict on it.
        pred_labels = self.predict(training_data)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
         
        test_labels = []

        # Loop over each test sample
        for x in test_data:
            
        # Step 1: Compute distances from x to all training points
            # We use Euclidean distance 
            distances = np.linalg.norm(self.training_data - x, axis=1)

        # Step 2: Get indices of the k nearest neighbors
            nn_indices = np.argsort(distances)[:self.k]

        # Step 3: Retrieve the corresponding labels
            nn_labels = self.training_labels[nn_indices]
            nn_distances = distances[nn_indices]

        # Step 4: Prediction depending on task

            #classification 
            if self.task_kind == "classification":

                # Majority vote (ymv)
                labels, counts = np.unique(nn_labels, return_counts=True)
                max_count = np.max(counts)

                # Handle tie cases
                candidates = labels[counts == max_count]

                if len(candidates) > 1:
                    # Tie-break: choose the class with smallest average distance (like in class)
                    avg_distances = []
                    for c in candidates:
                        avg_distances.append(np.mean(nn_distances[nn_labels == c]))
                    pred = candidates[np.argmin(avg_distances)]
                else:
                    pred = candidates[0]
           
            # regression
            else:  

                # Weighted average
                weights = 1 / (nn_distances + 1e-8)
                pred = np.sum(weights * nn_labels) / np.sum(weights)

            test_labels.append(pred)
        return np.array(test_labels)