# coding=utf8
"""
Created on Sun Nov 30 01:56:33 2025
@author: Neal LONG

Note:
    1. DO NOT import additional packages
"""


import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score


class NB:

    def fit(self, df, y):
        """
        Fit the Naive Bayes classifier by counting attribute–class co-occurrences.

        This method scans all training examples once and builds two internal
        dictionaries:

        1. event_count_dict:
           - Key: (event_str, label), where event_str is the string
             "attr=attr_val", e.g. "Outlook=Sunny".
           - Value: integer count of how many times this attribute value was
             observed together with the given class label in the training data.

        2. prior_count_dict:
           - Key: class label (one of the values in y).
           - Value: integer count of how many times this label appears in y.

        It also stores the attribute names and the total number of training
        examples, which are later used for prediction.

        Parameters
        ----------
        df : pandas.DataFrame
            Training feature matrix. Each row represents one training example,
            and each column represents one attribute (feature). Attribute values
            should be categorical or discretized numeric values that can be
            compared with equality, for example strings or small integers.

        y : array-like (e.g. pandas.Series, list, numpy.ndarray)
            One-dimensional collection of class labels for each training example.
            It must have the same length as the number of rows in df; that is,
            len(y) must equal df.shape[0].

        ------------
        After calling this method, the following attributes are created/updated:

        self.event_count_dict : dict
            Maps (event_str, label) → count, where event_str has the form
            "attr=attr_val".

        self.prior_count_dict : dict
            Maps label → count (the number of training examples with that label).

        self.attrs : pandas.Index
            The column names of df, representing the attribute names.

        self.size : int
            Total number of training examples used to fit the model.

        Notes
        -----
        - This implementation uses simple frequency counts without smoothing.
        - It assumes df and y are aligned by row index (i.e. the i-th row of df
          corresponds to the i-th element of y).
        """

        self.event_count_dict = dict()
        self.prior_count_dict = dict()
        self.attrs = df.columns  # names of attributes
        self.size = 0  # number of observations

        # Update count of individual observations
        for attr in self.attrs:
            for attr_val, label in zip(df[attr], y):
                event = "{}={}".format(attr, attr_val)
                if (event, label) in self.event_count_dict:
                    self.event_count_dict[(event, label)] += 1
                else:
                    self.event_count_dict[(event, label)] = 1

        # Update count of class prior
        for label in y:
            self.size += 1
            if label in self.prior_count_dict:
                self.prior_count_dict[label] += 1
            else:
                self.prior_count_dict[label] = 1

    def conditional_proba(self, attr, attr_val, label):
        """
        The conditional probability is estimated using the frequency counts
        collected during fit():
            P(attr = attr_val | label) =
                count(attr = attr_val, label) / count(label)

        If the combination (attr = attr_val, label) has never been seen in the
        training data, this method returns 0.0 (no smoothing is applied).

        Parameters
        ----------
        attr : str
            Name of the attribute (must be one of the columns seen during fit).

        attr_val : object
            Value of the attribute for which we want the conditional probability.
            This should match the type of the values observed in the training
            DataFrame, for example a string, integer, etc.

        label : object
            Class label for the conditioning event. It must be a label that
            appeared in the training labels y passed to fit().

        Returns
        -------
        conditional_proba : float
            The estimated conditional probability P(attr = attr_val | label).
            - Returns 0.0 if the event (attr = attr_val, label) was never
              observed in the training data.
            - Returns a value in [0, 1] otherwise.

        Notes
        -----
        - This implementation assumes that fit() has already been called, so
          that self.event_count_dict and self.prior_count_dict are available.
        """
        conditional_proba = None
        # ++insert your code below ++ to complete the definition of function
        event = "{}={}".format(attr, attr_val)
        numerator = self.event_count_dict.get((event, label), 0)
        denominator = self.prior_count_dict.get(label, 0)
        conditional_proba = numerator / denominator if denominator else 0.0
        return conditional_proba

    def predict(self, X):
        """
        Predict class labels for a set of examples using the Naive Bayes model.

        For each example (feature vector) in X, this method computes the
        posterior probability of each possible class label using the Naive
        Bayes assumption that attributes are conditionally independent given
        the class.


        The predicted label for each example is the label c with the highest
        computed joint probability.

        Parameters
        ----------
        X : array-like (e.g. numpy.ndarray, list of lists)
            Feature matrix of examples to classify.


        Returns
        -------
        labels : list
            A list of length M containing the predicted class label for each
            example in X.

        Notes
        -----
        - This method assumes that fit() has already been called, so that
          self.prior_count_dict, self.event_count_dict, self.size, and
          self.attrs are initialized.
        - Use the function "conditional_proba()" defined above to compute the
          conditional probability of each (joint) event
        - zip function would be helpful for iterating the events
        """

        labels = []
        for vec in X:
            # ++insert your code below ++ to complete the definition of function
            best_label = None
            best_score = -1.0
            for label, label_count in self.prior_count_dict.items():
                proba = label_count / self.size
                for attr, attr_val in zip(self.attrs, vec):
                    proba *= self.conditional_proba(attr, attr_val, label)
                    if proba == 0.0:
                        break
                if proba > best_score:
                    best_score = proba
                    best_label = label
            labels.append(best_label)
        return labels


if __name__ == "__main__":
    # ======Application of defined NB model
    df = pd.read_csv("./data/golf.csv")
    X = df.drop(columns=["Play"])
    y = df["Play"]

    print("\nAfter feature engineering:")
    print(X)

    kbd = KBinsDiscretizer(3, encode="ordinal")
    X[["Temperature", "Humidity"]] = kbd.fit_transform(X[["Temperature", "Humidity"]])
    print("\nAfter feature engineering:")
    print(X)

    clf = NB()

    clf.fit(X, y)
    pred_labels = clf.predict(X.values)
    mean_acc = accuracy_score(pred_labels, y)

    print("\nEvaluation Results:")
    print(" The predicted label of 5th example is", pred_labels[4])
    print(" The average accuracy score on training data of NB is", round(mean_acc, 3))
