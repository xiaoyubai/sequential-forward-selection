## sequential-forward-selection
### Implementation of sequential forward selection algorithm for linear regression

Instead of using RFE to do backward selection, I created a LinearRegression class that implements sequential forward selection, which involves starting with no variables in the model, testing the addition of each variable using a chosen model comparison criterion, adding the variable (if any) that improves the model the most, and repeating this process until none improves the model.

    Modification from backward feature selection from:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/rfe.py
