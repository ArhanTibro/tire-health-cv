"""
Split data/raw/<class>/ into data/processed/{train,val,test}/<class>/,
stratified per class (not a global random split — see project notes on why).

TODO (next phase): implement using split-folders or sklearn's
train_test_split with stratify=labels.
"""
