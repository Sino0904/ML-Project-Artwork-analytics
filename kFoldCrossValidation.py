

def kFoldCrossValidation(data_set,data_set_labels,k = 10):
#K-Fold cross validation method, to be used to evalute different models for hyperparameter tuning.

    folds = []
    folds_labels = []
    for i in xrange(k):
        folds.append(data_set[i::k])#Data set being divided in k-subsets
        folds_labels.append(data_set_labels[i::k])#Labels for dataset being divided in k-subsets

    for i in xrange(k):
        #Formation of validation set, its labels and the subsequent Training set and its labels
        validation_set = folds[i] #validation set
        validation_set_labels = folds_labels[i] #validation set labels
        #Training set
        for s in folds:
		if s is not validation_set:
			train_set = []
			for item in s:
				train_set.append(item)
	#Training set labels			
	for s_labels in folds_labels:
		if s_labels is not validation_set_labels:
			train_set_labels = []
			for item_labels in s_labels:
				train_set_labels.append(item_labels)
	yield train_set,validation_set,train_set_labels,validation_set_labels

    

    
