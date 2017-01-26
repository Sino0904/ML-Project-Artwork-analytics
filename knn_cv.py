
#K-Nearest neighbors with hyperparameter tuning using Grid search with k-fold cross validation

from sklearn.metrics import accuracy_score
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from kFoldCrossValidation import kFoldCrossValidation
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import SVC
import time
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def knn_cv(featuresTrain,labelsTrain,labelsTrain2,featuresTest,labelsTest,labelsTest2):

    #NumberOfNeighbors = [3,4,5,6,7,8,9,10] # Range of hyperparameters over which the model would be trained to find the optimal hyperparameter value
    n_estimators = [10,50,100,150, 200, 250,350,500]
    #n_estimators = [0.1,0.5,1.0,5.0,10.0,50.0,100.0]
    model_avg_score = []
    trainingError = []
    
    #for a in NumberOfNeighbors:
    for a in n_estimators:
        #model = KNeighborsClassifier(n_neighbors = a)
        model = RandomForestClassifier(n_estimators = a)
        #model = OneVsRestClassifier(SVC(C = a,kernel='linear'))
        score = []
        #Evaluating each new model with different hyperparameter to find the optimum value, using k fold cross validation
        for train_set,val_set,train_set_labels,val_set_labels in kFoldCrossValidation(featuresTrain,labelsTrain,10):
            model.fit(train_set,train_set_labels)
            pred_val = model.predict(val_set)
            score.append(accuracy_score(val_set_labels, pred_val)) 
        average_score = float(sum(score))/len(score) # accuracy score of the classifier for the given hyperparameter
        model_avg_score.append(average_score)#accuracy score for each generated classifier model
        pred_1 = model.predict(featuresTrain)
        trainingError.append(1 - accuracy_score(labelsTrain, pred_1))# training error

    #optimum_hyperparameterVal = NumberOfNeighbors[model_avg_score.index(max(model_avg_score))]
    optimum_hyperparameterVal = n_estimators[model_avg_score.index(max(model_avg_score))]#Optimum hyperparameter, for which the max accuracy score is obtained

    #final_model = KNeighborsClassifier(n_neighbors = optimum_hyperparameterVal)
    final_model = RandomForestClassifier(n_estimators = optimum_hyperparameterVal)
    #final_model = OneVsRestClassifier(SVC(C = optimum_hyperparameterVal,kernel='linear'))
    final_model.fit(featuresTrain,labelsTrain)#Training
    
    predicted = final_model.predict(featuresTest)#Testing
    #predicted2 = label_binarize(predicted, classes=[1,2,3])
    accuracy = accuracy_score(labelsTrain,predicted)

    # Compute ROC curve and ROC area for each class
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(labelsTrain[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))


    print "Accuracy:",accuracy
    print "Best hyperparameter:",optimum_hyperparameterVal


    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Random Forest)')
    plt.legend(loc="lower right")

    plt.figure(2)
    plt.plot(n_estimators,model_avg_score)
    plt.xlabel('Hyperparameter values')
    plt.ylabel('Accuracy Score')
    plt.title('Hyperparameter vs Accuracy (Random Forest)')

    plt.show()

    
    return accuracy,optimum_hyperparameterVal
