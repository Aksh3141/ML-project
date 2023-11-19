import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
import pylab as pl
import seaborn as sns

warnings.filterwarnings("ignore")

class classification():
     def __init__(self,path=' ',clf_opt='lr',no_of_selected_features=None):
        self.path = path
        self.clf_opt=clf_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features) 

# Selection of classifiers  
     def classification_pipeline(self):    
    # AdaBoost 
        if self.clf_opt=='ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = svm.SVC(kernel='linear', class_weight='balanced',probability=True)              
            be2 = LogisticRegression(solver='liblinear',class_weight='balanced') 
            be3 = DecisionTreeClassifier(max_depth=50,criterion='entropy',max_features='sqrt',ccp_alpha=0.009)
#            clf = AdaBoostClassifier(algorithm='SAMME',n_estimators=100)            
            clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=100)
            clf_parameters = {
            'clf__base_estimator':(be3,),
            'clf__random_state':(1,2,3,5),
            }

     # Bagging classifier
        elif self.clf_opt=='bag':
             print('\n\t### Training Bagging Classifier ### \n')
             clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=20,criterion='entropy',max_features='sqrt',ccp_alpha=0.009), random_state=0)
             clf_parameters = {
             'clf__max_samples':(300,475,500,600),
             'clf__n_estimators':(410,450,460,470)
             }


    # Neural Networks
        elif self.clf_opt == 'nn':
             print('\n\t### Multilayer Perceptron classifier ### \n')
             clf = MLPClassifier(solver='lbfgs')
             clf_parameters = {
               'clf__hidden_layer_sizes':((500,3),),
               'clf__random_state':(1,)
                  }

    # K-nearest Neighbors
        elif self.clf_opt == 'knn':
             print('\n\t### K-Nearest Neighbors ### \n')
             clf = KNeighborsClassifier()
             clf_parameters = {
               'clf__algorithm':('auto', 'ball_tree', 'kd_tree'),
               'clf__n_neighbors':(4,5,6,7),
               'clf__metric':('euclidean','manhattan')

                  }
    # Decision Tree
        elif self.clf_opt=='dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40) 
            clf_parameters = {
            'clf__criterion':('gini', 'entropy'), 
            'clf__max_features':('auto', 'sqrt', 'log2'),
            'clf__max_depth':(10,40,45,60),
            'clf__ccp_alpha':(0.009,0.01,0.05,0.1),
            } 
    # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
            'clf__random_state':(0,10),
            } 
        
    # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
            'clf__alpha':(0,1,10,50),
            }            
    # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
            'clf__criterion':('entropy',),       
            'clf__n_estimators':(40,50,60,70),
            'clf__max_depth':(20,30,60,65),
            }          
    # Support Vector Machine  
        elif self.clf_opt=='svm': 
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced',probability=True)  
            clf_parameters = {
            'clf__C':(1,0.1,5),
            'clf__kernel':('linear','rbf','poly'),
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)        
        return clf,clf_parameters    
 
# Statistics of individual classes
     def get_class_statistics(self,labels):
        class_statistics=Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t'+str(item)+'\t\t\t'+str(class_statistics[item]))
       
# Load the data 
     def get_data(self,filename,filename2):

 
    # Load the file using Pandas       
        reader=pd.read_csv(self.path+filename)  
        feader=pd.read_csv(self.path+filename2)

        #ONE HOT ENCODING
        df3 = pd.get_dummies(reader, columns = ['ifo'])
        df3.drop(["id","GPStime"],axis=1,inplace=True)
        data=df3

        #using label encoder to convert output labels to numeric values
        label_encoder = LabelEncoder()
        feader["class_labels"] = label_encoder.fit_transform(feader['class_labels'])
        labels=feader.iloc[:,1]

        # normalizing the data using min max scalar
        scaler = MinMaxScaler()
        s_data = scaler.fit_transform(data)

        # Utilizing SMOTE (Synthetic Minority Over-sampling Technique) to address dataset imbalance through oversampling
        smote = SMOTE(sampling_strategy='auto', random_state=42,k_neighbors =2)
        X, y = smote.fit_resample(s_data, labels)

        return X, y
    
# Classification using the Gold Statndard after creating it from the raw text    
     def classification(self):  
   # Get the data
        data,labels=self.get_data('glitch_trn_data.csv','glitch_trn_class_labels.csv')
        data=np.asarray(data)

# Experiments using training data only during training phase (dividing it into training and validation set)
        skf = StratifiedKFold(n_splits=5)
        predicted_class_labels=[]; actual_class_labels=[]; 
        count=0; probs=[];
        for train_index, test_index in skf.split(data,labels):
            X_train=[]; y_train=[]; X_test=[]; y_test=[]
            for item in train_index:
                X_train.append(data[item])
                y_train.append(labels[item])
            for item in test_index:
                X_test.append(data[item])
                y_test.append(labels[item])
            count+=1                
            print('Training Phase '+str(count))

            
            clf,clf_parameters=self.classification_pipeline()
            pipeline = Pipeline([('clf', clf),])

            # GRID SEARCH FOR hyper parameter tuning
            grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10)          
            grid.fit(X_train,y_train)     
            clf= grid.best_estimator_  
            print('\n\n The best set of parameters of the pipiline are: ')
            print(clf)     
            predicted=clf.predict(X_test)  
            predicted_probability = clf.predict_proba(X_test) 
            for item in predicted_probability:
                probs.append(float(max(item)))
            for item in y_test:
                actual_class_labels.append(item)
            for item in predicted:
                predicted_class_labels.append(item)           
        confidence_score=statistics.mean(probs)-statistics.variance(probs)
        confidence_score=round(confidence_score, 3)
        print ('\n The Probablity of Confidence of the Classifier: \t'+str(confidence_score)+'\n') 


        
       
    # Evaluation
        class_names=list(Counter(labels).keys())
        class_names = [str(x) for x in class_names] 
        # print('\n\n The classes are: ')
        # print(class_names)      
       
        print('\n ##### Classification Report on Training Data ##### \n')
        print(classification_report(actual_class_labels, predicted_class_labels, target_names=class_names))        
                
        pr=precision_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n Precision:\t'+str(pr)) 
        
        rl=recall_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n Recall:\t'+str(rl))
        
        fm=f1_score(actual_class_labels, predicted_class_labels, average='macro') 
        print ('\n F1-Score:\t'+str(fm))

        cm=confusion_matrix(actual_class_labels, predicted_class_labels)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax,cmap="Blues")
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels') 
        ax.set_title('Confusion Matrix')
        plt.show()



clf=classification('/home/aksh/Desktop/ML-project/', clf_opt='rf',no_of_selected_features=7)

clf.classification()



