from cgitb import lookup
from html.entities import name2codepoint
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from csv import reader
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from vecstack import stacking
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# importing machine learning models for prediction
#import xgboost as xgb
 
# importing bagging module
from sklearn.ensemble import BaggingRegressor

from mpl_toolkits.mplot3d import Axes3D

class poseData:
    jointPos =   []
    jointStdev = []
    JointAng =   []
    jointAngStdev = []
    id = 0
    idName = ""

data = pd.read_csv('train-final.csv')
data2 = pd.read_csv("test-final.csv")

#print(data)
#print(data["bye"][0])

newcolumns = []
i=1
while i<21:
    name1 = "joint"+str(i)+"x"
    name2 = "joint"+str(i)+"y"
    name3 = "joint"+str(i)+"z"
    newcolumns.append(name1)
    newcolumns.append(name2)
    newcolumns.append(name3)
    
    i+=1

testfeatures = newcolumns[:]
i=1
while i<21:
    name1 = "joint"+str(i)+"stdevx"
    name2 = "joint"+str(i)+"stdevy"
    name3 = "joint"+str(i)+"stdevz"
    newcolumns.append(name1)
    newcolumns.append(name2)
    newcolumns.append(name3)
    
    i+=1
i=1
while i<21:
    name1 = "joint"+str(i)+"anglex"
    name2 = "joint"+str(i)+"angley"
    name3 = "joint"+str(i)+"anglez"
    newcolumns.append(name1)
    newcolumns.append(name2)
    newcolumns.append(name3)
    
    i+=1
i=1
while i<21:
    name1 = "joint"+str(i)+"anglexstdev"
    name2 = "joint"+str(i)+"angleystdev"
    name3 = "joint"+str(i)+"anglezstdev"
    newcolumns.append(name1)
    newcolumns.append(name2)
    newcolumns.append(name3)
    
    i+=1

featurecolumns = newcolumns[:]
testfeatures = newcolumns[:60] + newcolumns[121:180]

newcolumns.append("posename")
newcolumns.append("poseid")
#print(newcolumns)


data.columns = newcolumns
data2.columns = newcolumns

#Fixing missing values
#data.fillna(data.mean())
def fill_class_mean(frame, colname):
    frame[colname] = frame[colname].fillna(frame.groupby('posename')[colname].transform('mean')) 
    
""" fill_class_mean(data, "joint3y")
fill_class_mean(data, "joint3z")
fill_class_mean(data, "joint4x")
fill_class_mean(data, "joint5z")
fill_class_mean(data, "joint6x")
fill_class_mean(data, "joint6y")


fill_class_mean(data2, "joint3x")
fill_class_mean(data2, "joint3y")
fill_class_mean(data2, "joint3z")
fill_class_mean(data2, "joint4z")
fill_class_mean(data2, "joint5x")
fill_class_mean(data2, "joint5y") """

data["joint3y"]=data2["joint3y"].fillna(data2["joint3y"].mean())
data["joint3z"]=data2["joint3z"].fillna(data2["joint3z"].mean())
data["joint4x"]=data2["joint4x"].fillna(data2["joint4x"].mean())
data["joint5z"]=data2["joint5z"].fillna(data2["joint5z"].mean())
data["joint6x"]=data2["joint6x"].fillna(data2["joint6x"].mean())
data["joint6y"]=data2["joint6y"].fillna(data2["joint6y"].mean())


data2["joint3x"]=data2["joint3x"].fillna(data2["joint3x"].mean())
data2["joint3y"]=data2["joint3y"].fillna(data2["joint3y"].mean())
data2["joint3z"]=data2["joint3z"].fillna(data2["joint3z"].mean())
data2["joint4z"]=data2["joint4z"].fillna(data2["joint4z"].mean())
data2["joint5x"]=data2["joint5x"].fillna(data2["joint5x"].mean())
data2["joint5y"]=data2["joint5y"].fillna(data2["joint5y"].mean())

newdata = data[data.columns[:240]]
#newdata = pd.DataFrame()
#print(newdata)
#newdata.drop("posename", inplace=True, axis=1)
#   newdata.drop("poseid", inplace=True, axis=1)


#print(newdata)
#print(data)


NUM_OF_JOINTS = 20
JOINTSET_SIZE = NUM_OF_JOINTS*3

def read_pose_data(file_name):
    def extract_coords(offset, row):
        range_end = JOINTSET_SIZE + offset
        flat_list = row[offset:range_end]
        for I in range(0,len(flat_list)):
            try:
                flat_list[I] = float(flat_list[I])
            except ValueError:
                flat_list[I] = None

        return numpy.reshape(flat_list,(NUM_OF_JOINTS,3));

    pose_list = []

    with open(file_name, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)

    for row in list_of_rows:
        pose = poseData()
        pose.jointPos      = extract_coords(0,row)
        pose.jointStdev    = extract_coords(JOINTSET_SIZE,row)
        pose.jointAng      = extract_coords(JOINTSET_SIZE*2,row)
        pose.jointAngStdev = extract_coords(JOINTSET_SIZE*3,row)

        pose.idName                   = row[JOINTSET_SIZE*4]
        pose.id                       = row[JOINTSET_SIZE*4 + 1]
        pose_list.append(pose)

    return pose_list


# This function plots a given pose in 3 dimensions
def showPose(posenum):
    fig = plt.figure(figsize=(4,4))
    fig2 = plt.figure(figsize=(4,4))
    fig3 = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection=Axes3D.name)
    ax3 = fig3.add_subplot(111, projection=Axes3D.name)
    j=1
    while j<21:
        lookup1 = "joint"+str(j)+"x"
        lookup2 = "joint"+str(j)+"y"
        lookup3 = "joint"+str(j)+"z"
        lookup4 = "joint"+str(j)+"stdevx"
        lookup5 = "joint"+str(j)+"stdevy"
        lookup6 = "joint"+str(j)+"stdevz"
        ax.scatter(data[lookup1][posenum], data[lookup2][posenum], data[lookup3][posenum])
        ax2.scatter(data[lookup1][posenum]+data[lookup4][posenum], data[lookup2][posenum]+data[lookup5][posenum], data[lookup3][posenum]+data[lookup6][posenum])
        ax3.scatter(data[lookup1][posenum]-data[lookup4][posenum], data[lookup2][posenum]-data[lookup5][posenum], data[lookup3][posenum]-data[lookup6][posenum])
        j+=1

    print(data["posename"][posenum])
    plt.show()

#testlist = read_pose_data("train-final.csv")
#pose = testlist[465]

#fig = plt.figure(figsize=(4,4))
#fig2 = plt.figure(figsize=(4,4))

#ax = fig.add_subplot(111, projection='3d')
#ax2 = fig2.add_subplot(111, projection=Axes3D.name) 

""" for point in pose.jointPos:
    ax.scatter(point[0], point[1], point[2])
    ax2.scatter(point[0], point[1], point[2]) """


#plt.show()
#print(pose.idName)

#showPose(5)

# NOW LET'S APPLY SOME ALGORITHMS
X_train = data[featurecolumns] # Features
Y_train = data.posename # Target variable
Y2_train = data.poseid

X_test = data2[featurecolumns] # Features
Y_test = data2.posename # Target variable
Y2_test = data2.poseid


def treeClass():
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=120)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,Y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

def kNNClass(k):
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    #Train the model using the training sets
    knn.fit(X_train, Y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

def svmClass():
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, Y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

def forestClass():
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=1000, max_depth=120)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,Y_train)

    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


def mlpClass():
    classifier = MLPClassifier(hidden_layer_sizes=(200,150,100,50),activation = 'relu',solver='adam', max_iter=600,random_state=1)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    #Importing Confusion Matrix
    
    #Comparing the predictions against the actual observations in y_val
    #cm = confusion_matrix(y_pred, Y_test)

    #Printing the accuracy
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

def ensembleClass1():
    model_1 = MLPClassifier(hidden_layer_sizes=(200,150,100,50), max_iter=600,activation = 'relu',solver='adam',random_state=1)
    model_2 = svm.SVC(kernel='linear')
    model_3 = RandomForestClassifier(n_estimators=1000)
    model_4 = KNeighborsClassifier(n_neighbors = 1)
    model_5 = RandomForestClassifier(n_estimators=1000)
    model_6 = DecisionTreeClassifier(criterion="entropy", max_depth=120)
    model_7 = LogisticRegression(max_iter = 600)
    model_8 = svm.SVC(kernel='linear')
    final_model = VotingClassifier(
    estimators=[('mlp', model_1), ('svm', model_2), ('rf', model_3), ('knn', model_4), ('dt', model_6), ('lr', model_7)], voting='hard', weights=[1,2,3,0.25,0.25,1], n_jobs=-1)
 
    # training all the model on the train dataset
    final_model.fit(X_train, Y_train)
 
    # predicting the output on the test dataset
    pred_final = final_model.predict(X_test)
    print("Accuracy for voting:",metrics.accuracy_score(Y_test, pred_final))
    
def ensembleClass2():
    model_1 = KNeighborsClassifier(n_neighbors=1)
    model_2 = svm.SVC(kernel='linear')
    model_3 = DecisionTreeClassifier(criterion="entropy", max_depth=240)
    final_model = VotingClassifier(
    estimators=[('knn', model_1), ('svm', model_2), ('dt', model_3)], voting='hard', weights=[1, 1, 1])
 
    # training all the model on the train dataset
    final_model.fit(X_train, Y_train)
 
    # predicting the output on the test dataset
    pred_final = final_model.predict(X_test)
    print("Accuracy for voting 2:",metrics.accuracy_score(Y_test, pred_final))

def ensembleStack():
    level0 = list()
#    level0.append(('lr', LogisticRegression(max_iter = 1000)))
    level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))
    level0.append(('svm', svm.SVC(kernel='linear', C=0.8)))
    level0.append(('rf', RandomForestClassifier(n_estimators=1000, max_depth=1100)))
    level0.append(('mlp', MLPClassifier(hidden_layer_sizes=(200,150,100,50), max_iter=600,activation = 'relu',solver='adam',random_state=1)))
    level0.append(('dt', DecisionTreeClassifier(max_depth = 130, criterion='gini')))

    level1 = RandomForestClassifier(n_estimators = 1000, max_depth=1100)

    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print("Accuracy for stack:",metrics.accuracy_score(Y_test, pred))
    
def boosterClass():
    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth = 100)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print("Accuracy for booster:",metrics.accuracy_score(Y_test, pred))
    
def adaBoost():
    
    svc=SVC(kernel='linear')
    # Create adaboost classifer object
    abc = AdaBoostClassifier(n_estimators=500, base_estimator=LogisticRegression(max_iter=1000),learning_rate=1, algorithm='SAMME.R', random_state=0)

    # Train Adaboost Classifer
    model = abc.fit(X_train, Y_train)

    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy for Adaboost:",metrics.accuracy_score(Y_test, y_pred))



def ensembleBagging():
    model = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=1000, max_depth=1100), n_estimators=10, random_state=0, n_jobs=-1)
 
    # training model
    model.fit(X_train, Y_train)
    
    # predicting the output on the test dataset
    pred = model.predict(X_test)
    print("Accuracy for bagging:",metrics.accuracy_score(Y_test, pred)) 

def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

def plotPerformance():
    models = {}
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier(n_neighbors=1)
    models['svm'] = svm.SVC(kernel='linear')
    models['rf']=RandomForestClassifier(n_estimators=100)

    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X_test, Y_test)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()

#mlpClass()
#forestClass()
#ensembleClass1()
#ensembleClass2()
#ensembleStack() 
#boosterClass()
ensembleBagging()
adaBoost()