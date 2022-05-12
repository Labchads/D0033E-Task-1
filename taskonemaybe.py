from cgitb import lookup
from html.entities import name2codepoint
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from csv import reader
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

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

newcolumns.append("posename")
newcolumns.append("poseid")
#print(newcolumns)


data.columns = newcolumns
data2.columns = newcolumns

#Fixing missing values
#data.fillna(data.mean())
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

X_test = data2[featurecolumns] # Features
Y_test = data2.posename # Target variable


def treeClass():
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=120)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,Y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

def kNNClass():
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=2)

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
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,Y_train)

    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

def mlpClass():
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    #Importing Confusion Matrix
    
    #Comparing the predictions against the actual observations in y_val
    #cm = confusion_matrix(y_pred, Y_test)

    #Printing the accuracy
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    #print("Accuracy of MLPClassifier : ", cm)
    """ clf = MLPClassifier(random_state=1, max_iter=500).fit(X_train, Y_train)
    clf.predict_proba(X_test)
    
    clf.predict(X_test)
    
    clf.score(X_test, Y_test) """

mlpClass()