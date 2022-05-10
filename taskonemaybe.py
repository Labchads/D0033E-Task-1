from cgitb import lookup
from html.entities import name2codepoint
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from csv import reader

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

""" j=1
while j<21:
    column = "joint"+str(j)+"coords"
    newdata[column] = data[data.columns[j-1:j+2]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
    )
    
    j+=1

j=1
while j<21:
    column = "joint"+str(j)+"coordstdev"
    newdata[column] = data[data.columns[20:j+22]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
    )
    
    j+=1

j=1
while j<61:
    column = "joint"+str(j)+"angles"
    newdata[column] = data[data.columns[j-1:j+2]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
    )
    
    j+=1

j=1
while j<81:
    column = "joint"+str(j)+"anglesstdev"
    newdata[column] = data[data.columns[j-1:j+2]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
    )
    
    j+=1 """


#print(newdata)
#print(data)

""" reformedData = {}

dataKeys = data.keys()
i=0
jointNr = 1
while i<60:
    index = "joint"+str(jointNr)
    reformedData[jointNr]=[[data[dataKeys[i]], data[dataKeys[i+1]], data[dataKeys[i+2]]]]
    i = i+3 """



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

showPose(5)
