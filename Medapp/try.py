# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:47:04 2022

@author: Nila
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import csv,numpy as np,pandas as pd
import os
from sklearn import tree
from sklearn.tree import export_text



data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)
print ("Acurracy: ", clf_dt.score(x_test,y_test))

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]
dictionary = dict(zip(symptoms,indices))

# build decision tree
clf = DecisionTreeClassifier(max_depth =3, random_state = 42)
clf.fit(x_train, y_train)
#plt the figure, setting a black background
plt.figure(figsize=(30,10), facecolor ='k')
#create the tree plot
feature_names = x.columns
labels = y.unique()
a = tree.plot_tree(clf,
                   #use the feature names stored
                   feature_names = feature_names,
                   #use the class names stored
                   class_names = labels,
                   rounded = True,
                   filled = True,
                   fontsize=14)
#show the plot
plt.show()


test_pred_decision_tree = clf.predict(x_test)

#text based diagram
tree_rules = export_text(clf,

                        feature_names = list(feature_names))

print(tree_rules)



# with open('templates/Testing.csv', newline='') as f:
#         reader = csv.reader(f)
#         symptoms = next(reader)
#         symptoms = symptoms[:len(symptoms)-1]





def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1,1)).transpose()
    return(dt.predict(user_input_label))

"""
print(dosomething(['headache','muscle_weakness','puffy_face_and_eyes','mild_fever','skin_rash']))
prediction = []
for i in range(7):
    pred = dosomething(['headache'])   
    prediction.append(pred) 
    print(prediction)
"""
