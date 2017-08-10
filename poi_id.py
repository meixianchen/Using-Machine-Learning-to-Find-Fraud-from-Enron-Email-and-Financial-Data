#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments',  'bonus', \
    'deferred_income', 'total_stock_value', 'expenses', \
    'exercised_stock_options', 'long_term_incentive', \
    'restricted_stock', 'director_fees',\
    'percent_from_this_person_to_poi',\
    'percent_from_poi_to_this_person', 'percent_shared_receipt'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "total person: ", len(data_dict)
count_poi = 0

for name in data_dict:
    if data_dict[name]['poi'] == True:
        count_poi += 1


print "number of poi:",count_poi

### Task 2: Remove outliers

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E') # poi == false, the other features NaN


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#
for point in my_dataset:

    if 'NaN' in [my_dataset[point]['from_poi_to_this_person'],my_dataset[point]['to_messages']]:
        my_dataset[point]['percent_from_poi_to_this_person'] = 'NaN'
    else:
        my_dataset[point]['percent_from_poi_to_this_person'] = float(my_dataset[point]['from_poi_to_this_person'])/float(my_dataset[point]['to_messages'])

    if 'NaN' in [my_dataset[point]['from_this_person_to_poi'],my_dataset[point]['from_messages']]:
        my_dataset[point]['percent_from_this_person_to_poi'] = 'NaN'
    else:
        my_dataset[point]['percent_from_this_person_to_poi'] = float(my_dataset[point]['from_this_person_to_poi'])/float(my_dataset[point]['from_messages'])

    if 'NaN' in [my_dataset[point]['shared_receipt_with_poi'],my_dataset[point]['to_messages']]:
        my_dataset[point]['percent_shared_receipt'] = 'NaN'
    else:
        my_dataset[point]['percent_shared_receipt'] = float(my_dataset[point]['shared_receipt_with_poi'])/float(my_dataset[point]['to_messages'])




### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


from sklearn.feature_selection import SelectKBest

skb = SelectKBest()
skb.fit(features,labels)


for i, s in enumerate(skb.scores_):
    print features_list[i+1],s

from sklearn.decomposition import PCA
pca = PCA(n_components=3)


### Task 4: Try a varity of classifiers

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()

from sklearn.svm import SVC
svc = SVC()

from sklearn.neighbors import NearestCentroid
nc = NearestCentroid()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# try different clf 

from sklearn.pipeline import Pipeline
estimators = [('scaler', min_max_scaler),('SKB', skb),('reduce_dim', pca), ('clf', nc)]
pipeline = Pipeline(estimators)


### Task 5: Tune your classifier to achieve better than .3 precision and recall

from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(labels, n_iter=10, random_state = 42)
param_spaces = {
    'SKB__k':[5,8,10,12],
    'reduce_dim__n_components':[2,3,4,5],
# tune    NearestCentroid
    'clf__metric': ['euclidean','manhattan']
}

gs = GridSearchCV(pipeline,param_grid = param_spaces, n_jobs = -1,cv = cv, scoring = 'precision',verbose=10)
gs.fit(features, labels)

print gs.best_params_
clf = gs.best_estimator_




### Task 6: Dump classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, features_list)
