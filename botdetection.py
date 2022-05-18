
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

file= '/content/training_data_2_csv_UTF.csv'
training_data = pd.read_csv(file)
bots = training_data[training_data.bot==1]
nonbots = training_data[training_data.bot==0]

"""Feature Independence using Spearman correlation"""

traning_data.corr(method='spearman')

plt.figure(figsize=(8,4))
sns.heatmap(df.corr(method='spearman'), cmap='coolwarm', annot=True)
plt.tight_layout()
plt.show()

"""Performing Feature Engineering"""

bag_of_words_bot = r'zero bot|Demo|Free|Act Now|Access for Free|b0t|0%|Access Now|Bargain|Believe ME|
                           r'expos|kill|clit|bbb|butt|fuck|fake|anony|free|virus|funky|RNA|kuck|jargon'            
training_data['screen_name_binary'] = training_data.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['name_binary'] = training_data.name.str.contains(bag_of_words_bot, case=False, na=False)
training_data['description_binary'] = training_data.description.str.contains(bag_of_words_bot, case=False, na=False)
training_data['status_binary'] = training_data.status.str.contains(bag_of_words_bot, case=False, na=False)

"""Performing Feature Extraction"""

training_data['listed_count_binary'] = (training_data.listed_count>20000)==False
features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

"""Implementing Different Models

Decision Tree Classifier

"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]
dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=50, min_samples_split=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
dt = dt.fit(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)
print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
print("For test dataset decision tree")
print(classification_report(y_test, y_pred_test))
print("\nFor train dataset")
print(classification_report(y_train, y_pred_train))

print(training_data[features])

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid' : False})
scores_train = dt.predict_proba(X_train)
scores_test = dt.predict_proba(X_test)
y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])
for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])    
fpr_dt_train, tpr_dt_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_dt_test, tpr_dt_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)
plt.plot(fpr_dt_train, tpr_dt_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
plt.plot(fpr_dt_test, tpr_dt_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
plt.title("Decision Tree ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')

"""Multinomial Naive Bayes Classifier"""

from sklearn.naive_bayes import MultinomialNB
X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]
mnb = MultinomialNB(alpha=0.0009)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
mnb = mnb.fit(X_train, y_train)
y_pred_train = mnb.predict(X_train)
y_pred_test = mnb.predict(X_test)
print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))

sns.set_style("whitegrid", {'axes.grid' : False})
scores_train = mnb.predict_proba(X_train)
scores_test = mnb.predict_proba(X_test)
y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])
for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])    
fpr_mnb_train, tpr_mnb_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_mnb_test, tpr_mnb_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)
plt.plot(fpr_mnb_train, tpr_mnb_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_mnb_train, tpr_mnb_train))
plt.plot(fpr_mnb_test, tpr_mnb_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_mnb_test, tpr_mnb_test))
plt.title("Multinomial NB ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')

"""Result: Clearly, Multinomial Niave Bayes peforms comparatively poorly and is not a good choice as the Train AUC is just 0.7 and Test is 0.69.

Random Forest Classifier
"""

from sklearn.ensemble import RandomForestClassifier
X = training_data[features].iloc[:,:-1]
y = training_data[features].iloc[:,-1]
rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=100, min_samples_split=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
rf = rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
print("Trainig Accuracy: %.5f" %accuracy_score(y_train, y_pred_train))
print("Test Accuracy: %.5f" %accuracy_score(y_test, y_pred_test))

print("For test dataset random forest")
print(classification_report(y_test, y_pred_test))
print("\nFor train dataset")
print(classification_report(y_train, y_pred_train))

sns.set_style("whitegrid", {'axes.grid' : False})
scores_train = rf.predict_proba(X_train)
scores_test = rf.predict_proba(X_test)
y_scores_train = []
y_scores_test = []
for i in range(len(scores_train)):
    y_scores_train.append(scores_train[i][1])
for i in range(len(scores_test)):
    y_scores_test.append(scores_test[i][1])    
fpr_rf_train, tpr_rf_train, _ = roc_curve(y_train, y_scores_train, pos_label=1)
fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, y_scores_test, pos_label=1)
plt.plot(fpr_rf_train, tpr_rf_train, color='darkblue', label='Train AUC: %5f' %auc(fpr_rf_train, tpr_rf_train))
plt.plot(fpr_rf_test, tpr_rf_test, color='red', ls='--', label='Test AUC: %5f' %auc(fpr_rf_test, tpr_rf_test))
plt.title("Random ForestROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')

import pickle
DIR = "./drive/MyDrive/"
Forest_pickle = "%sforest.pkl" % DIR
with open(Forest_pickle, 'wb') as file:
    pickle.dump(rf, file)

"""Our Classifier"""

import sklearn.metrics as metrics
class twitter_bot(object):
    def __init__(self):
        pass
    def perform_train_test_split(df):
        msk = np.random.rand(len(df)) < 0.75
        train, test = df[msk], df[~msk]
        X_train, y_train = train, train.iloc[:,-1]
        X_test, y_test = test, test.iloc[:, -1]
        return (X_train, y_train, X_test, y_test)
    def solve(df):
        train_df = df.copy()
        train_df['id'] = train_df.id.apply(lambda x: int(x))
        train_df['followers_count'] = train_df.followers_count.apply(lambda x: 0 if x=='None' else int(x))
        train_df['friends_count'] = train_df.friends_count.apply(lambda x: 0 if x=='None' else int(x))
        if train_df.shape[0]>600:          
            bag_of_words_bot = r'zero bot|Demo|Free|Act Now|Access for Free|b0t|0%|Access Now|Bargain|Believe ME|
                           r'expos|kill||bbb|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon'                     
        else: 
            bag_of_words_bot = r'bot|b0t|free|mishear|updates every'
        train_df['verified'] = train_df.verified.apply(lambda x: 1 if ((x == True) or x == 'TRUE') else 0)
        condition = ((train_df.name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.description.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)) |
                     (train_df.status.str.contains(bag_of_words_bot, case=False, na=False))
                     )
        predicted_df = train_df[condition]
        predicted_df.bot = 1
        predicted_df = predicted_df[['id', 'bot']]
        verified_df = train_df[~condition]
        condition = (verified_df.verified == 1)
        predicted_df1 = verified_df[condition][['id', 'bot']]
        predicted_df1.bot = 0
        predicted_df = pd.concat([predicted_df, predicted_df1]) 
        predicted_df1 = listed_count_df[~condition][['id', 'bot']]
        predicted_df1.bot = 0 # these all are nonbots
        predicted_df = pd.concat([predicted_df, predicted_df1])
        return predicted_df
    def get_predicted_and_true_values(features, target):
        y_pred, y_true = twitter_bot.bot_prediction_algorithm(features).bot.tolist(), target.tolist()
        return (y_pred, y_true)
    def getaccuracy(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        y_pred_train, y_true_train = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        train_acc = metrics.accuracy_score(y_pred_train, y_true_train)
        y_pred_test, y_true_test = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        test_acc = metrics.accuracy_score(y_pred_test, y_true_test)
        return (train_acc, test_acc)
    def roc_curve(df):
        (X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
        y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
        scores = np.linspace(start=0.01, stop=0.9, num=len(y_true))
        fpr_train, tpr_train, threshold = metrics.roc_curve(y_pred_train, scores, pos_label=0)
        plt.plot(fpr_train, tpr_train, label='Train AUC: %5f' % metrics.auc(fpr_train, tpr_train), color='darkblue')
        y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
        scores = np.linspace(start=0.01, stop=0.9, num=len(y_true))
        fpr_test, tpr_test, threshold = metrics.roc_curve(y_pred_test, scores, pos_label=0)
        plt.plot(fpr_test,tpr_test, label='Test AUC: %5f' %metrics.auc(fpr_test,tpr_test), ls='--', color='red')
        #Misc
        plt.xlim([-0.1,1])
        plt.title("Reciever Operating Characteristic (ROC)")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend(loc='lower right')
        plt.show()
if __name__ == '__main__':
    train_df = pd.read_csv('/content/training_data_2_csv_UTF.csv')
    test_df = pd.read_csv('/content/test_data_4_students.csv',encoding = 'unicode_escape',sep='\t')
    print("Train Accuracy: ", twitter_bot.getaccuracy(train_df)[0])
    print("Test Accuracy: ", twitter_bot.getaccuracy(train_df)[1])
    predicted_df = twitter_bot.solve(test_df)
 
    twitter_bot.roc_curve(train_df)
