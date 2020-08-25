import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import joblib
# 加载数据
x_data_train = joblib.load('save/x_data_train')
x_data_test = joblib.load('save/x_data_test')
y_data_train = joblib.load('save/y_data_train')
y_data_test = joblib.load('save/y_data_test')
x_test_1 = joblib.load('save/x_test_1')
x_test_2 = joblib.load('save/x_test_2')
y_test_1 = joblib.load('save/y_test_1')
y_test_2 = joblib.load('save/y_test_2')

model_names = ['DecisionTreeClassifier', 'LinearSVC', 'RandomForestClassifier', 'MultinomialNB',
               'LinearDiscriminantAnalysis', 'LogisticRegression', 'GradientBoostingClassifier',
               'AdaBoostClassifier', 'GaussianNB', 'QuadraticDiscriminantAnalysis','XGBClassifier',
               'KNeighborsClassifier', 'VotingClassifier']
dict = {}
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
# 遍历，装载模型
for model_name in model_names:
    if model_name == 'KNeighborsClassifier':
        print(f'Loading {model_name}...')
        print(f'Testing {model_name}...')
        # model = joblib.load(f'save/{model_name}')
        # predictions = model.predict(x_test_1)
        # joblib.dump(predictions, 'save/knn_predictions')
        predictions = joblib.load('save/knn_predictions')
    else:
        print(f'Loading {model_name}...')
        print(f'Testing {model_name}...')
        model = joblib.load(f'save/{model_name}')
        predictions = model.predict(x_test_1)
    y = y_test_1
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    value = [accuracy, precision, recall, f1]
    dict[f'{model_name}'] = value
    dict['accuracy'] = accuracy_list
    dict['precision'] = precision_list
    dict['recall'] = recall_list
    dict['f1'] = f1_list
print(dict['DecisionTreeClassifier'])