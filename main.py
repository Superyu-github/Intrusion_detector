import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


# 用于打印评分的函数
def print_score(y, predictions):
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    print('-------------score-------------')
    print(f'The accuracy is: {accuracy}')
    print(f'The precision is: {precision}')
    print(f'The recall is: {recall}')
    print(f'The f1 is: {f1}')
    print('-------------------------------')


# 用于训练和保存模型的函数
def fit_model(model, name ):
    print(f'{name} Training...')
    model.fit(x_data_train, y_data_train)
    joblib.dump(model, f'save/{name}')
    print(f"#{name}")
    predictions = model.predict(x_test_1)
    print_score(y_test_1, predictions)

# 数据预处理
# 表头
name = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
]
# 读取文件，文件'kddcup.data_10_percent_corrected'是训练集，'corrected'是测试集
print('Reading the file...')
data = pd.read_csv('kddcup.data_10_percent_corrected', names=name)
test = pd.read_csv('corrected', names=name)
# 预处理
print('Preprocessing...')
# 将class下所有非normal.的标签改为attack.
test['class'].loc[test['class'] != 'normal.'] = 'attack.'
data['class'].loc[data['class'] != 'normal.'] = 'attack.'
# 将文字标签编码
data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])
test.iloc[:, -1] = LabelEncoder().fit_transform(test.iloc[:, -1])
data.iloc[:, 1:4] = OrdinalEncoder().fit_transform(data.iloc[:, 1:4])
test.iloc[:, 1:4] = OrdinalEncoder().fit_transform(test.iloc[:, 1:4])
# 划分训练集和测试集
x_data = data.drop(columns=['class'])
y_data = data['class']
x_test = test.drop(columns=['class'])
y_test = test['class']
# 划分用于训练的数据(只用了x/y_date_train,其他两个舍弃了)，此举用于控制用于训练的数据的数量，因总数据较多，个别模型训练耗时太长，
# 此时可以调高test_size减少训练时间
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1)
# 将corrected分成两份，一份当作测试集调参，另一份当作“未知数据”用于展示最终效果
x_test_1, x_test_2, y_test_1, y_test_2 = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
# 保存数据
joblib.dump(x_data_train, 'save/x_data_train')
joblib.dump(x_data_test, 'save/x_data_test')
joblib.dump(y_data_train, 'save/y_data_train')
joblib.dump(y_data_test, 'save/y_data_test')
joblib.dump(x_test_1, 'save/x_test_1')
joblib.dump(x_test_2, 'save/x_test_2')
joblib.dump(y_test_1, 'save/y_test_1')
joblib.dump(y_test_2, 'save/y_test_2')


# 选择模型进行训练

# 1.决策树模型
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=40, min_samples_split=5)
fit_model(model, 'DecisionTreeClassifier')

# 2.线性支持向量机模型
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=10)
fit_model(model, 'LinearSVC')

# 3.随机森林模型
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=10, n_estimators=106, max_depth=31, n_jobs=-1)
fit_model(model, 'RandomForestClassifier')

# 4.高斯朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
fit_model(model, 'GaussianNB')

# 5.多项式朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
fit_model(model, 'MultinomialNB')

# 6.线性鉴别分析模型
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
fit_model(model, 'LinearDiscriminantAnalysis')

# 7.Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
model = QuadraticDiscriminantAnalysis(priors=1)
fit_model(model, 'QuadraticDiscriminantAnalysis')

# 8.逻辑回归模型
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1200, n_jobs=-1)
fit_model(model, 'LogisticRegression')

# 9.GBDT(Gradient Boosting Decision Tree)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.01, max_depth=15, min_samples_leaf=80)
fit_model(model, 'GradientBoostingClassifier')

# 10.KNeighborsClassifier模型
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(leaf_size=27, n_neighbors=4, n_jobs=-1)
fit_model(model, 'KNeighborsClassifier')

# 11.AdaBoost模型
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=16)
fit_model(model, 'AdaBoostClassifier')

# 12.xgboost模型
from xgboost import XGBClassifier
model = XGBClassifier(max_depth=3, subsample=0.8, colsample_bytree=0.8)
fit_model(model, 'XGBClassifier')

# 13.voting_classify
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
clf1 = GradientBoostingClassifier(n_estimators=200)
clf2 = RandomForestClassifier(random_state=10, n_estimators=106, max_depth=31)
# clf3 = LogisticRegression(random_state=1)
clf4 = GaussianNB()
# clf5 = xgboost.XGBClassifier()
model = VotingClassifier(estimators=[
    # ('gbdt',clf1),
    ('rf',clf2),
    # ('lr',clf3),
    ('nb',clf4),
    # ('xgboost',clf5),
    ],
    voting='soft',
    n_jobs=-1)
model.fit(x_data_train, y_data_train)
fit_model(model, 'VotingClassifier')