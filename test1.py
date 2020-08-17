import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

#表头
name = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
    ,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
    ,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count'
    ,'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate'
    ,'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate'
    ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','class'
]
#读取文件，文件'kddcup.data_10_percent_corrected'是训练集，'corrected'是测试集
print('Reading the file...')
data = pd.read_csv('kddcup.data_10_percent_corrected', names=name)
test = pd.read_csv('corrected',names=name)
#预处理
print('Preprocessing...')
#将class下所有非normal.的标签改为attack.
test['class'].loc[test['class'] != 'normal.'] = 'attack.'
data['class'].loc[data['class'] != 'normal.'] = 'attack.'
#将文字标签编码
data.iloc[:,-1] = LabelEncoder().fit_transform(data.iloc[:,-1])
test.iloc[:,-1] = LabelEncoder().fit_transform(test.iloc[:,-1])
data.iloc[:,1:4] = OrdinalEncoder().fit_transform(data.iloc[:,1:4])
test.iloc[:,1:4] = OrdinalEncoder().fit_transform(test.iloc[:,1:4])
newdata=data
newtest=test
#划分训练集和测试集
x_train = newdata.drop(columns=['class'])
y_train = newdata['class']
x_test = newtest.drop(columns=['class'])
y_test = newtest['class']
#选择决策树模型进行训练
model = DecisionTreeClassifier()
print('training...')
model.fit(x_train, y_train)
#打分，输出准确率
predictions = model.predict(x_test)
score = accuracy_score(y_test, predictions)
print(f'The score is: {score}')