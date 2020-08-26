import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import joblib


# 自定义绘图函数1
def show_picture_1(index):
    y = np.arange(len(model_names)) + 1
    x = np.array(list(dict[index]))
    # 默认颜色表
    color_list = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
    # 查找最大值，返回其索引，更改条形图颜色
    max_index = dict[index].index(max(dict[index]))
    color_list[max_index] = 'lightcoral'
    # 绘制条形图
    plt.barh(y - 0.2, x, height=0.5, tick_label=model_names, color=color_list)
    plt.title(index)
    # 设置数字标签
    for a, b in zip(x, y):
        plt.text(a + 0.06, b - 0.2, '%.4f' % a, fontsize=8, ha='right', va='center')
    # 保存图片
    plt.savefig(f'picture1/{index}.jpg', bbox_inches='tight', dpi=200)
    # 清空画板
    plt.cla()


# 自定义绘图函数2
def show_picture_2(index):
    x_label = ['accuracy', 'precision', 'recall', 'f1']
    x = np.array(x_label)
    y = np.array(list(dict[index]))
    # 绘制条形图
    plt.bar(x, y, color=['lightcoral', 'mediumseagreen', 'darkorange', 'cornflowerblue'])
    plt.title(index)
    # 设置数字标签
    for a, b in zip(x, y):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=7)
    # 保存图片
    plt.savefig(f'picture2/{index}.jpg')
    # 清空画板
    plt.cla()


# 加载数据
x_data_train = joblib.load('save/x_data_train')
x_data_test = joblib.load('save/x_data_test')
y_data_train = joblib.load('save/y_data_train')
y_data_test = joblib.load('save/y_data_test')
x_test_1 = joblib.load('save/x_test_1')
x_test_2 = joblib.load('save/x_test_2')
y_test_1 = joblib.load('save/y_test_1')
y_test_2 = joblib.load('save/y_test_2')

model_names = ('DecisionTreeClassifier', 'LinearSVC', 'RandomForestClassifier', 'MultinomialNB',
               'LinearDiscriminantAnalysis', 'LogisticRegression', 'GradientBoostingClassifier',
               'KNeighborsClassifier', 'AdaBoostClassifier', 'GaussianNB', 'QuadraticDiscriminantAnalysis',
               'XGBClassifier', 'VotingClassifier')
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
        # 注：因knn模型数据巨大，测试时间长，为节省展示时间，这里直接将预测好的数据储存，
        # 展示时直接调用储存的数据，不在进行运算
        # model = joblib.load(f'save/{model_name}')
        # predictions = model.predict(x_test_2)
        # joblib.dump(predictions, 'save/knn_predictions_test2')
        predictions = joblib.load('save/knn_predictions_test2')
    else:
        print(f'Loading {model_name}...')
        print(f'Testing {model_name}...')
        model = joblib.load(f'save/{model_name}')
        predictions = model.predict(x_test_2)
    y = y_test_2
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

# 开始画图

# 第一种图 四个指标对应每个的13个模型

# accuracy 图
show_picture_1('accuracy')
# precision 图
show_picture_1('precision')
# recall 图
show_picture_1('recall')
# f1 图
show_picture_1('f1')

# 第二种图 每个模型与其对应的四个指标

for model_name in model_names:
    show_picture_2(model_name)
