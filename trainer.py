# 训练器逻辑# 导入必要的库
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_path = "./datasets/train.jsonl"
test_path = "./datasets/raw/test.jsonl"

def load_train_data(train_path):
    data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行尾的换行符并解析 JSON
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    features = [item['feature'] for item in data]
    labels = [item['label'] for item in data]
    return np.array(features), np.array(labels)

def load_test_data(test_path):
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行尾的换行符并解析 JSON
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    features = [item['feature'] for item in data]
    return np.array(features)

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        # 使用 json.dump 方法将列表写入文件
        json.dump(data, file, indent=4)
    return 

train_data, train_labels = load_train_data(train_path)
test_data = load_test_data(test_path)

def save_test_data(test_path, label):
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate (f):
            # 去掉行尾的换行符并解析 JSON
            json_obj = json.loads(line.strip())
            json_obj['label'] = int(label[i])
            data.append(json_obj)
    return data

# 初始化随机森林分类器
model = RandomForestClassifier(
    n_estimators=100,          # 树的数量（越大可能越准确，但计算量越大）
    max_depth=None,            # 树的最大深度（None表示完全生长）
    min_samples_split=2,       # 内部节点再划分所需最小样本数
    min_samples_leaf=1,        # 叶子节点最少样本数
    criterion='gini',          # 不纯度指标（'gini'或'entropy'）
    class_weight=None,         # 类别权重（处理类别不平衡时设置）
    random_state=42            # 随机种子，保证结果可复现
)

# 训练模型
model.fit(train_data, train_labels)
# 预测测试集
y_pred = model.predict(test_data)
print("预测结果：", y_pred)

# 保存预测结果到 test.jsonl
test_merge_data = save_test_data(test_path, y_pred)
save_data(test_merge_data, "./datasets/merge_test.jsonl")


# 保存模型
model_filename = './checkpoints/model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"模型已保存到 {model_filename}")



# 加载模型
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
# 使用加载的模型进行预测
test_data = np.array(test_data).reshape(1, 36)
y_ = loaded_model.predict(test_data)
print("预测结果：", y_)



