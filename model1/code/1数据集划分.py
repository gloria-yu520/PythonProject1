# 导入pandas，给它起个小名pd（大家都这么做，方便后续使用）
import pandas as pd
# 从sklearn中导入两种常用的划分工具（按需选择即可）
# 1. 简单随机划分（适合样本均衡的情况）
from sklearn.model_selection import train_test_split
# 2. 分层随机划分（适合样本不平衡，影像组学更常用，优先推荐）
from sklearn.model_selection import StratifiedShuffleSplit
# 选择对应情况的代码运行即可（二选一）

# 情况A：加载你自己的CSV文件（以桌面为例，修改为你的文件实际路径）
# 说明：Windows系统路径用\，Mac/Linux用/，如果路径有中文，可添加encoding='utf-8'
df = pd.read_csv("../datasetfile/originalData.csv")

# 情况B：创建模拟影像组学数据（无需自己准备文件，直接运行）


# 可选：查看加载的数据，确认是否正确（类似Excel中查看表格）
print("原始数据集：")
print(df)


# 初始化分层划分工具，设置比例和随机种子
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=123)

# 分层因素为bone_metastasis列
for train_index, test_index in sss.split(df, df["骨转移"]):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]

# 验证划分后比例是否一致
print("原始数据骨转移占比：", df["骨转移"].mean())
print("训练集骨转移占比：", train_set["骨转移"].mean())
print("测试集骨转移占比：", test_set["骨转移"].mean())
# 打印训练集和测试集，查看划分效果
print("\n" + "="*50)
print("训练集（70%样本）：")
print(train_set)
print("\n测试集（30%样本）：")
print(test_set)
# 保存训练集为CSV文件（保存到桌面，可修改路径）
train_set.to_csv("resultset/trainOriginal.csv", index=False)
# 保存测试集为CSV文件
test_set.to_csv("resultset/testOriginal.csv", index=False)

# 若想保存为Excel文件，需额外安装openpyxl库，命令：pip install openpyxl
# train_set.to_excel("C:/Users/你的用户名/Desktop/训练集.xlsx", index=False, engine="openpyxl")
# test_set.to_excel("C:/Users/你的用户名/Desktop/测试集.xlsx", index=False, engine="openpyxl")




