import pandas as pd
from sklearn.preprocessing import StandardScaler

# 假设train_data（训练集）、test_data（测试集）已筛选核心特征
continuous_features = [
    "铁蛋白",  # 血清标志物（连续）
    "CA125",  # 代谢指标（连续）
    "NSE" , # 年龄（连续）
    "CYFRA21-1"]
scaler = StandardScaler()
train_data= pd.read_csv("../datasetfile/trainDataModel1.csv")
test_data= pd.read_csv("../datasetfile/testDataDodel1.csv")

# 训练集标准化（拟合+转换）
train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features])

# 测试集标准化（仅用训练集的scaler转换，不拟合）
test_data[continuous_features] = scaler.transform(test_data[continuous_features])
# 定义需要进行Z-score标准化的连续型特征（替换为你的实际特征名称）
# 注意：排除结局变量（bone_metastasis）和分类变量（如Smoking、TNM_stage）


# 确认结局变量（无需标准化，仅作为参考）
outcome_feature = "骨转移"
# 初始化StandardScaler（Z-score标准化器）
scaler = StandardScaler()
# 训练集：拟合+转换
train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features])
# 测试集：仅转换
test_data[continuous_features] = scaler.transform(test_data[continuous_features])

# 步骤5：验证标准化结果
print("===== 训练集标准化后统计结果 =====")
print(train_data[continuous_features].describe().round(3))
print("\n===== 测试集标准化后统计结果 =====")
print(test_data[continuous_features].describe().round(3))

# 步骤6：保存标准化后的数据集
train_data.to_csv("../datasetfile/train_data_model1_standardized.csv", index=False, encoding="utf-8-sig")
test_data.to_csv("../datasetfile/test_data_model1_standardized.csv", index=False, encoding="utf-8-sig")

print("\n✅ 标准化流程全部完成，文件已保存！")