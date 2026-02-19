# 步骤1：导入所需库
import pandas as pd
import statsmodels.api as sm

# 步骤2：读取标准化后的数据集
train_standardized = pd.read_csv("../datasetfile/train_data_model1_standardized.csv", encoding="utf-8-sig")
test_standardized = pd.read_csv("../datasetfile/test_data_model1_standardized.csv", encoding="utf-8-sig")

# 步骤3：准备建模数据（X：特征变量，y：结局变量）
# 配置参数：替换为你实际的列名
outcome_col = "骨转移"  # 结局变量名称
feature_cols = [
    "铁蛋白",  # 血清标志物（连续）
    "CA125",  # 代谢指标（连续）
    "NSE" , # 年龄（连续）
    "CYFRA21-1"]

# 提取结局变量和特征变量
y_train = train_standardized[outcome_col]
X_train = train_standardized[feature_cols]

# 给特征变量添加常数项（statsmodels Logistic回归必需）
X_train_with_const = sm.add_constant(X_train)

# 步骤4：拟合二元Logistic回归模型
logit_model = sm.Logit(y_train, X_train_with_const)
model1_fitted = logit_model.fit(max_iter=1000)

# 输出模型详细结果
print("===== 二元Logistic回归模型拟合结果 =====")
print(model1_fitted.summary())

# 步骤5：提取训练集和测试集的预测概率
# 训练集预测概率
# 本次变更测试
train_standardized["predict_prob_train"] = model1_fitted.predict(X_train_with_const)
aaa = 123
# 测试集预测概率（先添加常数项，再预测）
X_test = test_standardized[feature_cols]
X_test_with_const = sm.add_constant(X_test)
test_standardized["predict_prob_test"] = model1_fitted.predict(X_test_with_const)

# 步骤6：验证预测概率分布
print("\n===== 训练集预测概率统计结果 =====")
print(train_standardized["predict_prob_train"].describe().round(4))
print("\n===== 测试集预测概率统计结果 =====")
print(test_standardized["predict_prob_test"].describe().round(4))

# 步骤7：保存包含预测概率的数据集
train_standardized.to_csv("../datasetfile/train_data_model1_with_prob.csv", index=False, encoding="utf-8-sig")
test_standardized.to_csv("../datasetfile/test_data_model1_with_prob.csv", index=False, encoding="utf-8-sig")

print("\n✅ 全部流程完成！预测概率已保存至新数据集")