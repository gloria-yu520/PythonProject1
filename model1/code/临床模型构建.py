import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import roc_curve, auc

# 构建Logistic回归模型（添加常数项）
features=['CA125','铁蛋白','NSE','细胞角蛋白19片段（CYFRA21-1）']
train_data= pd.read_csv("../datasetfile/trainOriginal.csv")
X_train = sm.add_constant(train_data[features])
y_train = train_data['骨转移']
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary())  # 输出模型系数、P值、OR值

# 预测测试集概率
test_data= pd.read_csv("../datasetfile/testOriginal.csv")
X_test = sm.add_constant(test_data[features])
test_data['predict_prob'] = result.predict(X_test)
test_data.to_csv("resultset/概率表.csv", index=False)