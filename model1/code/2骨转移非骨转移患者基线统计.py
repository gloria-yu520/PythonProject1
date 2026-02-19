
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 构造数据
data = pd.read_csv("../datasetfile/基本特征2026-01-07.csv")
# 转换为DataFrame
df = pd.DataFrame(data)
df["P"] = df["P"].fillna("")  # 将所有NaN值替换为空字符串

# 设置字体（确保中文和特殊符号正常显示）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# 创建图和轴
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')  # 隐藏坐标轴

# 生成表格
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]  # 让表格充满整个画布
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # 调整行高

# 绘制三线表的三条线
# 1. 顶部粗线
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    # 2. 表头下方的线
    if i == 1:
        cell.set_edgecolor('black')
        cell.set_linewidth(1)
    # 3. 底部粗线
    if i == len(df) + 1:
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    # 隐藏其他边框
    if i > 0 and i < len(df) + 1:
        cell.set_edgecolor('white')

# 设置标题
plt.title("Table 1 Comparison of clinical features between patients with and without bone metastasis",
          fontsize=12, pad=20)

# 保存为高清图片
plt.savefig("../picture/clinical_table.png", dpi=300, bbox_inches='tight')
plt.show()