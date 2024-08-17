import pandas as pd

# 读取两个CSV文件
df_ground_truth = pd.read_csv("ground_truth.csv")
df_heuristic = pd.read_csv("dataset_heuristic.csv")

# 假设两个CSV文件有相同的行数并且按相同的顺序排序
assert len(df_ground_truth) == len(df_heuristic), "The datasets should have the same number of rows."

# 提取标签列
labels_ground_truth = df_ground_truth['label']
labels_heuristic = df_heuristic['label']

# 计算准确率
correct_count = 0
total_count = len(labels_ground_truth)

for true_label, heuristic_label in zip(labels_ground_truth, labels_heuristic):
    if true_label == heuristic_label:
        correct_count += 1

accuracy = (correct_count / total_count) * 100
print(f"The accuracy of dataset_heuristic.csv compared to ground_truth.csv is {accuracy}%.")
