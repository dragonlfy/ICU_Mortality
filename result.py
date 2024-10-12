import pandas as pd

result = pd.DataFrame()
for csv_index in range(3):
    path = f'logs/00/46b4470_nbp_h_2dw/att_combination3_option1_last_6/20240226T194840/mimic3.fold_{csv_index}_all.csv'
    # 读取CSV文件
    data = pd.read_csv(path)
    # 获取第一个最大值的索引
    max_value_index = data['test_f1'].idxmax()
    # 提取第一个最大值所在的行
    max_value_row = data.loc[max_value_index]
    # 将结果添加到结果DataFrame中
    # result = result.append(max_value_row, ignore_index=True)
    result = pd.concat([result, max_value_row.to_frame().T], ignore_index=True)

result.to_csv('logs/00/46b4470_nbp_h_2dw/att_combination3_option1_last_6/max_value_rows.csv', index=False)

# 计算每一列的平均值和标准差
column_means = result.mean()  # 平均值
column_stds = result.std()  # 标准差

# 计算误差并添加到新的一行
error_row = column_means.astype(str) + " ± " + column_stds.astype(str)
error_frame = pd.DataFrame([error_row])  # 将误差行转换为DataFrame以便能够使用concat
result = pd.concat([result, error_frame], ignore_index=True)
# result = result.append(error_row, ignore_index=True)

# 保存结果到CSV文件
result.to_csv('logs/00/46b4470_nbp_h_2dw/att_combination3_option1_last_6/max_value_rows_with_error.csv', index=False)
# %%
