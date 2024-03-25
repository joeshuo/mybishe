import os
from scipy import io
import numpy as np
import pandas as pd


# 定义输入和输出文件夹路径
input_folder_path = r"D:\毕设相关\毕设数据\BCICIV_2a_gdf\3_epochmat"
output_folder_path = r"D:\毕设相关\毕设数据\BCICIV_2a_gdf\4_augmented_data"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 定义文件名模板
file_name_template = os.path.join(input_folder_path, "{}\\{}_{}.mat")

# 使用循环遍历文件夹和文件，并逐个读取.mat文件
for folder in os.listdir(input_folder_path):
    folder_dir = os.path.join(input_folder_path, folder)

    if not os.path.isdir(folder_dir):
        continue

    output_subfolder = os.path.join(output_folder_path, folder)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    for i in range(1, 289):
        file_path = file_name_template.format(folder, folder, i)
        print(file_path)

        # 使用scipy.io.loadmat方法读取.mat文件
        try:
            mat_data = io.loadmat(file_path)    # 数据在mat_data['a']中

            # 在这里对读取到的mat数据进行处理或使用
            # 获取原始数据
            if 'a' in mat_data:
                sliding_data = mat_data['a']        # 数据维度（22,875）

                # 在这里对提取到的数据进行处理或使用
                # 进行数据扩增
                # 获取矩阵维度C、T值
                dimensions = sliding_data.shape
                C = dimensions[0]  # 行的维度
                T = dimensions[1]  # 列的维度

                # 打印查看维度
                # print("C------------->", C)
                # print("T------------->", T)

                # 指定滑动步长
                S = 80

                # 时间序列重组
                k = 1
                while k * S < T:
                    sliding_data = np.roll(sliding_data, S, axis=1)

                    print("--------滚动", k, "次后--------------------")
                    # print(sliding_data)

                    # 生成不同的扩增文件名
                    output_file_name = "{}_{}_{}.csv".format(folder, i, k)
                    output_file_number = "{}_{}".format(folder, i)

                    output_folder = os.path.join(output_subfolder, output_file_number)

                    # 创建输出文件夹（如果不存在）
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    output_file_path = os.path.join(output_subfolder, output_file_number, output_file_name)

                    # 将滑动后的数据保存到CSV文件
                    df = pd.DataFrame(sliding_data)
                    df.to_csv(output_file_path, header=False, index=False)

                    print("保存文件：", output_file_path)

                    k += 1

            else:
                print("在文件 '{}' 中找不到键名 'a'".format(file_path))

        except IOError:
            print("无法读取文件：", file_path)
