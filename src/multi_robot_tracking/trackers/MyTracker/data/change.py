import csv
import os
import sys

# 检查文件是否存在
def check_file_exists(filename):
    if not os.path.exists(filename):
        print(f"错误：文件 '{filename}' 不存在")
        print("请确认：")
        print("1. 文件是否在当前目录")
        print("2. 文件名是否拼写正确（包括扩展名.csv）")
        print("3. 是否使用了完整的文件路径")
        sys.exit(1)

# 输入和输出文件名
input_filename = 'seq1.csv'  # 默认文件名
output_filename = 'output.csv'

# 检查输入文件是否存在
check_file_exists(input_filename)

# 处理CSV
try:
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)
        
        row_count = 0
        for row in csv_reader:
            if len(row) != 9:
                print(f"警告：第 {row_count+1} 行有 {len(row)} 列（应为9列），已跳过")
                continue
            
            # 添加一个新值（这里用0占位）
            row.append('0')  # 现在有10列
            
            # 保留前6列，后4列改为-1
            new_row = row[:6] + ['-1'] * 4
            csv_writer.writerow(new_row)
            row_count += 1
        
        print(f"处理完成！共处理 {row_count} 行数据")
        print(f"结果已保存到: {os.path.abspath(output_filename)}")

except Exception as e:
    print(f"处理过程中出错: {str(e)}")
    sys.exit(1)
