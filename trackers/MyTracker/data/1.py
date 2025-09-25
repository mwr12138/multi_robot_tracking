import csv
import os
import sys

def check_file_exists(filename):
    if not os.path.exists(filename):
        print(f"错误：文件 '{filename}' 不存在")
        print("请确认：")
        print("1. 文件是否在当前目录")
        print("2. 文件名是否拼写正确（包括扩展名.csv）")
        print("3. 是否使用了完整的文件路径")
        sys.exit(1)

# 输入和输出文件名
input_filename = 'seq11.csv'  # 替换为您的文件名
output_filename = 'seq1.csv'

# 检查输入文件是否存在
check_file_exists(input_filename)

try:
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        csv_reader = csv.reader(infile)
        csv_writer = csv.writer(outfile)
        
        row_count = 0
        for row in csv_reader:
            # 检查列数是否足够
            if len(row) < 7:
                print(f"警告：第 {row_count+1} 行只有 {len(row)} 列（至少需要7列），已跳过")
                continue
            
            # 修改第七列为1（索引为6）
            row[6] = '1'
            
            csv_writer.writerow(row)
            row_count += 1
        
        print(f"处理完成！共修改 {row_count} 行数据")
        print(f"结果已保存到: {os.path.abspath(output_filename)}")
        print(f"每行的第七列已改为1")

except Exception as e:
    print(f"处理过程中出错: {str(e)}")
    sys.exit(1)
