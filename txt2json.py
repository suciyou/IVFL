import json
import os.path
from fscil import MyConfig

file_path = '/root/autodl-tmp/data'

data_name = 'miniImageNet'

txt_path = os.path.join(file_path, data_name, "class_text.txt")
print(txt_path)

with open(txt_path, 'r') as file:
    lines = file.readlines()

data = {}


for line in lines:
    parts = line.strip().split(' ', 1)
    if len(parts) == 2:
        class_name = parts[0]
        sample = parts[1]

        if class_name in data:
            data[class_name].append(sample)
        else:
            data[class_name] = [sample]


json_data = json.dumps(data, indent=4)

with open(f'{file_path}/{data_name}/class_text.json', 'w') as json_file:
    json_file.write(json_data)

print(f"转换完成，结果已保存到 {file_path}/{data_name}/class_text.json 文件中。")


