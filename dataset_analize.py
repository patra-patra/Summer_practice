import os
import matplotlib.pyplot as plt

dataset_path = "C:\\Users\\user\\Desktop\\dataset"

folder_count = {}

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)

    if os.path.isdir(folder_path):
        num_files = len(os.listdir(folder_path))
        folder_count[folder_name] = num_files


plt.figure(figsize=(10, 6))
bars = plt.bar(folder_count.keys(), folder_count.values())
plt.xlabel('Классы')
plt.ylabel('Количество изображений')
plt.title('Распределение изображений по классам')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

