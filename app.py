import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from animal_classifier import AnimalClassifier

file_path2 = ""
model = AnimalClassifier()


def crop(image_path, output_path):
    img = Image.open(image_path)
    width, height = img.size

    new_side = min(width, height)
    left = (width - new_side) // 2
    top = (height - new_side) // 2
    right = left + new_side
    bottom = top + new_side

    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(output_path)
    img.close()
    img_cropped.close()


def open_image():
    global file_path2  # Объявляем переменную как глобальную
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        crop(file_path, file_path)
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Изменяем размер изображения
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        recognize_btn.pack(pady=10)  # Показываем кнопку "Распознать"
        file_path2 = file_path  # Обновляем значение file_path2
        label_result.config(text="")


def recognize_image():
    global file_path2
    class_, probability = model.predict(file_path2)
    label_result.config(text=f"{class_}: {probability}%")


root = tk.Tk()
root.title("Распознавание изображений")


root.geometry("400x550")
root.resizable(False, False)


btn = tk.Button(root, text="Open Image", command=open_image)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack(pady=20)

recognize_btn = tk.Button(root, text="Распознать", command=recognize_image)
recognize_btn.pack_forget()

label_result = tk.Label(root, text="")
label_result.pack(pady=20)

root.mainloop()
