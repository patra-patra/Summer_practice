import os
from PIL import Image
import splitfolders


def check_image_format(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return img.format in ["JPEG", "PNG", "GIF", "BMP"]
    except (IOError, SyntaxError) as e:
        print(f'Bad file: {file_path}')
        return False


def crop_to_square(image_path, output_path):
    try:
        img = Image.open(image_path)
        width, height = img.size

        if width < 224 or height < 224:
            img.close()
            os.remove(image_path)
        else:
            new_side = min(width, height)
            left = (width - new_side) // 2
            top = (height - new_side) // 2
            right = left + new_side
            bottom = top + new_side

            img_cropped = img.crop((left, top, right, bottom))
            img_cropped.save(output_path)
            img.close()
            img_cropped.close()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


def check_and_crop(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            if not check_image_format(filepath):
                os.remove(filepath)
                print(f'Removed: {filepath}')
            else:
                crop_to_square(filepath, filepath)


input_folder = "C:\\Users\\user\\Desktop\\dataset"
output_folder = "C:\\Users\\user\\Desktop\\split_dataset"
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .2, .1))
check_and_crop(input_folder)




