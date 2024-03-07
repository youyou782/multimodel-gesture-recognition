import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(dataset_path):
    images = []
    label = []
    for foldername in os.listdir(dataset_path):
        folderpath = os.path.join(dataset_path, foldername)
        for filename in os.listdir(folderpath):
            if filename.endswith(".jpg"):
                img = Image.open(os.path.join(folderpath, filename)).convert('L')
                images.append(np.array(img))
                label.append(foldername)
    print(len(images))
    print(len(label))

    return images, label


if __name__ == "__main__" :
    
    script_path = os.path.abspath(__file__)
    folder_path = os.path.join(os.path.dirname(script_path), "../DataSet/Image")
    image, label = load_images_from_folder(folder_path)

    print("Number of images:", len(image))
    if len(image) > 0:
        plt.imshow(image[0], cmap='gray')
        plt.colorbar()
        plt.show()
        print("Size of image:", image[0].shape) 
    # save images and labels
    # np.save(os.path.join(os.path.dirname(script_path), 'image_data.npy'), image)
    # np.save(os.path.join(os.path.dirname(script_path), 'label_data.npy'), label)