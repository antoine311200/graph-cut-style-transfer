from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os

def resize_image(image, size):
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    return image.resize((new_width, new_height))

if __name__ == "__main__":


    dataset = load_dataset("huggan/wikiart")

    if not os.path.exists("./data/wikiart"):
        os.makedirs("./data/wikiart")

    for i in tqdm(range(len(dataset["train"]))):
        image = dataset["train"][i]["image"] # JpegImageFile
        image = resize_image(image, 512)
        image.save(f"./data/wikiart/{i}.jpg")

    directory = os.listdir("./data/val2017")
    for i in tqdm(range(len(directory))):
        image = Image.open(f"./data/val2017/{directory[i]}")
        image = resize_image(image, 512)
        image.save(f"./data/coco/{i}.jpg")

    