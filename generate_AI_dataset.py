import json
import time

from openai import OpenAI

from utilities import get_key, create_folder_for_label, convert_and_save_images, generate_image

with open("Labels.json", "r") as file:
    dataset_labels = json.load(file)

labels = list(dataset_labels.values())

openai_client = OpenAI(api_key=get_key())

for j, label in enumerate(labels):
    print(f"Working on label {j + 1}/{len(labels)}: {label}")
    folder_path = create_folder_for_label(label, base_path="dataset/ai")

    twenty_images = generate_image(openai_client,
                                   prompt=f"Generate a realistic photo of a {label} in its natural habitat.", n=5,
                                   size="256x256")

    convert_and_save_images(twenty_images, folder_path, label)

    time.sleep(60)
