import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from typing import List, Literal
import openai
from typing import Literal

ImageSize = Literal["auto", "1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"]


load_dotenv()

def get_key():
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return key


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image.

    :param base64_str: The base64 encoded string of the image.

    :return: A PIL Image object.
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def create_folder_for_label(label: str, base_path: str = "dataset/ai") -> str:
    """
    Create a folder for the given label under the specified base path.
    If the folder already exists, it will not raise an error.

    :param label: The label for which the folder is to be created.
    :param base_path: The base path where the folder will be created.

    :return: The path to the created folder.
    """
    folder_path = os.path.join(base_path, label)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def convert_and_save_images(images: List[str], folder_path: str, label: str) -> None:
    """
    Convert base64 images to PIL format and save them in the specified folder.

    :param images: List of base64 encoded image strings.
    :param folder_path: The path to the folder where images will be saved.
    :param label: The label to use for naming the images.

    :return: None
    """
    for i, img_base64 in enumerate(images):
        img = base64_to_pil(img_base64)
        img.save(os.path.join(folder_path, f"{label}_{i}.png"))
        print(f"Saved image {i} in {folder_path}")



def generate_image(client: openai.OpenAI, prompt: str, n: int = 1, size: ImageSize = "256x256") -> str | list[str]:

    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=n,
        size=size,
        response_format="b64_json"
    )

    if response and response.data:
        return [img.b64_json for img in response.data]
    else:
        raise ValueError("Image generation failed or no data returned.")
