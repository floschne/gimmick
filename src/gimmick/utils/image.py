import base64
import io
import math

import backoff
import requests
from PIL import Image

IMAGE_PLACEHOLDER_TOKEN = "<image_placeholder>"


def stack_images(
    images: list[Image.Image],
    num_cols: int,
    stack_glutter: int = 10,
) -> Image.Image:
    if stack_glutter < 0:
        raise ValueError("stack_glutter must be non-negative")
    if not 0 < num_cols < len(images):
        raise ValueError(f"{num_cols=} must be between 1 and {len(images)=}")

    n_rows = math.ceil(len(images) / num_cols)

    img_width = max(img.width for img in images)
    img_height = max(img.height for img in images)

    width = num_cols * img_width + (num_cols - 1) * stack_glutter
    height = n_rows * img_height + (n_rows - 1) * stack_glutter

    stacked_image = Image.new("RGB", (width, height))

    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * (img_width + stack_glutter)
        y = row * (img_height + stack_glutter)
        stacked_image.paste(image, (x, y))

    return stacked_image


def image_to_b64(image: Image.Image, return_url: bool = False) -> str:
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    img_bytes = image_bytes.getvalue()
    encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    if return_url:
        return f"data:image/jpeg;base64,{encoded_string}"
    return encoded_string


def split_prompt_by_image_tokens(
    prompt: str, num_images: int, image_token: str = IMAGE_PLACEHOLDER_TOKEN
) -> list[str]:
    prompt_parts = prompt.split(image_token)
    if len(prompt_parts) != num_images + 1:
        raise ValueError(
            f"Number of images ({num_images}) does not match "
            f"number of placeholders ({len(prompt_parts) - 1})"
        )

    return prompt_parts


@backoff.on_exception(backoff.expo, Exception, max_time=120)
def load_image_from_url(img_url: str) -> Image.Image:
    return Image.open(requests.get(img_url, stream=True).raw)  # type: ignore
