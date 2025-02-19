import os
import re
from pathlib import Path
from typing import Any, Literal

import backoff
from decord import VideoReader
from openai import BadRequestError, OpenAI
from PIL import Image

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import (
    IMAGE_PLACEHOLDER_TOKEN,
    image_to_b64,
    split_prompt_by_image_tokens,
)
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
    extract_frames_from_video,
    load_video,
)


class OAIModel(BaselineModel):
    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        img_detail: Literal["low", "auto", "high"] = "auto",
        system_prompt: str | None = None,
        video_fps: int = 1,
    ):
        if model_id not in (
            openai_models := [
                "gpt-4o-mini-2024-07-18",
                "gpt-4o-2024-11-20",
                "gpt-4o-2024-08-06",
                "o1-2024-12-17",
                "o1-mini-2024-09-12",
                "o3-mini-2025-01-31",
            ]
        ):
            raise ValueError(
                f"OpenAI model {model_id} is not supported! Available models: {openai_models}"
            )
        super().__init__(
            model_id=model_id,
        )

        # make sure there is a YYYY-MM-DD in the model_id
        if not re.search(r"\d{4}-\d{2}-\d{2}", model_id):
            raise ValueError(
                "Please provide a valid model_id with a date in YYYY-MM-DD"
            )

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("Please provide an OpenAI API key!")
        self.api_key = api_key

        self.client = OpenAI(api_key=api_key)
        if not self.client:
            raise ValueError(f"OpenAI model {model_id} could not be loaded.")
        self.img_detail = img_detail
        self.system_prompt = system_prompt
        self.video_fps = video_fps
        self.generation_kwargs = {
            "max_completion_tokens": 512,
            "temperature": 0.0,
            "top_p": None,
        }
        if model_id.startswith("o1"):
            del self.generation_kwargs["temperature"]
            del self.generation_kwargs["top_p"]
            self.generation_kwargs["temperature"] = 1

    def _create_image_parts(
        self, images: Image.Image | list[Image.Image] | None
    ) -> list:
        if images is None:
            return []
        if not isinstance(images, list):
            images = [images]
        base64_image_urls = [
            image_to_b64(image, return_url=True) for image in images for image in images
        ]
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_image,
                    "detail": self.img_detail,
                },
            }
            for base64_image in base64_image_urls
        ]

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "images" not in sample:
            raise ValueError("Sample must contain 'images' key")

        messages = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )

        response = self._generate_response_str(messages, **kwargs)

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
            response=response,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        messages = self._prepare_model_input_for_videos(
            sample["prompt"], sample["videos"]
        )

        response = self._generate_response_str(messages, **kwargs)

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
            response=response,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        messages = self._prepare_model_input_for_text(sample["prompt"])

        response = self._generate_response_str(messages, **kwargs)

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
            response=response,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def supports_task(self, task_type: TaskType) -> bool:
        if self.model_id.startswith("o1") or self.model_id.startswith("o3"):
            return task_type in {
                TaskType.TEXT_QA,
            }
        return task_type in {
            TaskType.SINGLE_IMAGE_QA,
            TaskType.MULTI_IMAGE_QA,
            TaskType.VIDEO_QA,
            TaskType.TEXT_QA,
        }

    def _prepare_model_input_for_text(self, prompt: str) -> dict:
        messages = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }

        return messages

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> dict:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")
        if len(images) == 1 and prompt.count(IMAGE_PLACEHOLDER_TOKEN) == 0:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        prompt_parts = split_prompt_by_image_tokens(prompt, num_images=len(images))
        content = []
        for i in range(len(images)):
            # Append the text part (prompt_parts[i])
            if prompt_parts[i]:  # Only add if it's not empty
                content.append({"type": "text", "text": prompt_parts[i]})

            # Append the image
            content.append(self._create_image_parts(images[i])[0])

        # Append the final text part
        if prompt_parts[-1]:
            content.append({"type": "text", "text": prompt_parts[-1]})

        message = {
            "role": "user",
            "content": content,
        }

        return message

    def _prepare_model_input_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> dict:
        if not isinstance(videos, list):
            videos = [videos]

        if len(videos) == 0:
            raise ValueError("At least one video is required")
        elif len(videos) > 1:
            raise ValueError("Only one video is supported")

        video = videos[0]
        if isinstance(videos[0], (str, Path)):
            video = load_video(videos[0])
        elif not isinstance(video, VideoReader):
            raise ValueError("Video must be a VideoReader or a path to a video")

        # we only allow one video before the prompt
        if prompt.count(VIDEO_PLACEHOLDER_TOKEN) > 1:
            raise ValueError("Only one video placeholder is supported")
        prompt = prompt.replace(VIDEO_PLACEHOLDER_TOKEN, "")

        frames = extract_frames_from_video(video, self.video_fps)
        for _ in frames:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        return self._prepare_model_input_for_images(prompt, frames)

    def generate_response_for_audios(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def _generate_response_str(
        self,
        prompt: str | dict[str, Any],
        images: Image.Image | list[Image.Image] | None = None,
        **kwargs,
    ) -> str:
        try:
            messages = []
            if self.system_prompt and self.system_prompt != "":
                messages.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )

            if isinstance(prompt, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *self._create_image_parts(images),
                        ],
                    }
                )
            elif isinstance(prompt, dict):
                # If prompt is a dict, it should be a valid message object
                if "role" not in prompt or "content" not in prompt:
                    raise ValueError("Invalid prompt format!")
                messages.append(prompt)
            else:
                raise ValueError("Invalid prompt format!")

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **self.generation_kwargs,
                **kwargs,
            )
        except Exception as e:
            if isinstance(e, BadRequestError) and e.code == "content_policy_violation":
                print(f"Content Policy Violation!: {e}")
                return "Content Policy Violation!"
            print(f"Error when getting response content: {e}")
            raise e
        try:
            return response.choices[0].message.content.strip()
        except Exception as e:
            if isinstance(e, (ValueError, IndexError)):
                print(f"Error when getting response content: {e}")
                return "Error when getting response content!"
            raise e
