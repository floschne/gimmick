import os
from pathlib import Path
from typing import Any

import anthropic
from decord import VideoReader
from PIL import Image

import backoff
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


class ClaudeSonnetModel(BaselineModel):
    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        video_fps: int = 1,
    ):
        super().__init__(model_id=model_id)

        if model_id not in ["claude-3-5-sonnet-20241022"]:
            raise ValueError("Please provide a valid Claude 3.5 Sonnet model_id!")

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key is None:
            raise ValueError("Please provide an Anthropic API key!")

        self.client = anthropic.Anthropic(api_key=api_key)
        if not self.client:
            raise ValueError(f"Anthropic model {model_id} could not be loaded.")

        self.system_prompt = system_prompt
        self.video_fps = video_fps

        max_tokens = self.generation_kwargs.pop("max_new_tokens")
        self.generation_kwargs["max_tokens"] = max_tokens
        self.generation_kwargs["temperature"] = 0.0
        del self.generation_kwargs["do_sample"]
        del self.generation_kwargs["top_p"]
        del self.generation_kwargs["top_k"]

    def _create_image_parts(
        self, images: Image.Image | list[Image.Image] | None
    ) -> list:
        if images is None:
            return []

        if not isinstance(images, list):
            images = [images]

        content_parts = []
        for image in images:
            # Return the raw base64 string (NOT a data URL).
            b64_data = image_to_b64(image, return_url=False)
            content_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_data,
                    },
                }
            )
        return content_parts

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
        # For simple text, we just create a single user message with text content
        messages = {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["prompt"]},
            ],
        }
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

    def generate_response_for_audios(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError("Audio support not yet implemented for Claude Sonnet")

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type in {
            TaskType.SINGLE_IMAGE_QA,
            TaskType.MULTI_IMAGE_QA,
            TaskType.VIDEO_QA,
            TaskType.TEXT_QA,
        }

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> dict:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")

        # If only 1 image, but no placeholder found, prepend it
        if len(images) == 1 and prompt.count(IMAGE_PLACEHOLDER_TOKEN) == 0:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        prompt_parts = split_prompt_by_image_tokens(prompt, num_images=len(images))
        content = []
        for i, image in enumerate(images):
            # Append text part
            if prompt_parts[i]:
                content.append({"type": "text", "text": prompt_parts[i]})
            # Append the corresponding image
            content.append(self._create_image_parts(image)[0])

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
        if isinstance(video, (str, Path)):
            video = load_video(video)
        elif not isinstance(video, VideoReader):
            raise ValueError("Video must be a VideoReader or a path to a video")

        if prompt.count(VIDEO_PLACEHOLDER_TOKEN) > 1:
            raise ValueError("Only one video placeholder is supported")

        prompt = prompt.replace(VIDEO_PLACEHOLDER_TOKEN, "")
        frames = extract_frames_from_video(video, self.video_fps)

        for _ in frames:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        return self._prepare_model_input_for_images(prompt, frames)

    @backoff.on_exception(backoff.expo, Exception, max_time=600)
    def _generate_response_str(
        self,
        prompt: str | dict[str, Any],
        images: Image.Image | list[Image.Image] | None = None,
        **kwargs,
    ) -> str:
        try:
            messages = []
            if self.system_prompt and self.system_prompt.strip():
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": self.system_prompt},
                        ],
                    }
                )

            if isinstance(prompt, str):
                # If prompt is just a string, put it into user content
                # plus any images that were passed in
                user_content = [{"type": "text", "text": prompt}]
                if images:
                    user_content.extend(self._create_image_parts(images))
                messages.append({"role": "user", "content": user_content})
            elif isinstance(prompt, dict):
                # If prompt is already a dict, we assume it has 'role' and 'content'
                if "role" not in prompt or "content" not in prompt:
                    raise ValueError(
                        "Invalid prompt format (missing 'role'/'content')."
                    )
                messages.append(prompt)
            else:
                raise ValueError("Invalid prompt format!")

            call_params = {**self.generation_kwargs, **kwargs}

            response = self.client.messages.create(
                model=self.model_id,
                messages=messages,
                **call_params,
            )

            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error when getting response content: {e}")
            raise e
