import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from decord import VideoReader
from PIL.Image import Image
from vllm import LLM
from vllm.sampling_params import SamplingParams

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import (
    IMAGE_PLACEHOLDER_TOKEN,
    image_to_b64,
    split_prompt_by_image_tokens,
)
from gimmick.utils.video import VIDEO_PLACEHOLDER_TOKEN, load_video

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


class Pixtral(BaselineModel):
    def __init__(
        self,
        max_img_per_msg: int = 50,
        video_fps: int = 1,
    ):
        model_id = "mistralai/Pixtral-12B-2409"
        super().__init__(model_id=model_id)
        self.model = LLM(
            model=model_id,
            tokenizer_mode="mistral",
            limit_mm_per_prompt={"image": max_img_per_msg},
            max_model_len=32768 // 2,
        )
        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
        )

        self.video_fps = video_fps

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _generate_response(
        self,
        sample: GimmickSample,
        messages: list[dict],
        **kwargs,
    ) -> GimmickModelResponse:
        final_prompt_without_image_urls = deepcopy(messages)
        # replace image urls
        for message in final_prompt_without_image_urls:
            for content in message["content"]:
                if content["type"] == "image_url":
                    content["image_url"] = "<IMAGE_URL>"

        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        output_texts = [output.text for output in outputs[0].outputs]

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=str(final_prompt_without_image_urls),
            ground_truth=sample["ground_truth"],
            response=output_texts[0],
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "images" not in sample:
            raise ValueError("Sample must contain 'images' key")

        messages = self._prepare_messages_for_images(sample["prompt"], sample["images"])

        return self._generate_response(sample, messages)

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        messages = self._prepare_messages_for_videos(sample["prompt"], sample["videos"])

        return self._generate_response(sample, messages)

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        messages = [{"role": "user", "content": sample["prompt"]}]

        return self._generate_response(sample, messages)

    def _prepare_messages_for_images(
        self, prompt: str, images: list[Image]
    ) -> list[dict]:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")
        if len(images) == 1 and prompt.count(IMAGE_PLACEHOLDER_TOKEN) == 0:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        image_urls = [image_to_b64(image, return_url=True) for image in images]
        prompt_parts = split_prompt_by_image_tokens(prompt, num_images=len(image_urls))
        content = []
        for i in range(len(image_urls)):
            # Append the text part (prompt_parts[i])
            if prompt_parts[i]:  # Only add if it's not empty
                content.append({"type": "text", "text": prompt_parts[i]})

            # Append the image
            content.append({"type": "image_url", "image_url": {"url": image_urls[i]}})

        # Append the final text part
        if prompt_parts[-1]:
            content.append({"type": "text", "text": prompt_parts[-1]})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        return messages

    def _prepare_messages_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> list[dict]:
        if not isinstance(videos, list):
            videos = [videos]

        if len(videos) == 0:
            raise ValueError("At least one video is required")
        elif len(videos) > 1:
            raise ValueError("Only one video is supported")

        # we only allow one video before the prompt
        if prompt.count(VIDEO_PLACEHOLDER_TOKEN) > 1:
            raise ValueError("Only one video placeholder is supported")
        prompt = prompt.replace(VIDEO_PLACEHOLDER_TOKEN, "")

        video = videos[0]
        if isinstance(videos[0], (str, Path)):
            video = load_video(videos[0])
        elif not isinstance(video, VideoReader):
            raise ValueError("Video must be a VideoReader or a path to a video")

        frames = extract_frames_from_video(video, self.video_fps)
        for _ in frames:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        return self._prepare_messages_for_images(prompt, frames)
