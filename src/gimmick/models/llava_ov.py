from pathlib import Path
from typing import Any

from decord import VideoReader
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import (
    IMAGE_PLACEHOLDER_TOKEN,
    split_prompt_by_image_tokens,
)
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
    extract_frames_from_video,
    load_video,
)


class LlavaOneVision(BaselineModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        video_fps: int = 1,
    ):
        if model_id not in [
            "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        ]:
            raise ValueError(f"Model ID {model_id} not supported")

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.video_fps = video_fps

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _generate_response(self, sample: GimmickSample, inputs, final_prompt: str):
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]

        answer = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0].strip()

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=final_prompt,
            ground_truth=sample["ground_truth"],
            response=answer,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "images" not in sample:
            raise ValueError("Sample must contain 'images' key")
        inputs, final_prompt = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )
        return self._generate_response(sample, inputs, final_prompt)

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")
        inputs, final_prompt = self._prepare_model_input_for_videos(
            sample["prompt"], sample["videos"]
        )
        return self._generate_response(sample, inputs, final_prompt)

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        inputs, final_prompt = self._prepare_model_input_for_text(sample["prompt"])
        return self._generate_response(sample, inputs, final_prompt)

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> Any:
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
            content.append({"type": "image"})

        # Append the final text part
        if prompt_parts[-1]:
            content.append({"type": "text", "text": prompt_parts[-1]})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        final_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=final_prompt, images=images, return_tensors="pt"
        ).to(self.model.device, torch.bfloat16)

        return inputs, final_prompt

    def _prepare_model_input_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> Any:
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

    def _prepare_model_input_for_text(self, prompt: str) -> Any:
        messages = [
            {"role": "user", "content": prompt},
        ]

        final_prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(final_prompt, return_tensors="pt").to("cuda")

        return inputs, final_prompt
