from typing import Any

import torch
from PIL.Image import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from pathlib import Path
from decord import VideoReader

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
    extract_frames_from_video,
    load_video,
)


class PaliGemma2(BaselineModel):
    def __init__(
        self,
        model_id: str = "google/paligemma2-3b-pt-896",
        video_fps: int = 1,
    ):
        if not model_id.startswith("google/paligemma2-"):
            raise ValueError(f"Model ID {model_id} not supported")

        super().__init__(model_id=model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        self.processor: PaliGemmaProcessor = PaliGemmaProcessor.from_pretrained(
            model_id
        )  # type: ignore

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
        model_inputs: dict[str, Any],
        final_prompt: str,
    ) -> GimmickModelResponse:
        generated_ids = self.model.generate(**model_inputs, **self.generation_kwargs)
        generated_ids_trimmed = generated_ids[0][model_inputs["input_ids"].shape[-1] :]
        genereted_text = self.processor.decode(
            generated_ids_trimmed, skip_special_tokens=True
        )

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=final_prompt,
            ground_truth=sample["ground_truth"],
            response=genereted_text,
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

    def _prepare_model_input_for_text(self, prompt: str) -> tuple[dict[str, Any], str]:
        raise NotImplementedError("Not implemented yet")

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image]
    ) -> tuple[dict[str, Any], str]:
        raise NotImplementedError("Not implemented yet")

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
