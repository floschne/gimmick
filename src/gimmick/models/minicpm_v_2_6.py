from pathlib import Path

import torch
from decord import VideoReader
from PIL.Image import Image
from transformers import AutoModel, AutoTokenizer

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import split_prompt_by_image_tokens, IMAGE_PLACEHOLDER_TOKEN
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
    extract_frames_from_video,
    load_video,
)


class MiniCPM_V_2_6(BaselineModel):
    def __init__(
        self,
        video_fps: int = 1,
        video_frame_slices: int = 4,
    ):
        model_id = "openbmb/MiniCPM-V-2_6"
        super().__init__(model_id=model_id)
        self.model = (
            AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.video_fps = video_fps
        self.video_frame_slices = video_frame_slices

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
        for_video: bool = False,
    ) -> GimmickModelResponse:
        video_gen_kwargs = {
            "use_image_id": False,
            "max_slice_nums": self.video_frame_slices,
        }
        gen_kw_args = self.generation_kwargs.copy()
        if for_video:
            gen_kw_args.update(video_gen_kwargs)
        answer = self.model.chat(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer,
            **gen_kw_args,
        )

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=str(messages),
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

        messages = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )

        return self._generate_response(sample, messages)

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "prompt" not in sample:
            raise ValueError("Sample must contain 'prompt' key")

        messages = self._prepare_model_input_for_text(sample["prompt"])

        return self._generate_response(sample, messages)

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        messages = self._prepare_model_input_for_videos(
            sample["prompt"], sample["videos"]
        )

        return self._generate_response(sample, messages, for_video=True)

    def _prepare_model_input_for_text(self, prompt: str) -> list[dict]:
        messages = [{"role": "user", "content": prompt}]
        return messages

    def _prepare_model_input_for_videos(
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

        messages = self._prepare_model_input_for_images(prompt, frames)

        return messages

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image]
    ) -> list[dict]:
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
                content.append(prompt_parts[i])

            # Append the image
            content.append(images[i])

        # Append the final text part
        if prompt_parts[-1]:
            content.append(prompt_parts[-1])

        messages = [{"role": "user", "content": content}]

        return messages
