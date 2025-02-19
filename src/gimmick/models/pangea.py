from pathlib import Path

import torch
from decord import VideoReader
from PIL.Image import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

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


class Pangea(BaselineModel):
    def __init__(
        self,
        video_fps: int = 1,
    ):
        model_id = "neulab/Pangea-7B-hf"
        super().__init__(model_id=model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained("neulab/Pangea-7B-hf")
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.video_fps = video_fps

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _generate_response(
        self, sample: GimmickSample, model_inputs: dict, text_input: str
    ) -> GimmickModelResponse:
        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_kwargs,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = generated_ids[:, model_inputs["input_ids"].shape[1] :]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=text_input,
            ground_truth=sample["ground_truth"],
            response=output_texts[0],
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        inputs, final_prompt = self.prepare_model_inputs_for_text(sample["prompt"])

        return self._generate_response(sample, inputs, final_prompt)

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "images" not in sample:
            raise ValueError("Sample must contain 'images' key")

        model_inputs, final_prompt = self._prepare_model_inputs_for_images(
            sample["prompt"], sample["images"]
        )

        return self._generate_response(sample, model_inputs, final_prompt)

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        model_inputs, final_prompt = self._prepare_model_inputs_for_videos(
            sample["prompt"], sample["videos"]
        )

        return self._generate_response(sample, model_inputs, final_prompt)

    def _apply_prompt_template(self, prompt: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def prepare_model_inputs_for_text(self, prompt: str) -> tuple[dict, str]:
        final_prompt = self._apply_prompt_template(prompt)
        model_inputs = self.processor(text=final_prompt, return_tensors="pt").to(
            "cuda", torch.float16
        )

        return model_inputs, final_prompt

    def _prepare_model_inputs_for_images(
        self, prompt: str, images: list[Image]
    ) -> tuple[dict, str]:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")

        img_token_cnt = prompt.count(IMAGE_PLACEHOLDER_TOKEN)
        if img_token_cnt == 0 and len(images) == 1:
            prompt = f"{IMAGE_PLACEHOLDER_TOKEN}\n{prompt}"
            img_token_cnt = 1

        if not img_token_cnt == len(images):
            raise ValueError(
                f"Number of images ({len(images)}) must match the number "
                f"of {IMAGE_PLACEHOLDER_TOKEN} in the prompt ({img_token_cnt})"
            )

        pangea_image_token = "<image>"
        user_prompt = prompt.replace(IMAGE_PLACEHOLDER_TOKEN, pangea_image_token)
        final_prompt = self._apply_prompt_template(user_prompt)

        model_inputs = self.processor(
            images=images, text=final_prompt, return_tensors="pt"
        ).to("cuda", torch.float16)

        return model_inputs, final_prompt

    def _prepare_model_inputs_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> tuple[dict, str]:
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

        return self._prepare_model_inputs_for_images(prompt, frames)
