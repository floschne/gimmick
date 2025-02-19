from pathlib import Path
from typing import Any

from decord import VideoReader
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.perplexity import (
    AnswerPerplexityResult,
    compute_perplexity_and_avg_log_likelihood,
)
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
    extract_frames_from_video,
    load_video,
)


class Phi3_5_Vision(BaselineModel):
    def __init__(
        self,
        video_fps: int = 1,
    ):
        model_id = "microsoft/Phi-3.5-vision-instruct"
        super().__init__(model_id=model_id)
        self.multi_image_processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4,
        )
        self.single_image_processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        ).eval()

        self.video_fps = video_fps

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _get_processor(self, images: list):
        if len(images) < 1:
            processor = self.single_image_processor
        else:
            processor = self.multi_image_processor

        return processor

    def _generate_response(
        self,
        sample: GimmickSample,
        final_prompt: str,
        inputs,
    ) -> GimmickModelResponse:
        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=self.single_image_processor.tokenizer.eos_token_id,
            **self.generation_kwargs,
        )
        # trim the input tokens from the generated output
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]

        if "images" in sample:
            processor = self._get_processor(sample["images"])
        else:
            processor = self.single_image_processor

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=final_prompt,
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

        inputs, final_prompt = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )

        return self._generate_response(sample, final_prompt, inputs)

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        inputs, final_prompt = self._prepare_model_input_for_videos(
            sample["prompt"], sample["videos"]
        )

        return self._generate_response(sample, final_prompt, inputs)

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        inputs, final_prompt = self._prepare_model_input_for_text(
            sample["prompt"],
        )

        return self._generate_response(sample, final_prompt, inputs)

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> Any:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")

        num_image_placeholders = prompt.count(IMAGE_PLACEHOLDER_TOKEN)
        if num_image_placeholders == 0 and len(images) == 1:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt
            num_image_placeholders = 1

        if num_image_placeholders != len(images):
            raise ValueError(
                f"Number of images ({len(images)}) must match the number "
                f"of IMAGE_PLACEHOLDER_TOKENs in the prompt ({num_image_placeholders})"
            )

        # replace the IMAGE_PLACEHOLDER_TOKEN with the NUMBERED! Phi3.5 image placeholder tokens
        phi_image_placeholder_token = "<|image_{NUM}|>"
        for i in range(1, num_image_placeholders + 1):
            prompt = prompt.replace(
                IMAGE_PLACEHOLDER_TOKEN, phi_image_placeholder_token.format(NUM=i), 1
            )
        messages = [
            {"role": "user", "content": prompt},
        ]

        processor = self._get_processor(images)

        final_prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(final_prompt, images, return_tensors="pt").to("cuda")

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

        final_prompt = self.single_image_processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.single_image_processor(final_prompt, return_tensors="pt").to(
            "cuda"
        )

        return inputs, final_prompt

    def compute_answer_perplexity(
        self, sample: GimmickSample, **kwargs
    ) -> AnswerPerplexityResult:
        # First, get the tokens for the question and the ground truth answer
        eos_token = "<|endoftext|>"

        if "images" not in sample:
            raise ValueError("Sample must contain images")

        model_inputs, prompt = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )
        question_tokens = model_inputs.input_ids.cpu()

        gt_answer_tokens = self.single_image_processor.tokenizer(
            text=[f"{sample['ground_truth']}{eos_token}"], return_tensors="pt"
        ).input_ids.cpu()

        question_answer_tokens = torch.concat(
            (question_tokens, gt_answer_tokens), dim=1
        )

        # Get logits for each position
        outputs = self.model(
            question_answer_tokens.to("cuda"), labels=question_answer_tokens
        )
        logits = outputs.logits  # BxSxV

        perplexity, avg_log_likelihood = compute_perplexity_and_avg_log_likelihood(
            question_answer_tokens, question_tokens, gt_answer_tokens, logits
        )

        return AnswerPerplexityResult(
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            model_id=self.model_id,
            ground_truth=sample["ground_truth"],
            countries=sample["countries"],
            regions=sample["regions"],
            hints=sample["hints"],
            prompt=prompt,
            question_tokens=question_tokens.squeeze().cpu().numpy().tolist(),
            ground_truth_tokens=gt_answer_tokens.squeeze().cpu().numpy().tolist(),
            perplexity=perplexity,
            avg_log_likelihood=avg_log_likelihood,
        )
