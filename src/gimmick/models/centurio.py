from pathlib import Path

import torch
from decord import VideoReader
from PIL import Image
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


class Centurio(BaselineModel):
    def __init__(
        self,
        model_id: str,
        video_fps: int = 1,
    ):
        if model_id not in ["WueNLP/centurio_aya", "WueNLP/centurio_qwen"]:
            raise ValueError("Model ID not supported")
        super().__init__(model_id=model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            padding_side="right" if model_id == "WueNLP/centurio_qwen" else "left",
        )

        self.video_fps = video_fps

        if model_id == "WueNLP/centurio_qwen":
            self.generation_kwargs = {
                **self.generation_kwargs,
                "pad_token_id": 151643,
                "min_new_tokens": 1,
            }

    def _apply_chat_template(self, prompt: str):
        if self.model_id == "WueNLP/centurio_qwen":
            prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{PROMPT}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            prompt_template = (
                "<BOS_TOKEN>"
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\n{PROMPT}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )

        return prompt_template.format(PROMPT=prompt)

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _generate_response(
        self, sample: GimmickSample, model_inputs: dict, final_prompt: str
    ) -> GimmickModelResponse:
        outputs = self.model.generate(**model_inputs, **self.generation_kwargs)

        outputs = outputs[0, model_inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(outputs, skip_special_tokens=True).strip()

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
        inputs = self._prepare_model_inputs_for_text(sample["prompt"])

        return self._generate_response(sample, inputs, sample["prompt"])

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> tuple[dict, str]:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")

        if self.model_id == "WueNLP/centurio_qwen":
            prompt = "You are a helpful assistant. " + prompt

        num_image_placeholders = prompt.count(IMAGE_PLACEHOLDER_TOKEN)

        if num_image_placeholders == 0 and len(images) == 1:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt
            num_image_placeholders = 1

        if num_image_placeholders != len(images):
            raise ValueError(
                f"Number of images ({len(images)}) must match the number "
                f"of {IMAGE_PLACEHOLDER_TOKEN} in the prompt ({num_image_placeholders})"
            )
        final_prompt = self._apply_chat_template(prompt)
        inputs = (
            self.tokenizer(images=images, text=final_prompt, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )

        return inputs, final_prompt

    def _prepare_model_input_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> tuple[dict, str]:
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

    def _prepare_model_inputs_for_text(self, prompt: str) -> dict:
        final_prompt = self._apply_chat_template(prompt)
        return (
            self.tokenizer(text=final_prompt, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )

    def compute_answer_perplexity(
        self, sample: GimmickSample, **kwargs
    ) -> AnswerPerplexityResult:
        # First, get the tokens for the question and the ground truth answer
        if self.model_id == "WueNLP/centurio_qwen":
            eos_token = "<|im_end|>"
        else:
            eos_token = "<EOS_TOKEN>"

        if "images" not in sample:
            raise ValueError("Sample must contain images")

        model_inputs, prompt = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )
        question_tokens = model_inputs.input_ids.cpu()

        gt_answer_tokens = self.tokenizer(
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
