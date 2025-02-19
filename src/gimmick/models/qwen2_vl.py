import tempfile
from pathlib import Path
from typing import Any

import torch
from decord import VideoReader
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.perplexity import (
    AnswerPerplexityResult,
    compute_perplexity_and_avg_log_likelihood,
)
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import (
    split_prompt_by_image_tokens,
    IMAGE_PLACEHOLDER_TOKEN,
)
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
)


class Qwen2VL(BaselineModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        video_fps: int = 1,
        **kwargs,
    ):
        if not model_id.lower().startswith("qwen/qwen2-vl-"):
            raise KeyError(f"Model ID {model_id} not supported")
        super().__init__(model_id=model_id)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()

        self.video_fps = video_fps

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _generate_response(self, sample: GimmickSample, inputs, chat_messages):
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=str(chat_messages),
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

        inputs, chat_messages = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )

        return self._generate_response(sample, inputs, chat_messages)

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        inputs, chat_messages = self._prepare_model_input_for_videos(
            sample["prompt"], sample["videos"]
        )

        return self._generate_response(sample, inputs, chat_messages)

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        inputs, chat_messages = self._prepare_model_input_for_text(
            sample["prompt"],
        )

        return self._generate_response(sample, inputs, chat_messages)

    def _prepare_model_input_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> Any:
        if not isinstance(videos, list):
            videos = [videos]

        if len(videos) == 0:
            raise ValueError("At least one video is required")
        elif len(videos) > 1:
            raise ValueError("Only one video is supported")

        video_path = videos[0]
        if not isinstance(video_path, (str, Path)):
            raise ValueError("Video path must be a string or a Path")
        video_path = str(video_path)

        # we only allow one video before the prompt
        if prompt.count(VIDEO_PLACEHOLDER_TOKEN) > 1:
            raise ValueError("Only one video placeholder is supported")
        prompt = prompt.replace(VIDEO_PLACEHOLDER_TOKEN, "")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "fps": self.video_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda", torch.bfloat16)

        return inputs, messages

    def _prepare_model_input_for_images(self, prompt: str, images: list[Image]) -> Any:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")
        if len(images) == 1 and prompt.count(IMAGE_PLACEHOLDER_TOKEN) == 0:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            images_paths = []
            for img in images:
                img_fn = f"{tmpdir}/image_{len(images_paths)}.png"
                img.save(img_fn)
                images_paths.append("file://" + img_fn)

            prompt_parts = split_prompt_by_image_tokens(prompt, num_images=len(images))
            content = []
            for i in range(len(images)):
                # Append the text part (prompt_parts[i])
                if prompt_parts[i]:  # Only add if it's not empty
                    content.append({"type": "text", "text": prompt_parts[i]})

                # Append the image
                content.append({"type": "image", "image": str(images_paths[i])})

            # Append the final text part
            if prompt_parts[-1]:
                content.append({"type": "text", "text": prompt_parts[-1]})

            messages = [{"role": "user", "content": content}]
            chat_messages = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            ]
            image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=chat_messages,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda", torch.bfloat16)

        return inputs, chat_messages

    def _prepare_model_input_for_text(self, prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        chat_messages = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]

        inputs = self.processor(
            text=chat_messages,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda", torch.bfloat16)
        return inputs, chat_messages

    def compute_answer_perplexity(
        self, sample: GimmickSample, **kwargs
    ) -> AnswerPerplexityResult:
        # First, get the tokens for the question and the ground truth answer
        eos_token = "<|im_end|>"
        if "images" not in sample:
            raise ValueError("Sample must contain images")

        inputs, prompt = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )
        question_tokens = inputs.input_ids.cpu()
        gt_answer_tokens = self.processor(
            text=[f"{sample['ground_truth']}{eos_token}"], return_tensors="pt"
        ).input_ids

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
