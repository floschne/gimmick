from typing import Any

import torch
from PIL.Image import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.perplexity import (
    AnswerPerplexityResult,
    compute_perplexity_and_avg_log_likelihood,
)
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN


class Llama3_2_Vision(BaselineModel):
    def __init__(
        self,
    ):
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        super().__init__(model_id=model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type == TaskType.SINGLE_IMAGE_QA or task_type == TaskType.TEXT_QA

    def _generate_response(
        self,
        sample: GimmickSample,
        inputs: dict[str, Any],
        final_prompt: str,
    ) -> GimmickModelResponse:
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_kwargs,
        )
        # trim the input tokens from the generated output
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=final_prompt,
            ground_truth=sample["ground_truth"],
            response=output_text,
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

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        inputs, final_prompt = self._prepare_model_input_for_text(sample["prompt"])

        return self._generate_response(sample, inputs, final_prompt)

    def _prepare_model_input_for_text(self, prompt: str) -> tuple[dict[str, Any], str]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        final_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            None, final_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        return inputs, final_prompt

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image]
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(images, list):
            images = [images]
        if len(images) != 1:
            raise NotImplementedError("Only one image is supported!")

        if prompt.count(IMAGE_PLACEHOLDER_TOKEN) > 1:
            raise NotImplementedError("Only one image placeholder token is supported")

        prompt.replace(IMAGE_PLACEHOLDER_TOKEN, "")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        final_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            images[0], final_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        return inputs, final_prompt

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

        gt_answer_tokens = self.processor(
            None, [f"{sample['ground_truth']}{eos_token}"], return_tensors="pt"
        ).input_ids.cpu()

        question_answer_tokens = torch.concat(
            (question_tokens, gt_answer_tokens), dim=1
        )

        # Get logits for each position
        outputs = self.model(
            question_answer_tokens.to(self.model.device),
            labels=question_answer_tokens,
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
