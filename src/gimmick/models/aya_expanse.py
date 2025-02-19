from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType


class AyaExpanse(BaselineModel):
    def __init__(
        self,
        model_id: str = "CohereForAI/aya-expanse-8b",
    ):
        if model_id != "CohereForAI/aya-expanse-8b":
            raise ValueError(f"Invalid model ID: {model_id}")

        super().__init__(model_id=model_id)

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            .eval()
            .cuda()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type == TaskType.TEXT_QA

    def _generate_response(
        self,
        sample: GimmickSample,
        prompt: str,
        **kwargs,
    ) -> GimmickModelResponse:
        input_ids = self._prepare_model_input_for_text(prompt)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            **self.generation_kwargs,
            **kwargs,
        )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(input_ids, generated_ids)
        ]

        output_texts = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )

        answer = output_texts[0]

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=prompt,
            ground_truth=sample["ground_truth"],
            response=answer,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def _prepare_model_input_for_text(self, prompt: str) -> Any:
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)  # type: ignore

        return input_ids

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        return self._generate_response(
            sample,
            prompt=sample["prompt"],
            pixel_values=None,
            num_patches_list=None,
            **kwargs,
        )
