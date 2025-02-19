from typing import Any

from transformers import AutoTokenizer, AutoModelForCausalLM

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType


class Qwen2_5(BaselineModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        **kwargs,
    ):
        if not model_id.lower().startswith("qwen/qwen2.5-"):
            raise KeyError(f"Model ID {model_id} not supported")

        super().__init__(model_id=model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type == TaskType.TEXT_QA

    def _generate_response(self, sample: GimmickSample, inputs, chat_messages):
        generated_ids = self.model.generate(**inputs, **self.generation_kwargs)

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
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

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        inputs, chat_messages = self._prepare_model_input_for_text(
            sample["prompt"],
        )

        return self._generate_response(sample, inputs, chat_messages)

    def _prepare_model_input_for_text(self, prompt: str) -> Any:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        chat_messages = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([chat_messages], return_tensors="pt").to(
            self.model.device
        )
        return model_inputs, chat_messages
