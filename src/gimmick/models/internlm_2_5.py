import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType


class Intern_LM_2_5(BaselineModel):
    def __init__(
        self,
        model_id: str = "internlm/internlm2_5-7b-chat",
    ):
        if not model_id.lower().startswith("internlm/internlm2_5-"):
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type == TaskType.TEXT_QA

    def _generate_response(
        self,
        sample: GimmickSample,
        prompt: str,
        **kwargs,
    ) -> GimmickModelResponse:
        answer, _ = self.model.chat(
            self.tokenizer,
            prompt,
            **self.generation_kwargs,
        )

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
