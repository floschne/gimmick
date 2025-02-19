from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType


class Phi3_5_Mini(BaselineModel):
    def __init__(
        self,
        model_id: str = "microsoft/Phi-3.5-mini-instruct",
    ):
        if model_id != "microsoft/Phi-3.5-mini-instruct":
            raise ValueError(f"Invalid model ID: {model_id}")

        super().__init__(model_id=model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type == TaskType.TEXT_QA

    def _generate_response(
        self,
        sample: GimmickSample,
        prompt: str,
        **kwargs,
    ) -> GimmickModelResponse:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": sample["prompt"]},
        ]

        output = self.pipe(
            messages,
            **self.generation_kwargs,
            return_full_text=False,
        )
        answer = str(output[0]["generated_text"]).strip()  # type: ignore

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
