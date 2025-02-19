from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from decord import AudioReader, VideoReader
from PIL.Image import Image

from gimmick.eval.lmm_judge import load_lmm_judge, run_lmm_as_a_judge_scoring
from gimmick.eval.metrics import compute_match_accuracy
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.task import GimmickTask
from gimmick.tasks.types import TaskType


class GimmickSIVQA(GimmickTask):
    def __init__(
        self,
        without_images: bool = False,
        lmm_judge: str | None = None,
        **kwargs,
    ):
        self.lmm_judge = lmm_judge
        self.without_images = without_images

        super().__init__(
            ablate_region_and_country_hints=True,
            add_answer_with_single_word_hint=True,
            **kwargs,
        )

    def load_base_dataset(self) -> Dataset:
        base_ds = load_dataset("floschne/gimmick-sivqa", split="test")

        base_ds = base_ds.rename_column("question", "prompt")
        base_ds = base_ds.rename_column("answer", "ground_truth")
        base_ds = base_ds.rename_column("question_id", "sample_id")

        return base_ds  # type: ignore

    def get_sample_payload(
        self, sample: dict
    ) -> list[Image] | list[AudioReader | str | Path] | list[VideoReader | str | Path]:
        if self.without_images:
            return []
        return [sample["ich_image"]]

    def compute_scores(
        self,
        responses: list[GimmickModelResponse],
        samples: list[GimmickSample],
        **kwargs,
    ) -> dict[str, Any]:
        preds = [r["response"] for r in responses]
        targets = [r["ground_truth"] for r in responses]
        scores: dict = compute_match_accuracy(preds, targets)
        if self.lmm_judge is not None:
            judge = load_lmm_judge(self.lmm_judge)
            judge_acc = run_lmm_as_a_judge_scoring(
                judge=judge, samples=samples, responses=responses, **kwargs
            )
            scores.update(judge_acc)
        return scores

    @property
    def name(self) -> str:
        text_only = " -- Text Only" if self.without_images else ""
        return "Gimmick Single Image VQA" + text_only

    @property
    def task_id(self) -> str:
        text_only = "-text-only" if self.without_images else ""
        return "gimmick-sivqa" + text_only

    @property
    def task_type(self) -> TaskType:
        if self.without_images:
            return TaskType.TEXT_QA
        return TaskType.SINGLE_IMAGE_QA
