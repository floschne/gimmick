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


class GimmickVVQA(GimmickTask):
    def __init__(
        self,
        lmm_judge: str | None = None,
        use_decord_video_reader: bool = False,
        min_center_frame_similarity: float = 0.5,
        **kwargs,
    ):
        self.lmm_judge = lmm_judge
        self.use_decord_video_reader = use_decord_video_reader
        self.min_center_frame_similarity = min_center_frame_similarity
        super().__init__(
            ablate_region_and_country_hints=True,
            add_answer_with_single_word_hint=True,
            **kwargs,
        )

    def load_base_dataset(self) -> Dataset:
        ds = load_dataset(
            "floschne/gimmick-vvqa",
            split="test",
            trust_remote_code=True,
        )
        for video_fn in ds["video_fn"]:
            video_fn = Path(video_fn)
            if not video_fn.exists():
                raise FileNotFoundError(f"Video {video_fn} does not exist!")

        ds = ds.rename_column("question", "prompt")
        ds = ds.rename_column("answer", "ground_truth")
        ds = ds.rename_column("question_id", "sample_id")

        # filter out samples with low center frame similarity
        ds = ds.filter(
            lambda x: x["video_center_frame_sim"] >= self.min_center_frame_similarity
        )

        return ds  # type: ignore

    def get_sample_payload(
        self, sample: dict
    ) -> list[Image] | list[AudioReader | str | Path] | list[VideoReader | str | Path]:
        return [sample["video_fn"]]

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
        return "Gimmick Video QA"

    @property
    def task_id(self) -> str:
        return "gimmick-vvqa"

    @property
    def task_type(self) -> TaskType:
        return TaskType.VIDEO_QA
