from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import datasets
from decord import AudioReader, VideoReader
from PIL.Image import Image

from gimmick.eval.lmm_judge import load_lmm_judge, run_lmm_as_a_judge_scoring
from gimmick.eval.metrics import compute_bert_score
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.task import GimmickTask
from gimmick.tasks.types import TaskType
from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN


class GimmickCKT(GimmickTask):
    def __init__(
        self,
        lmm_judge: str | None = None,
        describing_with_images_with_title: bool = False,
        describing_with_images_no_title: bool = False,
        describing_no_images_with_title: bool = False,
        naming_with_images_no_title: bool = False,
        use_stacked_images: bool = False,
        **kwargs,
    ):
        if (
            sum(
                [
                    describing_with_images_with_title,
                    describing_with_images_no_title,
                    describing_no_images_with_title,
                    naming_with_images_no_title,
                ]
            )
            != 1
        ):
            raise ValueError(
                "Exactly one of the 'describing_with_images_with_title', 'describing_with_images_no_title', 'describing_no_images_with_title', 'naming_with_images_no_title' must be True"
            )
        if describing_no_images_with_title and use_stacked_images:
            raise ValueError(
                "The combination of 'describing_no_images_with_title' and 'use_stacked_images' is not allowed"
            )

        self.describing_with_images_with_title = describing_with_images_with_title
        self.describing_with_images_no_title = describing_with_images_no_title
        self.describing_no_images_with_title = describing_no_images_with_title
        self.naming_with_images_no_title = naming_with_images_no_title

        self.lmm_judge = lmm_judge
        self.use_stacked_images = use_stacked_images

        super().__init__(
            ablate_region_and_country_hints=False,
            add_answer_with_single_word_hint=False,
            **kwargs,
        )

    def load_base_dataset(self) -> Dataset:
        base_ds = load_dataset(
            "floschne/gimmick-coqa", trust_remote_code=True, split="test"
        )

        # cast the image columns to the Image type
        base_ds = base_ds.cast_column("images", datasets.Sequence(datasets.Image()))
        base_ds = base_ds.cast_column("stacked_images", datasets.Image())

        # add column "prompt" and "groundtruth" (desc) to the dataset
        base_ds = base_ds.map(
            lambda sample: {
                **sample,
                "prompt": self._build_prompt(sample),
                "ground_truth": sample["description"],
            },
            keep_in_memory=True,
        )

        base_ds = base_ds.rename_column("sample_uuid", "sample_id")

        return base_ds  # type: ignore

    def get_sample_payload(
        self, sample: dict
    ) -> list[Image] | list[AudioReader | str | Path] | list[VideoReader | str | Path]:
        if self.task_type == TaskType.TEXT_QA:
            # text only has no payload (besides the prompt)
            return []
        if self.task_type == TaskType.SINGLE_IMAGE_QA:
            return sample["stacked_images"]

        return sample["images"]

    def compute_scores(
        self,
        responses: list[GimmickModelResponse],
        samples: list[GimmickSample],
        **kwargs,
    ) -> dict[str, Any]:
        preds = [r["response"] for r in responses]
        targets = [r["ground_truth"] for r in responses]
        scores: dict = compute_bert_score(preds, targets)
        if self.lmm_judge is not None:
            judge = load_lmm_judge(self.lmm_judge)
            judge_acc = run_lmm_as_a_judge_scoring(
                judge=judge, samples=samples, responses=responses, **kwargs
            )
            scores.update(judge_acc)
        return scores

    @property
    def name(self) -> str:
        return (
            "Gimmick Cultural Knowledge Test"
            + (
                " -- Describing with Images and Title"
                if self.describing_with_images_with_title
                else ""
            )
            + (
                " -- Describing with Images without Title"
                if self.describing_with_images_no_title
                else ""
            )
            + (
                " -- Describing without Images and with Title"
                if self.describing_no_images_with_title
                else ""
            )
            + (
                " -- Naming with Images and without Title"
                if self.naming_with_images_no_title
                else ""
            )
            + (" -- Stacked Images" if self.use_stacked_images else "")
        )

    @property
    def task_id(self) -> str:
        return (
            "gimmick-ckt"
            + (
                "-describing-with-images-and-title"
                if self.describing_with_images_with_title
                else ""
            )
            + (
                "-describing-with-images-without-title"
                if self.describing_with_images_no_title
                else ""
            )
            + (
                "-describing-without-images-with-title"
                if self.describing_no_images_with_title
                else ""
            )
            + (
                "-naming-with-images-without-title"
                if self.naming_with_images_no_title
                else ""
            )
            + ("-stacked-images" if self.use_stacked_images else "")
        )

    @property
    def task_type(self) -> TaskType:
        if self.describing_no_images_with_title:
            return TaskType.TEXT_QA
        if self.use_stacked_images:
            return TaskType.SINGLE_IMAGE_QA

        return TaskType.MULTI_IMAGE_QA

    def _build_prompt(self, sample: dict) -> str:
        prompt_template = "{TASK_PROMPT}\n\n{IMAGE_PLACEHOLDERS}\n\nYour answer: "
        title = sample["title"]
        if self.naming_with_images_no_title:
            task_prompt = "Your task is it to name the cultural event or facet depicted by the following images. Answer brief and concise. "
        elif self.describing_no_images_with_title:
            task_prompt = f"Your task is it to write a brief essay about the cultural event or facet with the title '{title}'. "
        elif self.describing_with_images_no_title:
            task_prompt = "Your task is it to write a brief essay about the cultural event or facet depicted by the following images. "
        elif self.describing_with_images_with_title:
            task_prompt = f"Your task is it to write a brief essay about the cultural event or facet depicted by the following images. It has the title '{title}'. "
        else:
            raise ValueError("Invalid configuration")

        image_placeholders = "\n".join(
            [IMAGE_PLACEHOLDER_TOKEN] * len(self.get_sample_payload(sample))
        )
        if self.describing_no_images_with_title:
            image_placeholders = ""

        prompt = prompt_template.format(
            IMAGE_PLACEHOLDERS=image_placeholders,
            TASK_PROMPT=task_prompt,
        )

        return prompt
