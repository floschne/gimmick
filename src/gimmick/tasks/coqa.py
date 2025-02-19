from pathlib import Path
from typing import Any, Literal
import random

from datasets import Dataset, load_dataset
import datasets
from decord import AudioReader, VideoReader
from PIL.Image import Image

from gimmick.eval.lmm_judge import load_lmm_judge, run_lmm_as_a_judge_scoring
from gimmick.eval.metrics import compute_match_accuracy
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.task import GimmickTask
from gimmick.tasks.types import TaskType
from gimmick.utils.ich import ICH_REGIONS_TO_COUNTRIES
from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN


class GimmickCOQA(GimmickTask):
    def __init__(
        self,
        # The parameter names actually suck, but I'm keeping them for compatibility with the plotting code etc
        without_images: bool = False,
        with_title_hint: bool = False,
        use_stacked_images: bool = False,
        sample_distractor_countries_within_regions: bool = True,
        target_origin: Literal["regions", "countries"] = "regions",
        lmm_judge: str | None = None,
        **kwargs,
    ):
        if target_origin not in ["regions", "countries"]:
            raise ValueError("Invalid target. Must be either 'regions' or 'countries'.")
        self.target_origin = target_origin

        if without_images and use_stacked_images:
            raise ValueError(
                "Cannot have both `without_images` and `use_stacked_images` set to True."
            )

        if without_images and not with_title_hint:
            raise ValueError(
                "Cannot have `without_images` set to True and `with_title_hint` set to False. That would result in a task with no prompt."
            )

        self.lmm_judge = lmm_judge
        self.without_images = without_images
        self.with_title_hint = with_title_hint
        self.use_stacked_images = use_stacked_images and not without_images
        self.sample_distractor_countries_within_regions = (
            sample_distractor_countries_within_regions
        )

        super().__init__(
            ablate_region_and_country_hints=False,
            add_answer_with_single_word_hint=False,
            **kwargs,
        )

    def load_base_dataset(self) -> Dataset:
        base_ds = load_dataset(
            "floschne/gimmick-coqa", trust_remote_code=True, split="test"
        )

        if self.target_origin == "regions":
            # Filter out samples that have more than 3 regions (otherwise we cannot generate 4 multiple choice options)
            base_ds = base_ds.filter(lambda x: len(x["regions"]) <= 3)

        # add column "target_option" to the dataset
        base_ds = base_ds.map(
            lambda sample: {
                **sample,
                "target_option": sample[self.target_origin],
            }
        )
        # explode the "target_option" column to have one row per target option
        # so that each region / country appears one time if a sample has multiple regions / countries
        base_df = base_ds.to_pandas().explode("target_option")
        base_ds = Dataset.from_pandas(base_df)

        # cast the image columns to the Image type
        base_ds = base_ds.cast_column("images", datasets.Sequence(datasets.Image()))
        base_ds = base_ds.cast_column("stacked_images", datasets.Image())

        # add columns "prompt" and "ground_truth" to the dataset
        base_ds = base_ds.map(
            lambda sample: {
                **sample,
                **self._build_prompt_and_gt_for_sample(sample),
            },
        )

        base_ds = base_ds.rename_column("sample_uuid", "sample_id")

        return base_ds  # type: ignore

    def get_sample_payload(
        self, sample: dict
    ) -> list[Image] | list[AudioReader | str | Path] | list[VideoReader | str | Path]:
        if self.without_images:
            # text-only task has no payload besides the prompt
            return []
        elif self.use_stacked_images:
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
        return (
            "Gimmick Cultural Origin QA"
            + (" -- Regions" if self.target_origin == "regions" else " -- Countries")
            + (
                " -- Text-Image"
                if not self.without_images and self.with_title_hint
                else ""
            )
            + (" -- With Title Hint" if self.with_title_hint else "")
            + (
                " -- Hard Distractors"
                if self.target_origin == "countries"
                and self.sample_distractor_countries_within_regions
                else ""
            )
            + (" -- Stacked Images" if self.use_stacked_images else "")
        )

    @property
    def task_id(self) -> str:
        hard_distractors_suffix = (
            "-hard_distractors"
            if self.target_origin == "countries"
            and self.sample_distractor_countries_within_regions
            else ""
        )
        text_only_suffix = "-text_only" if self.without_images else ""
        text_image_suffix = (
            "-text-image" if not self.without_images and self.with_title_hint else ""
        )
        stacked_suffix = "-stacked_images" if self.use_stacked_images else ""
        return (
            "gimmick-coqa"
            + f"-{self.target_origin}"
            + text_only_suffix
            + text_image_suffix
            + hard_distractors_suffix
            + stacked_suffix
        )

    @property
    def task_type(self) -> TaskType:
        if self.without_images:
            return TaskType.TEXT_QA
        elif self.use_stacked_images:
            return TaskType.SINGLE_IMAGE_QA

        return TaskType.MULTI_IMAGE_QA

    def _build_prompt_and_gt_for_sample(self, sample: dict) -> dict[str, str]:
        prompt_template = (
            "{IMAGE_PLACEHOLDERS}\n{TASK_PROMPT}\n\n{OPTIONS}\n\nYour answer letter: "
        )

        if self.without_images:
            # TEXT ONLY (no images, with title hint)
            task_prompt = (
                f"From which of the following {self.target_origin} does the cultural event or facet with the title `{sample['title']}` originate? "
                "Choose from the following options and output only the corresponding letter."
            )
            image_placeholders = ""
        elif not self.without_images and not self.with_title_hint:
            # IMAGE ONLY (with images, no title hint)
            task_prompt = (
                f"In which of the following {self.target_origin} does the event shown in the images take place? "
                "Choose from the following options and output only the corresponding letter."
            )
            image_placeholders = "\n".join(
                [IMAGE_PLACEHOLDER_TOKEN] * len(self.get_sample_payload(sample))
            )
        elif not self.without_images and self.with_title_hint:
            # TEXT-IMAGE (with images, with title hint)
            task_prompt = (
                f"From which of the following {self.target_origin} does the cultural event or facet with the title `{sample['title']}` shown in the images originate? "
                "Choose from the following options and output only the corresponding letter."
            )
            image_placeholders = "\n".join(
                [IMAGE_PLACEHOLDER_TOKEN] * len(self.get_sample_payload(sample))
            )
        else:
            raise ValueError("Invalid configuration.")

        if self.target_origin == "regions":
            valid_options = sample["regions"]
            all_options = list(ICH_REGIONS_TO_COUNTRIES.keys())
        else:
            valid_options = sample["countries"]
            all_options = []
            if self.sample_distractor_countries_within_regions:
                # valid options are countries from the regions of the sample
                for region in sample["regions"]:
                    all_options.extend(ICH_REGIONS_TO_COUNTRIES[region])

                if len(all_options) - len(valid_options) < 3:
                    # not enough distractors, so we add all countries from all regions
                    all_options.extend(
                        [
                            country
                            for countries in ICH_REGIONS_TO_COUNTRIES.values()
                            for country in countries
                        ]
                    )
            else:
                # valid options are countries from all regions
                all_options.extend(
                    [
                        country
                        for countries in ICH_REGIONS_TO_COUNTRIES.values()
                        for country in countries
                    ]
                )

        mc_options, ground_truth = generate_multiple_choice_options(
            target_option=sample["target_option"],
            valid_options=valid_options,
            all_options=all_options,
        )

        prompt = prompt_template.format(
            IMAGE_PLACEHOLDERS=image_placeholders,
            TASK_PROMPT=task_prompt,
            OPTIONS=mc_options,
        ).strip()

        return {"prompt": prompt, "ground_truth": ground_truth}


def generate_multiple_choice_options(
    target_option: str,
    valid_options: list[str],
    all_options: list[str],
):
    # Input validation
    if not isinstance(valid_options, list):
        raise ValueError("valid_options must be a list.")

    if len(valid_options) == 0:
        raise ValueError("valid_options list cannot be empty.")

    if len(all_options) - len(valid_options) < 3:
        raise ValueError("Not enough distractors to generate multiple choice options.")

    if target_option not in valid_options:
        raise ValueError("Target option not in valid options.")

    for option in valid_options:
        if option not in all_options:
            raise ValueError(
                f"Invalid valid option: '{option}'. Please choose from the available elements."
            )

    # Get distractors by excluding the valid options
    distractors_pool = list(set(all_options) - set(valid_options))

    # Select 3 unique distractors
    distractors = random.sample(distractors_pool, 3)

    # Combine correct answer with distractors
    options = distractors + [target_option]

    # Shuffle the options so the correct answer isn't always in the same position
    random.shuffle(options)

    options_str = ""
    for idx, option in enumerate(options):
        options_str += f"{(chr(65 + idx))}. {option}\n"

    target_option_letter = chr(65 + options.index(target_option))

    return options_str, target_option_letter
