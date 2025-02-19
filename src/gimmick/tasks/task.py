import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset
from decord import AudioReader, VideoReader
from loguru import logger
from PIL.Image import Image
from tqdm.auto import tqdm

from gimmick.models.baseline_model import BaselineModel
from gimmick.models import is_model_supported
from gimmick.tasks.response import (
    GimmickModelResponse,
    is_sample_in_model_responses,
    load_model_responses,
    store_model_responses,
)
from gimmick.tasks.sample import GimmickSample, GimmickSampleBase
from gimmick.tasks.types import TaskType


class GimmickTask(ABC):
    def __init__(
        self,
        ablate_region_and_country_hints: bool = True,
        add_answer_with_single_word_hint: bool = True,
        add_culural_specific_terms_hint: bool = False,
        **kwargs,
    ):
        self.ablate_region_and_country_hints = ablate_region_and_country_hints
        self.add_answer_with_single_word_hint = add_answer_with_single_word_hint
        self.add_culural_specific_terms_hint = add_culural_specific_terms_hint

        logger.info(f"Loading dataset for {self.name}")
        self.dataset = self.load_base_dataset()

        # assert that the dataset contains the required columns
        required_columns = [
            "sample_id",
            "prompt",
            "ground_truth",
            "countries",
            "regions",
        ]
        for col in required_columns:
            if col not in self.dataset.column_names:
                raise ValueError(f"Column {col} is required in the dataset")

        logger.info(f"Loaded dataset with {len(self.dataset)} unique samples")
        if self.ablate_region_and_country_hints:
            self.hint_options = ["regions", "countries", "both", "none"]
        else:
            self.hint_options = ["none"]

    def run_eval(
        self,
        model: BaselineModel | None,
        output_root: Path,
        max_failed_samples: int | float = 0.025,
        only_recompute_scores: bool = False,
        model_name: str | None = None,
        **kwargs,
    ) -> dict[str, float]:
        if only_recompute_scores:
            if model_name is None:
                raise ValueError(
                    "model_name is required when only_recompute_scores is True"
                )
            model_id = model_name.lower().replace("__", "/")
            is_model_supported(model_id, raise_error=True)
        else:
            if model is None:
                raise ValueError(
                    "model is required when only_recompute_scores is False"
                )
            model_name = model.name

        output_dir = Path(output_root) / self.task_id / model_name
        responses_ofn = output_dir / "responses.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Responses file: {responses_ofn}")

        responses, samples = self.generate_responses(
            model=model,
            responses_ofn=responses_ofn,
            max_failed_samples=max_failed_samples,
            only_recompute_scores=only_recompute_scores,
            model_name=model_name,
            **kwargs,
        )
        scores = self.evaluate_results(
            output_dir=output_dir,
            responses=responses,
            samples=samples,
            **kwargs,
        )

        return scores

    def evaluate_results(
        self,
        output_dir: Path,
        responses: list[GimmickModelResponse],
        # we need the samples here for the LMM Judge that needs the payload
        samples: list[GimmickSample],
        **kwargs,
    ):
        logger.info(f"Computing scores for {self.name}")

        # since one sample can have multiple regions, we add the sample to all the regions it belongs to
        # and then compute the scores for each region.
        # Also, we group the samples and responses based on the hints.
        grouped_data = {}
        for sample, resp in zip(samples, responses):
            for region in sample["regions"] + ["all"]:
                if region not in grouped_data:
                    grouped_data[region] = {
                        hint: {"samples": [], "responses": []}
                        for hint in self.hint_options
                    }
                for hint in self.hint_options:
                    if hint == sample["hints"]:
                        grouped_data[region][hint]["samples"].append(sample)
                        grouped_data[region][hint]["responses"].append(resp)

        scores = {}
        for region, region_data in grouped_data.items():
            scores[region] = {}
            for hint, hint_data in region_data.items():
                logger.info(
                    f"Computing scores for {self.name} with Hint(s): {hint} in Region: {region}"
                )
                scores[region][hint] = self.compute_scores(
                    hint_data["responses"],
                    hint_data["samples"],
                    **kwargs,
                )

        logger.info(f"Scores: {json.dumps(scores, indent=2)}")
        scores_fn = output_dir / "scores.json"
        with open(scores_fn, "w") as f:
            json.dump(scores, f, indent=2)
        logger.info(f"Stored scores in {scores_fn}")
        return scores

    def generate_responses(
        self,
        model: BaselineModel | None,
        responses_ofn: Path,
        max_failed_samples: int | float,
        only_recompute_scores: bool,
        model_name: str | None,
        **kwargs,
    ) -> tuple[list[GimmickModelResponse], list[GimmickSample]]:
        """
        This is the main method that generates responses for the task using the model.
        """
        if isinstance(max_failed_samples, float):
            max_failed_samples = int(max_failed_samples * len(self.dataset))

        if only_recompute_scores:
            if not responses_ofn.exists():
                raise ValueError(
                    f"Responses file {responses_ofn} does not exist. Cannot recompute scores"
                )
            logger.info(
                f"Only recomputing scores. Loading responses from {responses_ofn}"
            )
        else:
            logger.info(f"Generating responses for {self.name} with {model_name}")

        responses: list[GimmickModelResponse] = load_model_responses(responses_ofn)

        samples: list[GimmickSample] = []
        failed_samples = 0

        for hints in tqdm(
            self.hint_options,
            desc="Iterating Hints",
            leave=False,
            position=0,
        ):
            if only_recompute_scores:
                pbar_msg = f"Collecting responses for {self.name} with Hint(s): {hints}"
            else:
                pbar_msg = f"Generating responses for {self.name} with Hint: {hints}"

            for sample in tqdm(
                self.dataset,
                desc=pbar_msg,
                total=len(self.dataset),
                leave=False,
                position=1,
            ):
                sample = dict(sample)
                prepared_sample = self.prepare_sample(sample, hints)
                samples.append(prepared_sample)

                if only_recompute_scores and not is_sample_in_model_responses(
                    responses, prepared_sample
                ):
                    logger.warning(
                        f"Sample {prepared_sample['sample_id']} is not in responses. Skipping..."
                    )
                    continue
                elif not only_recompute_scores:
                    if is_sample_in_model_responses(responses, prepared_sample):
                        logger.debug(
                            f"Skipping sample {prepared_sample['sample_id']} as it is already in responses"
                        )
                        continue
                    else:
                        if model is None:
                            raise ValueError(
                                "model is required when only_recompute_scores is False"
                            )
                        try:
                            response = model.generate_response(
                                prepared_sample, **kwargs
                            )
                            responses.append(response)
                        except Exception as e:
                            logger.error(
                                f"Failed to generate response for sample: {sample}"
                            )
                            logger.error(e)
                            failed_samples += 1
                            if failed_samples >= max_failed_samples:
                                msg = f"Failed to generate responses for {failed_samples} samples. Stopping..."
                                logger.error(msg)
                                raise SystemError(msg)

                if not only_recompute_scores and (
                    len(responses) % 100 == 0 or len(responses) == 10
                ):
                    store_model_responses(responses_ofn, responses)

            if not only_recompute_scores:
                store_model_responses(responses_ofn, responses)

        return responses, samples

    def prepare_sample(self, sample: dict, hints: str) -> GimmickSample:
        """
        This method prepares the GimmickSample (with payload) object that is used to generate responses for the model.

        Args:
            sample: The sample from the HF dataset
            hints: The hint to be used in the prompt for the sample

        Returns:
            GimmickSample: The prepared GimmickSample object (with the image, video, or audio payload)
        """
        base_sample = self.build_sample_base(sample, hints)
        sample_payload = self.get_sample_payload(sample)

        if self.task_type == TaskType.SINGLE_IMAGE_QA:
            payload_key = "images"
        elif self.task_type == TaskType.MULTI_IMAGE_QA:
            payload_key = "images"
        elif self.task_type == TaskType.VIDEO_QA:
            payload_key = "videos"
        elif self.task_type == TaskType.AUDIO_QA:
            payload_key = "audios"
        elif self.task_type == TaskType.TEXT_QA:
            payload_key = None

        prepared_sample = GimmickSample(
            **({payload_key: sample_payload} if payload_key else {}),  # type: ignore
            **base_sample,
        )

        return prepared_sample

    def build_sample_base(
        self,
        sample: dict,
        hints: str,
    ) -> GimmickSampleBase:
        """
        This method builds the base sample object that is common across all tasks.

        Args:
            sample: The sample from the HF dataset
            hints: The hint to be used in the prompt for the sample

        Returns:
            GimmickSampleBase: The base sample object (without the payload)
        """
        regions_hint = "regions" in hints or "both" in hints
        countries_hint = "countries" in hints or "both" in hints

        prompt = self.apply_gimmick_prompt_template(
            sample,
            regions_hint=regions_hint,
            countries_hint=countries_hint,
            answer_with_single_word_hint=self.add_answer_with_single_word_hint,
            culural_specific_terms_hint=self.add_culural_specific_terms_hint,
        )

        base_sample = GimmickSampleBase(
            task_id=self.task_id,
            # this has to be a unique key in the HF dataset
            sample_id=sample["sample_id"],
            ground_truth=sample["ground_truth"],
            countries=sample["countries"],
            regions=sample["regions"],
            hints=hints,
            prompt=prompt,
        )

        return base_sample

    def apply_gimmick_prompt_template(
        self,
        sample: dict,
        regions_hint: bool,
        countries_hint: bool,
        answer_with_single_word_hint: bool = True,
        culural_specific_terms_hint: bool = False,
    ) -> str:
        prompt_template = "{QUESTION}\n{HINTS}\n"
        if answer_with_single_word_hint:
            prompt_template += "Answer with a single word or phrase."
        hints = ""

        if culural_specific_terms_hint:
            hints += "Prefer culutral specific terms over a general ones.\n"

        if regions_hint:
            hints += (
                "Hint: The task is related to a cultural event or facet from the following region(s): "
                f"{', '.join(sample['regions'])}\n"
            )

        if countries_hint:
            hints += (
                "Hint: The task is related to a cultural event or facet from the following country or countries: "
                f"{', '.join(sample['countries'])}\n"
            )

        return prompt_template.format(
            QUESTION=sample["prompt"],
            HINTS=hints,
        )

    @abstractmethod
    def load_base_dataset(self) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def compute_scores(
        self,
        responses: list[GimmickModelResponse],
        samples: list[GimmickSample],
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_sample_payload(
        self, sample: dict
    ) -> list[Image] | list[AudioReader | str | Path] | list[VideoReader | str | Path]:
        """
        Depending on the task type, this method should return the payload for the sample. The payload is a list of
        Image, VideoReader, AudioReader, str or Path objects.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset) * len(self.hint_options)
