from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from decord import AudioReader, VideoReader
from loguru import logger
from PIL.Image import Image

from gimmick.tasks.perplexity import AnswerPerplexityResult
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType


class BaselineModel(ABC):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        logger.info(f"Loading model {self.model_id}")

    def generate_response(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        """
        Generate a response for the given sample.

        Args:
            sample: A dictionary containing the sample data.
            **kwargs: Additional keyword arguments.

        Returns:
            The response.
        """
        with torch.no_grad():
            if "images" in sample and len(sample["images"]) > 0:
                if any(image is None for image in sample["images"]):
                    raise ValueError("Images must not be None!")
                response = self.generate_response_for_images(sample, **kwargs)
            elif "videos" in sample and len(sample["videos"]) > 0:
                if any(video is None for video in sample["videos"]):
                    raise ValueError("Videos must not be None!")
                response = self.generate_response_for_videos(sample, **kwargs)
            elif "audios" in sample and len(sample["audios"]) > 0:
                if any(audio is None for audio in sample["audios"]):
                    raise ValueError("Audios must not be None!")
                response = self.generate_response_for_audios(sample, **kwargs)
            else:
                if not sample["prompt"]:
                    raise ValueError("Prompt must not be None!")
                response = self.generate_response_for_text(sample, **kwargs)
        return response

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    def generate_response_for_audios(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    def _prepare_model_input_for_text(
        self,
        prompt: str,
    ) -> Any:
        raise NotImplementedError

    def _prepare_model_input_for_images(
        self,
        prompt: str,
        images: list[Image],
    ) -> Any:
        raise NotImplementedError

    def _prepare_model_input_for_videos(
        self,
        prompt: str,
        videos: list[VideoReader | str | Path],
    ) -> Any:
        raise NotImplementedError

    def _prepare_model_input_for_audios(
        self,
        prompt: str,
        audios: list[AudioReader | str | Path],
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def supports_task(self, task_type: TaskType) -> bool:
        """
        Check if the model supports the given task type.

        Args:
            task_type: The task type.

        Returns:
            True if the model supports the task type, False otherwise.
        """
        raise NotImplementedError

    def compute_answer_perplexity(
        self, sample: GimmickSample, **kwargs
    ) -> AnswerPerplexityResult:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.model_id.lower().replace("/", "__")
