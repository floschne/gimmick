from pathlib import Path
from typing import NotRequired, TypedDict

from decord import AudioReader, VideoReader
from PIL.Image import Image


class GimmickSampleBase(TypedDict):
    """
    Base class for GimmickSamples. This is common to all tasks and does not include the multimodal task-specific data.
    """

    task_id: str
    sample_id: str
    ground_truth: str
    countries: list[str]
    regions: list[str]
    hints: str
    prompt: str


class GimmickSample(GimmickSampleBase):
    """
    This class represents a sample in the Gimmick framework. It includes task-specific multimodal data such as images, audios, and videos.
    """

    images: NotRequired[list[Image]]
    audios: NotRequired[list[AudioReader | str | Path]]
    videos: NotRequired[list[VideoReader | str | Path]]
