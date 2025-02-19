import os
import time
from collections import deque
from pathlib import Path

import backoff
import vertexai
from decord import VideoReader
from google.oauth2.service_account import Credentials
from loguru import logger
from PIL import Image
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    SafetySetting,
)

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import (
    IMAGE_PLACEHOLDER_TOKEN,
    image_to_b64,
    split_prompt_by_image_tokens,
)
from gimmick.utils.video import (
    VIDEO_PLACEHOLDER_TOKEN,
    extract_frames_from_video,
    load_video,
)


class GeminiRegionRotator:
    def __init__(self, rpm_limit: int = 100):
        """
        Initialize the GeminiRegionRotator with a rate limit per minute per region.

        :param rpm_limit: Maximum number of requests allowed per minute per region
        """
        self._available_regions = [
            "us-central1",
            "us-east4",
            "us-west1",
            "us-west4",
            "northamerica-northeast1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west9",
            "asia-northeast1",
            "asia-northeast3",
            "asia-southeast1",
        ]
        self.rpm_limit = rpm_limit
        self._region_timestamps = {
            region: deque() for region in self._available_regions
        }
        self._current_region_index = 0

    def _get_current_region(self):
        """
        Returns the current region available for making a request.

        :return: A region identifier as a string
        """
        return self._available_regions[self._current_region_index]

    def _rotate_region(self):
        """
        Rotates to the next region in the list.
        """
        self._current_region_index = (self._current_region_index + 1) % len(
            self._available_regions
        )

    def get_available_region(self):
        """
        Finds an available region that has not exceeded the rate limit.

        :return: A region identifier as a string, or None if all regions are rate-limited
        """
        start_index = self._current_region_index
        while True:
            region = self._get_current_region()
            if self._is_region_available(region):
                return region
            self._rotate_region()
            if self._current_region_index == start_index:
                # All regions are rate-limited
                return None

    def _is_region_available(self, region):
        """
        Checks if the specified region is available for making a request.

        :param region: The region identifier to check
        :return: True if available, False otherwise
        """
        current_time = time.time()
        timestamps = self._region_timestamps[region]

        # Remove timestamps older than 60 seconds
        while timestamps and current_time - timestamps[0] > 60:
            timestamps.popleft()

        return len(timestamps) < self.rpm_limit

    def mark_request(self, region):
        """
        Records a request made to the specified region.

        :param region: The region where the request is made
        """
        self._region_timestamps[region].append(time.time())


class Gemini(BaselineModel):
    def __init__(
        self,
        model_id: str,
        google_project_id: str | None = None,
        google_service_account_file: str | None = None,
        system_prompt: str | None = None,
        region: str | None = None,
        video_fps: int = 1,
    ):
        rpm_limits = {
            "gemini-1.5-flash-001": 100,
            "gemini-1.5-flash-002": 100,
            "gemini-1.5-pro-001": 100,
            "gemini-1.5-pro-002": 100,
        }
        if model_id not in rpm_limits:
            raise ValueError(
                f"Invalid model ID: {model_id}. Valid options are: {', '.join(rpm_limits.keys())}"
            )

        super().__init__(
            model_id=model_id,
        )
        self.video_fps = video_fps

        if google_service_account_file is None:
            google_service_account_file = os.environ.get(
                "GOOGLE_SERVICE_ACCOUNT_FILE", None
            )
        if google_service_account_file is None:
            raise ValueError("Google service account file not specified in config!")

        if google_project_id is None:
            google_project_id = os.environ.get("GOOGLE_PROJECT_ID", None)
        if google_project_id is None:
            raise ValueError("Google project ID not specified in config!")

        self._google_project_id = google_project_id
        self._creds = Credentials.from_service_account_file(google_service_account_file)
        self._model_id = model_id
        self._region_rotator = GeminiRegionRotator(rpm_limit=rpm_limits[model_id])
        self._last_region = None

        if region is None:
            self.fixed_region = None
        else:
            self.fixed_region = region

        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        self.system_prompt = system_prompt

        self._model: GenerativeModel = None  # type: ignore
        self._init_model()

    def _init_model(self) -> None:
        if self.fixed_region is None:
            current_region = self._region_rotator.get_available_region()
        else:
            current_region = self.fixed_region
        while current_region is None:
            print(
                "All regions are rate-limited. Waiting for a region to become available ..."
            )
            time.sleep(5)
        if (
            self._model is None
            or self._last_region is None
            or self._last_region != current_region
        ):
            vertexai.init(
                project=self._google_project_id,
                location=current_region,
                credentials=self._creds,
            )

            model = GenerativeModel(
                self.model_id,
                system_instruction=self.system_prompt
                if self.system_prompt is not None and self.system_prompt != ""
                else None,
                safety_settings=self.safety_settings,
            )
            if model is None:
                raise ValueError(
                    f"Gemini model {self.model_id} could not be initialized."
                )
            # test the model
            counts = model.count_tokens(
                [
                    Part.from_uri(
                        "gs://cloud-samples-data/generative-ai/image/scones.jpg",
                        mime_type="image/jpeg",
                    ),
                    "What is shown in this image?",
                ]
            )
            if counts is None:
                raise ValueError(
                    f"Gemini model {self.model_id} could not be initialized."
                )

            self._model = model
            self._last_region = current_region

    def _create_image_parts(
        self, images: Image.Image | list[Image.Image] | None
    ) -> list[Part]:
        if images is None:
            return []
        if not isinstance(images, list):
            images = [images]
        base64_images = [image_to_b64(img) for img in images]
        return [Part.from_data(b64, mime_type="image/jpg") for b64 in base64_images]  # type: ignore

    @backoff.on_exception(backoff.expo, Exception, max_time=600)
    def _generate_response_str(
        self,
        prompt: str | Part | list[Part],
        images: Image.Image | list[Image.Image] | None = None,
        **kwargs,
    ) -> str:
        try:
            gen_config = GenerationConfig(
                candidate_count=1,
                max_output_tokens=self.generation_kwargs["max_new_tokens"],
                temperature=self.generation_kwargs["temperature"],
                top_p=self.generation_kwargs["top_p"],
                top_k=self.generation_kwargs["top_k"],
                **kwargs,
            )

            self._init_model()

            if isinstance(prompt, str):
                contents = [
                    Part.from_text(prompt),
                    *self._create_image_parts(images),
                ]
            elif isinstance(prompt, Part) or (
                isinstance(prompt, list) and all(isinstance(p, Part) for p in prompt)
            ):
                contents = prompt
            else:
                raise ValueError("Invalid prompt format!")

            response = self._model.generate_content(
                contents=contents,
                generation_config=gen_config,
            )
            response_text = response.text.strip()
        except Exception as e:
            if str(e).startswith(
                "Response has no candidates (and thus no text). The response is likely blocked by the safety filters."
            ):
                logger.warning("Response is blocked by the safety filters.")
                return ""
            msg = f"Error when getting response content: {e}"
            logger.warning(msg)
            raise e
        finally:
            self._region_rotator.mark_request(self._last_region)

        return response_text

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "images" not in sample:
            raise ValueError("Sample must contain 'images' key")

        parts = self._prepare_model_input_for_images(sample["prompt"], sample["images"])

        response = self._generate_response_str(parts, **kwargs)

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
            response=response,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        parts = self._prepare_model_input_for_videos(sample["prompt"], sample["videos"])

        response = self._generate_response_str(parts, **kwargs)

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
            response=response,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        response = self._generate_response_str(sample["prompt"], **kwargs)

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=sample["prompt"],
            ground_truth=sample["ground_truth"],
            response=response,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def supports_task(self, task_type: TaskType) -> bool:
        return task_type in {
            TaskType.SINGLE_IMAGE_QA,
            TaskType.MULTI_IMAGE_QA,
            TaskType.VIDEO_QA,
            TaskType.TEXT_QA,
        }

    def generate_response_for_audios(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> list[Part]:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")
        if len(images) == 1 and prompt.count(IMAGE_PLACEHOLDER_TOKEN) == 0:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        prompt_parts = split_prompt_by_image_tokens(prompt, num_images=len(images))
        parts = []
        for i in range(len(images)):
            prompt_part = prompt_parts[i]
            if len(prompt_part) > 0:
                parts.append(Part.from_text(prompt_part))
            parts.extend(self._create_image_parts(images[i]))

        # add the last part of the prompt
        if len(prompt_parts) > len(images):
            parts.append(Part.from_text(prompt_parts[-1]))

        return parts

    def _prepare_model_input_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> list[Part]:
        # Note that we intentionally do not use the actual video clip although it would be supported by Gemini
        if not isinstance(videos, list):
            videos = [videos]

        if len(videos) == 0:
            raise ValueError("At least one video is required")
        elif len(videos) > 1:
            raise ValueError("Only one video is supported")

        video = videos[0]
        if isinstance(videos[0], (str, Path)):
            video = load_video(videos[0])
        elif not isinstance(video, VideoReader):
            raise ValueError("Video must be a VideoReader or a path to a video")

        # we only allow one video before the prompt
        if prompt.count(VIDEO_PLACEHOLDER_TOKEN) > 1:
            raise ValueError("Only one video placeholder is supported")
        prompt = prompt.replace(VIDEO_PLACEHOLDER_TOKEN, "")

        frames = extract_frames_from_video(video, self.video_fps)
        for _ in frames:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt

        return self._prepare_model_input_for_images(prompt, frames)
