import os
import time
from collections import deque
from typing import Literal, Any

import backoff
import vertexai
from google.oauth2.service_account import Credentials
from openai import BadRequestError, OpenAI
from PIL import Image
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    SafetySetting,
)
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType


from gimmick.models.baseline_model import BaselineModel
from gimmick.utils.image import image_to_b64


class GPT4oBase(BaselineModel):
    def __init__(
        self,
        model_id: str,
        api_key: str | None,
        max_img_size: int = 640,
        img_detail: Literal["low", "auto", "high"] = "low",
        system_prompt: str | None = None,
    ):
        super().__init__(
            model_id=model_id,
        )

        if not model_id.startswith("o1") and not model_id.startswith("gpt-4o"):
            raise ValueError("Please provide a valid model_id for GPT-4o or O1 models!")

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("Please provide an OpenAI API key!")
        self.api_key = api_key

        self.client = OpenAI(api_key=api_key)
        if not self.client:
            raise ValueError(f"OpenAI model {model_id} could not be loaded.")
        self.img_detail = img_detail
        self.max_img_size = max_img_size
        self.system_prompt = system_prompt

    def _create_image_parts(
        self, images: Image.Image | list[Image.Image] | None
    ) -> list:
        if images is None:
            return []
        if not isinstance(images, list):
            images = [images]
        base64_image_urls = [
            image_to_b64(image, return_url=True) for image in images for image in images
        ]
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_image,
                    "detail": self.img_detail,
                },
            }
            for base64_image in base64_image_urls
        ]

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

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        raise NotImplementedError

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def generate_response(
        self,
        prompt: str | dict[str, Any],
        images: Image.Image | list[Image.Image] | None = None,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        try:
            messages = []
            if self.system_prompt and self.system_prompt != "":
                messages.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )
            if isinstance(prompt, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *self._create_image_parts(images),
                        ],
                    }
                )
            elif isinstance(prompt, dict):
                messages.append(prompt)
            else:
                raise ValueError("Invalid prompt format!")

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
        except Exception as e:
            if isinstance(e, BadRequestError) and e.code == "content_policy_violation":
                print(f"Content Policy Violation!: {e}")
                return "Content Policy Violation!"
            print(f"Error when getting response content: {e}")
            raise e
        try:
            return response.choices[0].message.content.strip()
        except Exception as e:
            if isinstance(e, (ValueError, IndexError)):
                print(f"Error when getting response content: {e}")
                return "Error when getting response content!"
            raise e


class GeminiRegionRotator:
    def __init__(self, rpm_limit: int = 5):
        """
        Initialize the GeminiRegionRotator with a rate limit per minute per region.

        :param rpm_limit: Maximum number of requests allowed per minute per region
        """
        self._available_regions = [
            "asia-east1",
            "asia-east2",
            "asia-northeast1",
            "asia-northeast3",
            "asia-south1",
            "asia-southeast1",
            "australia-southeast1",
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
            "me-central1",
            "me-central2",
            "me-west1",
            "northamerica-northeast1",
            "southamerica-east1",
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-south1",
            "us-west1",
            "us-west4",
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


class GeminiBase(ApiModelBase):
    def __init__(
        self,
        model_id: str,
        google_project_id: str | None = None,
        google_service_account_file: str | None = None,
        max_img_size: int = 1024,
        system_prompt: str | None = None,
        region: str | None = None,
    ):
        super().__init__(
            model_id=model_id,
            model_family="gemini",
            max_img_size=max_img_size,
        )
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
        if region is None:
            self._region_rotator = GeminiRegionRotator(rpm_limit=5)
            self.fixed_region = None
        else:
            self._region_rotator = GeminiRegionRotator(rpm_limit=5)
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
            time.sleep(1)
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
        base64_images = [
            self.image_to_b64(image) for image in images for image in images
        ]
        return [Part.from_data(b64, mime_type="image/png") for b64 in base64_images]  # type: ignore

    @backoff.on_exception(backoff.expo, Exception, max_time=120)
    def generate_response(
        self,
        prompt: str | Part | list[Part],
        images: Image.Image | list[Image.Image] | None = None,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        try:
            gen_config = GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
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
            msg = f"Error when getting response content: {e}"
            print(msg)
            return msg
        finally:
            self._region_rotator.mark_request(self._last_region)

        return response_text


class GPT4o20240513(GPT4oBase):
    def __init__(
        self,
        api_key: str | None = None,
        img_detail: Literal["low", "auto", "high"] = "low",
        system_prompt: str | None = None,
    ):
        super().__init__(
            model_id="gpt-4o-2024-05-13",
            api_key=api_key,
            img_detail=img_detail,
            system_prompt=system_prompt,
        )


class GPT4o20240806(GPT4oBase):
    def __init__(
        self,
        api_key: str | None = None,
        img_detail: Literal["low", "auto", "high"] = "low",
        system_prompt: str | None = None,
    ):
        super().__init__(
            model_id="gpt-4o-2024-08-06",
            api_key=api_key,
            img_detail=img_detail,
            system_prompt=system_prompt,
        )


class GPT4oMini20240718(GPT4oBase):
    def __init__(
        self,
        api_key: str | None = None,
        img_detail: Literal["low", "auto", "high"] = "low",
        system_prompt: str | None = None,
    ):
        super().__init__(
            model_id="gpt-4o-mini-2024-07-18",
            api_key=api_key,
            img_detail=img_detail,
            system_prompt=system_prompt,
        )


class Gemini15Flash001(GeminiBase):
    def __init__(
        self,
        google_project_id: str | None = None,
        google_service_account_file: str | None = None,
        system_prompt: str | None = None,
        region: str | None = None,
    ):
        super().__init__(
            model_id="gemini-1.5-flash-001",
            google_project_id=google_project_id,
            google_service_account_file=google_service_account_file,
            system_prompt=system_prompt,
            region=region,
        )


class Gemini15Pro001(GeminiBase):
    def __init__(
        self,
        google_project_id: str | None = None,
        google_service_account_file: str | None = None,
        system_prompt: str | None = None,
        region: str | None = None,
    ):
        super().__init__(
            model_id="gemini-1.5-pro-001",
            google_project_id=google_project_id,
            google_service_account_file=google_service_account_file,
            system_prompt=system_prompt,
            region=region,
        )


class Gemini15Flash002(GeminiBase):
    def __init__(
        self,
        google_project_id: str | None = None,
        google_service_account_file: str | None = None,
        system_prompt: str | None = None,
        region: str | None = None,
    ):
        super().__init__(
            model_id="gemini-1.5-flash-002",
            google_project_id=google_project_id,
            google_service_account_file=google_service_account_file,
            system_prompt=system_prompt,
            region=region,
        )


class Gemini15Pro002(GeminiBase):
    def __init__(
        self,
        google_project_id: str | None = None,
        google_service_account_file: str | None = None,
        system_prompt: str | None = None,
        region: str | None = None,
    ):
        super().__init__(
            model_id="gemini-1.5-pro-002",
            google_project_id=google_project_id,
            google_service_account_file=google_service_account_file,
            system_prompt=system_prompt,
            region=region,
        )
