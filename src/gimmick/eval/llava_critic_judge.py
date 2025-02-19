from gimmick.eval.lmm_judge import LMMJudge

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates

import copy
import torch
import re
import warnings
from loguru import logger
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")


class LlavaCritic(LMMJudge):
    def __init__(
        self,
        hf_model_id: str = "lmms-lab/llava-critic-7b",
        device: str = "cuda",
    ):
        supported_models = [
            "lmms-lab/llava-critic-7b",
            "lmms-lab/llava-critic-72b",
        ]
        if hf_model_id not in supported_models:
            raise ValueError(f"hf_model_id must be one of , got {hf_model_id}")

        self.hf_model_id = hf_model_id
        self.model_type = "llava_qwen"
        self.conv_template = "qwen_1_5"
        self.device_map = "auto"
        self.device = device

        self.CRITIC_PROMPT_TEMPLATE = (
            "Given an image, a corresponding question, and the ground truth answer, please serve as an "
            "unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). "
            "Score the response out of 100 and explain your reasoning with specific details. "
            "Your task is provided as follows:\n"
            "Question: [{QUESTION}]\n"
            "Ground Truth: [{GROUND_TRUTH}]\n"
            "The LMM response: [{RESPONSE}]\n"
            "ASSISTANT:\n"
        )

        logger.info(f"Loading LlavaCritic {hf_model_id}")
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                hf_model_id, None, self.model_type, device_map=self.device_map
            )
        )
        self.model.eval()

    @property
    def model_name(self) -> str:
        return self.hf_model_id.replace("/", "__")

    def _parse_critic_output(self, output_text):
        pattern = r"Score: \[(\d{1,3})\]"
        match = re.search(pattern, output_text)
        if match:
            score = int(match.group(1))
            return score

        logger.error(f"Failed to parse critic output: {output_text}")
        return -100000

    def score_vqa_response(
        self,
        images: list[Image.Image],
        question: str,
        response: str,
        ground_truth: str,
        avg_of_n: int = 1,
    ) -> int:
        if isinstance(images, Image.Image):
            images = [images]
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = [
            _image.to(dtype=torch.float16, device=self.device)
            for _image in image_tensor
        ]

        critic_prompt = self.CRITIC_PROMPT_TEMPLATE.format(
            QUESTION=question, GROUND_TRUTH=ground_truth, RESPONSE=response
        )

        question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )
        image_sizes = [image.size]

        scores = []
        with torch.no_grad():
            for _ in range(avg_of_n):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                )
                text_output = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0]

                score = self._parse_critic_output(text_output)
                scores.append(score)

        return np.rint(np.mean(scores))
