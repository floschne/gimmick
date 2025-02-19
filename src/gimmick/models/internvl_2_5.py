from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from gimmick.models.baseline_model import BaselineModel
from gimmick.tasks.perplexity import (
    AnswerPerplexityResult,
    compute_perplexity_and_avg_log_likelihood,
)
from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.tasks.types import TaskType
from gimmick.utils.image import IMAGE_PLACEHOLDER_TOKEN
from gimmick.utils.video import VIDEO_PLACEHOLDER_TOKEN, load_video


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "internvl2_5-1b": 24,
        "internvl2_5-2b": 24,
        "internvl2_5-4b": 36,
        "internvl2_5-8b": 32,
        "internvl2_5-26b": 48,
        "internvl2_5-38b": 64,
        "internvl2_5-78b": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = int(np.ceil(num_layers / (world_size - 0.5)))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = int(np.ceil(num_layers_per_gpu[0] * 0.5))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


class Intern_V_2_5(BaselineModel):
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-8B",
        video_fps: int = 1,
        video_frame_tiles: int = 4,
    ):
        if not model_id.lower().startswith("opengvlab/internvl2_5-"):
            raise ValueError(f"Invalid model ID: {model_id}")

        multi_gpu = torch.cuda.device_count() > 1
        if multi_gpu:
            logger.info(f"Using Multi-GPU mode for {model_id}")

        super().__init__(model_id=model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=split_model(model_id.split("/")[1].lower())
            if multi_gpu
            else "auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        self.video_fps = video_fps
        self.video_frame_tiles = video_frame_tiles

        self.generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

    def supports_task(self, task_type: TaskType) -> bool:
        return (
            task_type == TaskType.SINGLE_IMAGE_QA
            or task_type == TaskType.MULTI_IMAGE_QA
            or task_type == TaskType.VIDEO_QA
            or task_type == TaskType.TEXT_QA
        )

    def _generate_response(
        self,
        sample: GimmickSample,
        prompt: str,
        pixel_values: torch.Tensor | None,
        num_patches_list: list[int] | None,
        **kwargs,
    ) -> GimmickModelResponse:
        answer = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            self.generation_kwargs,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        return GimmickModelResponse(
            model_id=self.model_id,
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            prompt=prompt,
            ground_truth=sample["ground_truth"],
            response=answer,
            regions=sample["regions"],
            countries=sample["countries"],
            hints=sample["hints"],
        )

    def generate_response_for_images(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "images" not in sample:
            raise ValueError("Sample must contain 'images' key")

        prompt, pixel_values, num_patches_list = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )

        return self._generate_response(
            sample,
            prompt=prompt,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            **kwargs,
        )

    def generate_response_for_videos(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        if "videos" not in sample:
            raise ValueError("Sample must contain 'videos' key")

        prompt, pixel_values, num_patches_list = self._prepare_model_input_for_videos(
            sample["prompt"], sample["videos"]
        )

        return self._generate_response(
            sample,
            prompt=prompt,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            **kwargs,
        )

    def generate_response_for_text(
        self, sample: GimmickSample, **kwargs
    ) -> GimmickModelResponse:
        return self._generate_response(
            sample,
            prompt=sample["prompt"],
            pixel_values=None,
            num_patches_list=None,
            **kwargs,
        )

    def _prepare_model_input_for_images(
        self, prompt: str, images: list[Image.Image]
    ) -> tuple[str, torch.Tensor, list]:
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            raise ValueError("At least one image is required")

        num_image_placeholders = prompt.count(IMAGE_PLACEHOLDER_TOKEN)

        if num_image_placeholders == 0 and len(images) == 1:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt
            num_image_placeholders = 1

        if num_image_placeholders != len(images):
            raise ValueError(
                f"Number of images ({len(images)}) must match the number "
                f"of {IMAGE_PLACEHOLDER_TOKEN} in the prompt ({num_image_placeholders})"
            )

        if num_image_placeholders > 1:
            # according to the authors, it improves the performance to number
            # the images if there are more than one
            for i in range(1, num_image_placeholders + 1):
                prompt = prompt.replace(
                    IMAGE_PLACEHOLDER_TOKEN, f"Image-{i}: <image>", 1
                )
        else:
            prompt = prompt.replace(IMAGE_PLACEHOLDER_TOKEN, "<image>", 1)

        pixel_values = []
        num_patches_list = []
        for img in images:
            pixel_values1 = _preprocess_image(img, max_num=4).to(torch.bfloat16).cuda()
            pixel_values.append(pixel_values1)
            num_patches_list.append(pixel_values1.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)

        return prompt, pixel_values, num_patches_list

    def _prepare_model_input_for_videos(
        self, prompt: str, videos: list[VideoReader | str | Path]
    ) -> tuple[str, torch.Tensor, list]:
        if not isinstance(videos, list):
            videos = [videos]

        if len(videos) == 0:
            raise ValueError("At least one video is required")
        elif len(videos) > 1:
            raise ValueError("Only one video is supported")

        # we only allow one video before the prompt
        if prompt.count(VIDEO_PLACEHOLDER_TOKEN) > 1:
            raise ValueError("Only one video placeholder is supported")
        prompt = prompt.replace(VIDEO_PLACEHOLDER_TOKEN, "")

        video = videos[0]
        if isinstance(videos[0], (str, Path)):
            video = load_video(videos[0])
        elif not isinstance(video, VideoReader):
            raise ValueError("Video must be a VideoReader or a path to a video")

        video_duration_s = video.get_frame_timestamp(-1)[1]
        num_frames = int(video_duration_s * self.video_fps)

        pixel_values, num_patches_list = _process_video(
            video, num_segments=num_frames, max_num=self.video_frame_tiles
        )
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # according to the authors, it improves the performance to number
        # the frames
        video_prefix = "".join(
            [f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]
        )
        prompt = video_prefix + prompt

        return prompt, pixel_values, num_patches_list

    @torch.inference_mode()
    def compute_answer_perplexity(
        self, sample: GimmickSample, **kwargs
    ) -> AnswerPerplexityResult:
        """
        Computes perplexity for a sample with images by applying the conversation template and
        manually replacing the image context tokens with vision embeddings.
        """

        # (1) Check that the sample has images.
        if "images" not in sample:
            raise ValueError("Sample must contain images")

        # (2) Set the image context token for the model.
        img_context_token = "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            img_context_token
        )

        # (3) Prepare the base prompt, pixel_values, and num_patches_list.
        # The prompt here will contain the IMAGE_PLACEHOLDER_TOKEN (e.g., "<image>")
        prompt, pixel_values, num_patches_list = self._prepare_model_input_for_images(
            sample["prompt"], sample["images"]
        )

        # (4) Apply the chat template (as done in generate_response_for_images).
        # This will add system messages and role markers.
        template = deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message
        # Append the sample prompt as the user message.
        template.append_message(template.roles[0], prompt)
        # Append an empty assistant response.
        template.append_message(template.roles[1], None)
        # Build the full conversation prompt.
        query = template.get_prompt()

        # (5) Replace the "<image>" placeholder in the conversation prompt with the full image token string.
        IMG_START_TOKEN = "<img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_END_TOKEN = "</img>"
        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + (IMG_CONTEXT_TOKEN * (self.model.num_image_token * num_patches))
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        # (6) Tokenize the conversation prompt.
        model_inputs = self.tokenizer(query, return_tensors="pt", padding=True)
        question_tokens = model_inputs.input_ids.to(pixel_values.device)
        attention_mask = model_inputs.attention_mask.to(pixel_values.device)

        # (7) Prepare ground-truth answer tokens by appending the EOS token.
        eos_token = self.model.conv_template.sep
        gt_text = f"{sample['ground_truth']}{eos_token}"
        gt_inputs = self.tokenizer(gt_text, return_tensors="pt", padding=True)
        gt_answer_tokens = gt_inputs.input_ids.to(pixel_values.device)
        gt_attention_mask = gt_inputs.attention_mask.to(pixel_values.device)

        # (8) Concatenate the conversation prompt (question) tokens with the ground-truth answer tokens.
        question_answer_tokens = torch.cat([question_tokens, gt_answer_tokens], dim=1)
        combined_attention_mask = torch.cat([attention_mask, gt_attention_mask], dim=1)

        # (9) Get the initial input embeddings from the language model.
        input_embeds = self.model.language_model.get_input_embeddings()(
            question_answer_tokens
        )
        B, L, C = input_embeds.shape

        # (10) Identify positions that correspond to the IMG_CONTEXT token.
        selected = question_answer_tokens == self.model.img_context_token_id

        # (11) Extract vision embeddings using the model's extract_feature method.
        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds.reshape(-1, C)

        # (12) Ensure the number of IMG_CONTEXT tokens matches the vision embeddings.
        num_selected = selected.sum()
        if num_selected != vit_embeds.shape[0]:
            raise ValueError(
                f"Mismatch: found {num_selected} IMG_CONTEXT tokens but got {vit_embeds.shape[0]} vision embeddings."
            )

        # (13) Replace the embeddings at IMG_CONTEXT positions with the vision embeddings.
        input_embeds = input_embeds.reshape(-1, C)
        selected_flat = selected.reshape(-1)
        input_embeds[selected_flat] = vit_embeds.to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, L, C)

        # (14) Forward the replaced embeddings through the language model to obtain logits.
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=combined_attention_mask,
            labels=question_answer_tokens,
        )
        logits = outputs.logits  # Shape: [B, L, vocab_size]

        question_answer_tokens = question_answer_tokens.cpu()
        question_tokens = question_tokens.cpu()
        gt_answer_tokens = gt_answer_tokens.cpu()
        logits = logits.cpu()

        # (15) Compute perplexity and average log likelihood.
        perplexity, avg_log_likelihood = compute_perplexity_and_avg_log_likelihood(
            question_answer_tokens,
            question_tokens,
            gt_answer_tokens,
            logits,
        )

        del (
            model_inputs,
            gt_inputs,
            input_embeds,
            selected,
            vit_embeds,
            outputs,
            pixel_values,
            num_patches_list,
            selected_flat,
            combined_attention_mask,
            attention_mask,
            gt_attention_mask,
        )

        # (16) Return the result.
        res = AnswerPerplexityResult(
            task_id=sample["task_id"],
            sample_id=sample["sample_id"],
            model_id=self.model_id,
            ground_truth=sample["ground_truth"],
            countries=sample["countries"],
            regions=sample["regions"],
            hints=sample["hints"],
            prompt=query,
            question_tokens=question_tokens.squeeze().numpy().tolist(),
            ground_truth_tokens=gt_answer_tokens.squeeze().numpy().tolist(),
            perplexity=perplexity,
            avg_log_likelihood=avg_log_likelihood,
        )

        del question_answer_tokens, question_tokens, gt_answer_tokens, logits
        torch.cuda.empty_cache()

        return res


# code is based on https://huggingface.co/OpenGVLab/InternVL2_5-8B

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _preprocess_image(image: Image.Image, input_size: int = 448, max_num: int = 12):
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def _get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def _process_video(vr, bound=None, input_size=448, max_num=1, num_segments=32):
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = _build_transform(input_size=input_size)
    frame_indices = _get_frame_indices(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = _dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list
