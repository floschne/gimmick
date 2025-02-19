import importlib
import json
import os
import random
import re
import string
from pathlib import Path

import numpy as np
import srsly
import torch
import transformers

from gimmick.tasks import get_all_supported_tasks


def seed_everything(seed: int):
    # inspired by lightning
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = "1"


def generate_random_experiment_id(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def setup_output_dir(
    output_base_dir: str | Path,
    overwrite: bool = False,
    continue_from_experiment: str | None = None,
    **config,
) -> tuple[Path, str]:
    if continue_from_experiment is not None:
        output_dir = Path(output_base_dir) / continue_from_experiment
        if not output_dir.exists():
            raise ValueError(f"Experiment directory {output_dir} does not exist. ")
        print(f"Continuing from experiment {output_dir}")
        return output_dir

    experiment_id = generate_random_experiment_id()
    output_dir = Path(output_base_dir) / experiment_id
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        print(
            f"Output directory {output_dir} already exists. "
            f"{'Overwriting' if overwrite else 'Skipping'}..."
        )
        if not overwrite:
            return

    cfg_fn = output_dir / "config.json"
    with open(cfg_fn, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment ID: {experiment_id}")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {cfg_fn}")

    return output_dir, experiment_id


def get_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }.get(dtype, torch.bfloat16)


def flash_attn_is_available() -> bool:
    return importlib.util.find_spec("flash_attn_2_cuda") is not None


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def fix_json_str(json_str: str):
    """
    Fixes JSON strings that may have missing commas between key-value pairs.

    Args:
        json_str (str): The JSON string to be fixed.

    Returns:
        str: The corrected JSON string with missing commas inserted.
    """
    if (json_str[0] != "{" and json_str[-1] != "}") and (
        json_str[0] != "[" and json_str[-1] != "]"
    ):
        json_str = f"{{{json_str}}}"

    # Pattern to match positions where a comma is missing between a value and the next key
    pattern = r'([}\]"\w])(\s*)(?="[^"]+":)'

    # Function to insert a comma where needed
    def replacer(match):
        prev_char = match.group(1)
        space = match.group(2)
        return f"{prev_char},{space}"

    # Use regex substitution to insert commas
    fixed_json = re.sub(pattern, replacer, json_str)
    return fixed_json


def find_score_files(results_root: str | Path, task_id: str | None = None):
    results_root = Path(results_root)
    all_task_ids = get_all_supported_tasks()
    if task_id is not None and task_id not in all_task_ids:
        raise ValueError(f"task_id must be one of {all_task_ids}")
    elif task_id is None:
        task_id = "*"

    glob_pattern = f"gimmick-{task_id}*/**/scores.json"
    results = list(results_root.glob(glob_pattern))
    return results


def find_response_files(results_root: str | Path, task_id: str | None = None):
    results_root = Path(results_root)
    all_task_ids = get_all_supported_tasks()
    if task_id is not None and task_id not in all_task_ids:
        raise ValueError(f"task_id must be one of {all_task_ids}")
    elif task_id is None:
        task_id = "*"

    glob_pattern = f"gimmick-{task_id}*/**/responses.jsonl"
    results = list(results_root.glob(glob_pattern))
    return results


def parse_scores_or_responses_path(results_root: str | Path, scores_path: Path):
    results_root = Path(results_root)
    scores_path = scores_path.relative_to(results_root)
    task_id = scores_path.parent.parent.name.replace("gimmick-", "")
    model = scores_path.parent.name

    return {"task_id": task_id, "model": model}


def parse_scores(results_root: str | Path, scores_path: Path):
    metadata = parse_scores_or_responses_path(
        results_root=results_root, scores_path=scores_path
    )
    scores = dict(srsly.read_json(scores_path))  # type: ignore
    return {**metadata, **scores}
