from pathlib import Path

import pandas as pd
from loguru import logger

from gimmick.tasks.sample import GimmickSample, GimmickSampleBase


class GimmickModelResponse(GimmickSampleBase):
    """
    This class represents a response from a model in the Gimmick framework.
    Besides the sample data, it includes the ID of the model that generated the response.
    """

    model_id: str
    response: str


def model_responses_to_df(model_responses: list[GimmickModelResponse]) -> pd.DataFrame:
    if len(model_responses) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(model_responses)
    if not df.task_id.nunique() == 1:
        raise ValueError("All results must be from the same task!")
    return df


def store_model_responses(
    output_fn: Path,
    results: list[GimmickModelResponse],
) -> None:
    df = model_responses_to_df(results)
    output_fn.parent.mkdir(parents=True, exist_ok=True)
    if not output_fn.suffix == ".jsonl":
        output_fn = output_fn.with_suffix(".jsonl")
    df.to_json(output_fn, orient="records", lines=True, force_ascii=False)

    logger.debug(f"Stored {len(df)} results in {output_fn}")


def load_model_responses(
    output_fn: Path,
) -> list[GimmickModelResponse]:
    if not output_fn.exists():
        logger.warning(f"Model responses file not found: {output_fn}")
        return []
    df = pd.read_json(output_fn, orient="records", lines=True)
    if not all(k in df.columns for k in GimmickModelResponse.__annotations__):
        logger.warning(f"Invalid model responses file: {output_fn}")
        return []
    logger.info(f"Loaded {len(df)} model responses from {output_fn}")
    return df.to_dict(orient="records")  # type: ignore


def is_sample_in_model_responses(
    model_responses: list[GimmickModelResponse],
    sample: GimmickSample,
) -> bool:
    if len(model_responses) == 0:
        return False
    df = model_responses_to_df(model_responses)
    rows = df[
        (df["sample_id"] == sample["sample_id"])
        & (df["task_id"] == sample["task_id"])
        & (df["hints"] == sample["hints"])
    ]
    return len(rows) == 1
