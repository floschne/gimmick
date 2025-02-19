from pathlib import Path
from typing import TypedDict

import pandas as pd
import torch
from loguru import logger

from gimmick.tasks.sample import GimmickSample


class AnswerPerplexityResult(TypedDict):
    task_id: str
    sample_id: str
    model_id: str
    ground_truth: str
    countries: list[str]
    regions: list[str]
    hints: str
    prompt: str
    question_tokens: list[int]
    ground_truth_tokens: list[int]
    perplexity: float
    avg_log_likelihood: float


def store_perplexity_analysis_results(
    results: list[AnswerPerplexityResult], filename: str | Path
) -> Path:
    df = _ppl_analysis_results_to_df(results)
    df.to_parquet(filename, index=False)
    logger.debug(f"Stored {len(df)} Perplexity Analysis Results to {filename}")
    return Path(filename)


def load_perplexity_analysis_results(
    filename: str | Path,
) -> list[AnswerPerplexityResult]:
    if not Path(filename).exists():
        logger.warning(f"Perplexity Analysis Results file not found: {filename}")
        return []
    df = pd.read_parquet(filename)
    if not all(k in df.columns for k in AnswerPerplexityResult.__annotations__):
        logger.warning(f"Invalid Perplexity Analysis Results file: {filename}")
        return []
    logger.info(f"Loaded {len(df)} Perplexity Analysis Results from {filename}")
    return df.to_dict(orient="records")  # type: ignore


def _ppl_analysis_results_to_df(
    results: list[AnswerPerplexityResult],
) -> pd.DataFrame:
    if len(results) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    if not df.task_id.nunique() == 1:
        raise ValueError("All results must be from the same task!")
    return df


def is_sample_in_perplexity_results(
    results: list[AnswerPerplexityResult],
    sample: GimmickSample,
) -> bool:
    if len(results) == 0:
        return False
    df = _ppl_analysis_results_to_df(results)
    rows = df[
        (df["sample_id"] == sample["sample_id"])
        & (df["task_id"] == sample["task_id"])
        & (df["hints"] == sample["hints"])
    ]
    return len(rows) == 1


def compute_perplexity_and_avg_log_likelihood(
    question_answer_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    gt_answer_tokens: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[float, float]:
    question_answer_tokens = question_answer_tokens.cpu()
    question_tokens = question_tokens.cpu()
    gt_answer_tokens = gt_answer_tokens.cpu()
    logits = logits.cpu()

    # Compute log_probs for the answer tokens only
    question_length = question_tokens.size(1)

    log_likelihood = 0.0
    count = 0
    # measure perplexity for tokens in answer_tokens only
    # i.e. from indices [question_length .. question_length + len(answer_tokens) - 1]
    for i in range(question_length, question_length + gt_answer_tokens.size(1)):
        # The label at position i is the token we want to predict
        target_token_id = question_answer_tokens[0, i]
        # The logits come from the previous position i-1
        # (for the first answer token, i-1 is the last question token)
        token_logits = logits[0, i - 1, :] if i > 0 else logits[0, i, :]
        # Convert logits to log_softmax to get log-probs
        log_probs = torch.log_softmax(token_logits, dim=-1)
        # Sum log-likelihood for the target token
        log_likelihood += log_probs[target_token_id].item()
        count += 1

    # Compute perplexity as exp(-avg_log_likelihood) = exp(-log_likelihood / count)
    avg_log_likelihood = log_likelihood / count

    perplexity = torch.exp(torch.Tensor([-avg_log_likelihood])).item()

    return perplexity, avg_log_likelihood
