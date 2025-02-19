from abc import ABC, abstractmethod

from PIL import Image

from gimmick.tasks.response import GimmickModelResponse
from gimmick.tasks.sample import GimmickSample
from gimmick.eval.metrics import compute_score_accuracy


class LMMJudge(ABC):
    @abstractmethod
    def score_vqa_response(
        self,
        images: list[Image.Image],
        question: str,
        response: str,
        ground_truth: str,
        avg_of_n: int = 1,
    ) -> int:
        """
        Scores a VQA response based on the image, question, response, and ground truth on a scale from 0 to 100.

        Args:
            image (list[Image.Image]): The image(s) the question is based on.
            question (str): The question the response is based on.
            response (str): The response to be evaluated.
            ground_truth (str): The ground truth response to be compared against.
            avg_of_n (int): The number of times to run the evaluation and average the results.

        Returns:
            int: The score of the response on a scale from 0 to 100. Negative values indicate an error.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The name of the LMM model.
        """
        raise NotImplementedError


def load_lmm_judge(
    hf_model_id: str,
    device: str = "cuda",
) -> LMMJudge:
    if hf_model_id in ["lmms-lab/llava-critic-7b", "lmms-lab/llava-critic-72b"]:
        from gimmick.eval.llava_critic_judge import LlavaCritic

        return LlavaCritic(hf_model_id=hf_model_id, device=device)
    else:
        raise NotImplementedError(f"Model {hf_model_id} not supported as LMM judge")


def run_lmm_as_a_judge_scoring(
    judge: LMMJudge,
    samples: list[GimmickSample],
    responses: list[GimmickModelResponse],
    avg_of_n: int = 1,
) -> dict[str, str | float]:
    if len(samples) != len(responses):
        raise ValueError("Length of samples and responses must match")
    scores = []
    for sample, response in zip(samples, responses):
        images = sample.get("images", [])
        if not images:
            raise ValueError("At least one image must be provided for each sample")
        score = judge.score_vqa_response(
            images=images,
            question=sample["prompt"],
            response=response["response"],
            ground_truth=sample["ground_truth"],
            avg_of_n=avg_of_n,
        )
        scores.append(score)
    scores = compute_score_accuracy(scores)
    return scores
