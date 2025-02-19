from fire import Fire
from loguru import logger
from pathlib import Path
import wandb
from gimmick.utils import seed_everything
from gimmick.tasks import load_task, get_all_supported_tasks
from gimmick.models import (
    load_model,
    get_all_supported_baseline_models,
    is_model_supported,
)


def run_eval(
    model_id: str,
    task_name: str,
    output_root: str = "../results",
    max_failed_samples: int | float = 0.025,
    seed: int | None = 1312,
    yes: bool = False,
    wandb_logging: bool = True,
    wandb_project: str = "gimmick",
    only_recompute_scores: bool = False,
    **task_config,
):
    if seed is not None:
        logger.info(f"Setting seed to {seed}")
        seed_everything(seed)

    logger.info("Running GIMMICK Benchmark with the following parameters")
    logger.info(f"Model {model_id}")
    logger.info(f"Task: {task_name}")
    logger.info(f"Task Config: {task_config}")
    logger.info(f"Output Root: {output_root}")
    logger.info(f"Max Failed Samples: {max_failed_samples}")
    logger.info(f"Wandb Logging: {wandb_logging}")
    logger.info(f"Wandb Project: {wandb_project}")
    logger.info(f"Only Recompute Scores: {only_recompute_scores}")

    if not yes:
        y_n = input("Continue? [y/n]: ")
        if y_n.lower() != "y":
            logger.info("Exiting...")
            return
    run = None
    if only_recompute_scores:
        wandb_logging = False

    if wandb_logging:
        run = wandb.init(
            project=wandb_project,
            entity="uhh-lt",
            job_type=f"eval-{task_name}",
            config={
                "task": task_name,
                "model": model_id,
                "max_failed_samples": max_failed_samples,
                "output_root": output_root,
                "seed": seed,
            },
            tags=[task_name, model_id, "eval"],
        )

    task = load_task(task_id=task_name, **task_config)
    if only_recompute_scores:
        is_model_supported(model_id, raise_error=True)
        model = None
        model_name = model_id.lower().replace("/", "__")
    else:
        model = load_model(model_id=model_id)
        if not model.supports_task(task.task_type):
            raise ValueError(f"Model {model_id} does not support task {task_name}")
        model_name = model.name

    scores = task.run_eval(
        model=model,
        output_root=Path(output_root),
        max_failed_samples=max_failed_samples,
        only_recompute_scores=only_recompute_scores,
        model_name=model_name,
    )

    if wandb_logging:
        wandb.log(scores)
        if run is not None:
            run.summary.update(scores)


if __name__ == "__main__":
    Fire(
        {
            "eval": run_eval,
            "models": get_all_supported_baseline_models,
            "tasks": get_all_supported_tasks,
        }
    )
