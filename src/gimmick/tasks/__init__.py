from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gimmick.tasks.task import GimmickTask


def get_all_supported_tasks():
    return ["sivqa", "vvqa", "coqa", "ckt"]


def load_task(
    task_id: str,
    **task_config,
) -> "GimmickTask":
    if task_id == "sivqa":
        from gimmick.tasks.sivqa import GimmickSIVQA

        return GimmickSIVQA(**task_config)
    elif task_id == "vvqa":
        from gimmick.tasks.vvqa import GimmickVVQA

        return GimmickVVQA(**task_config)
    elif task_id == "coqa":
        from gimmick.tasks.coqa import GimmickCOQA

        return GimmickCOQA(**task_config)
    elif task_id == "ckt":
        from gimmick.tasks.ckt import GimmickCKT

        return GimmickCKT(**task_config)
    else:
        raise NotImplementedError(
            f"Task {task_id} not supported. Supported tasks: {', '.join(get_all_supported_tasks())}"
        )
