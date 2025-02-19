from enum import Enum, unique


@unique
class TaskType(str, Enum):
    SINGLE_IMAGE_QA = "single_image_qa"
    MULTI_IMAGE_QA = "multi_image_qa"
    TEXT_QA = "text_qa"
    VIDEO_QA = "video_qa"
    AUDIO_QA = "audio_qa"
