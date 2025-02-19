from functools import lru_cache
from pathlib import Path

from decord import VideoReader, cpu
from PIL import Image

VIDEO_PLACEHOLDER_TOKEN = "<video_placeholder>"


def extract_frames_from_video(video: VideoReader, video_fps: int) -> list[Image.Image]:
    video_duration_s = video.get_frame_timestamp(-1)[1]
    num_frames = int(video_duration_s * video_fps)

    frames = video.get_batch(range(0, len(video), len(video) // num_frames)).asnumpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def split_prompt_by_videos(prompt: str) -> list[str]:
    prompt_parts = prompt.split(VIDEO_PLACEHOLDER_TOKEN)
    if len(prompt_parts) == 1:
        prompt_parts = [""] + prompt_parts

    return prompt_parts


@lru_cache(maxsize=2048)
def load_video(video_path: str | Path) -> VideoReader:
    video = VideoReader(video_path, ctx=cpu(0))
    return video
