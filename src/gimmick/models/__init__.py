from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gimmick.models.baseline_model import BaselineModel


def get_all_supported_baseline_models(
    return_list: bool = True,
    return_lower: bool = True,
    return_api: bool = True,
    return_vlms: bool = True,
    return_llms: bool = True,
) -> list[str] | dict[str, str]:
    # supported VLMS but not used:
    # "neulab/pangea-7b-hf": "neulab/Pangea-7B-hf",
    # "mistralai/pixtral-12b-2409": "mistralai/Pixtral-12B-2409",
    # "google/paligemma2-3b-pt-896": "google/paligemma2-3b-pt-896",
    # "google/paligemma2-3b-pt-448": "google/paligemma2-3b-pt-448",
    # "google/paligemma2-3b-pt-224": "google/paligemma2-3b-pt-224",
    # "google/paligemma2-10b-pt-896": "google/paligemma2-10b-pt-896",
    # "google/paligemma2-10b-pt-448": "google/paligemma2-10b-pt-448",
    # "google/paligemma2-10b-pt-224": "google/paligemma2-10b-pt-224",
    # "google/paligemma2-28b-pt-896": "google/paligemma2-28b-pt-896",
    # "google/paligemma2-28b-pt-448": "google/paligemma2-28b-pt-448",
    # "google/paligemma2-28b-pt-224": "google/paligemma2-28b-pt-224",
    # "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
    # "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf": "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf",
    # "opengvlab/internvl2_5-4b-mpo": "OpenGVLab/InternVL2_5-4B-MPO",  # qwen/qwen2.5-3b-instruct
    # "opengvlab/internvl2_5-8b-mpo": "OpenGVLab/InternVL2_5-8B-MPO",  # internlm/internlm2_5-7b-chat
    # "opengvlab/internvl2_5-26b-mpo": "OpenGVLab/InternVL2_5-26B-MPO",  # internlm/internlm2_5-20b-chat
    # "opengvlab/internvl2_5-38b-mpo": "OpenGVLab/InternVL2_5-38B-MPO",  # qwen/qwen2.5-32b-instruct
    # "opengvlab/internvl2_5-78b-mpo": "OpenGVLab/InternVL2_5-78B-MPO",  # qwen/qwen2.5-72b-instruct

    # we only store the case-sensitive models bc HF downloads are case-sensitive
    vlms = {
        "qwen/qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen/qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen/qwen2-vl-72b-instruct": "Qwen/Qwen2-VL-72B-Instruct",
        "microsoft/phi-3.5-vision-instruct": "microsoft/Phi-3.5-vision-instruct",
        "openbmb/minicpm-v-2_6": "openbmb/MiniCPM-V-2_6",
        "meta-llama/llama-3.2-11b-vision-instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "opengvlab/internvl2_5-1b": "OpenGVLab/InternVL2_5-1B",  # qwen/qwen2.5-0.5b-instruct
        "opengvlab/internvl2_5-2b": "OpenGVLab/InternVL2_5-2B",  # internlm/internlm2_5-1_8b-chat
        "opengvlab/internvl2_5-4b": "OpenGVLab/InternVL2_5-4B",  # qwen/qwen2.5-3b-instruct
        "opengvlab/internvl2_5-8b": "OpenGVLab/InternVL2_5-8B",  # internlm/internlm2_5-7b-chat
        "opengvlab/internvl2_5-26b": "OpenGVLab/InternVL2_5-26B",  # internlm/internlm2_5-20b-chat
        "opengvlab/internvl2_5-38b": "OpenGVLab/InternVL2_5-38B",  # qwen/qwen2.5-32b-instruct
        "opengvlab/internvl2_5-78b": "OpenGVLab/InternVL2_5-78B",  # qwen/qwen2.5-72b-instruct
        "wuenlp/centurio_aya": "WueNLP/centurio_aya",  # cohereforai/aya-expanse-8b
        "wuenlp/centurio_qwen": "WueNLP/centurio_qwen",  # qwen/qwen2.5-7b-instruct
    }
    api_vlms = {
        "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
        # "gpt-4o-2024-08-06": "gpt-4o-2024-08-06",  # supported but not used
        # "gemini-1.5-flash-001": "gemini-1.5-flash-001",  # supported but not used
        "gemini-1.5-flash-002": "gemini-1.5-flash-002",
        # "gemini-1.5-pro-001": "gemini-1.5-pro-001",  # supported but not used
        "gemini-1.5-pro-002": "gemini-1.5-pro-002",
        "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    }
    api_llms = {
        # "o1-2024-12-17": "o1-2024-12-17",  # supported but not used
        # "o1-mini-2024-09-12": "o1-mini-2024-09-12",  # supported but not used
        # "o3-mini-2025-01-31": "o3-mini-2025-01-31",  # supported but not used
    }
    llms = {
        "qwen/qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen/qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen/qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        "qwen/qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen/qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
        "qwen/qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        "internlm/internlm2_5-1_8b-chat": "internlm/internlm2_5-1_8b-chat",
        "internlm/internlm2_5-7b-chat": "internlm/internlm2_5-7b-chat",
        "internlm/internlm2_5-20b-chat": "internlm/internlm2_5-20b-chat",
        "cohereforai/aya-expanse-8b": "CohereForAI/aya-expanse-8b",
        "microsoft/phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    }
    supported = {}
    if return_vlms:
        supported = {**supported, **vlms}
    if return_api:
        supported = {**supported, **api_vlms}
    if return_llms:
        supported = {**supported, **llms, **api_llms}

    if return_list:
        if return_lower:
            return list(supported.keys())
        return list(supported.values())
    return supported


def get_all_perplexity_supported_models() -> dict[str, str]:
    return {
        "qwen/qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen/qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen/qwen2-vl-72b-instruct": "Qwen/Qwen2-VL-72B-Instruct",
        # "microsoft/phi-3.5-vision-instruct": "microsoft/Phi-3.5-vision-instruct",  # some weird cuda error
        # "meta-llama/llama-3.2-11b-vision-instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",  # some weird cuda error
        "wuenlp/centurio_aya": "WueNLP/centurio_aya",
        "wuenlp/centurio_qwen": "WueNLP/centurio_qwen",
        "opengvlab/internvl2_5-4b": "OpenGVLab/InternVL2_5-4B",
        "opengvlab/internvl2_5-8b": "OpenGVLab/InternVL2_5-8B",
        "opengvlab/internvl2_5-26b": "OpenGVLab/InternVL2_5-26B",
        "opengvlab/internvl2_5-38b": "OpenGVLab/InternVL2_5-38B",
        "opengvlab/internvl2_5-78b": "OpenGVLab/InternVL2_5-78B",
    }


def get_model_paper_names() -> dict[str, str]:
    paper_names = {
        # api models
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "gpt-4o-2024-11-20": "GPT-4o",
        "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
        "gemini-1.5-pro-002": "Gemini Pro",
        "gemini-1.5-flash-002": "Gemini Flash",
        # llms
        "qwen/qwen2.5-0.5b-instruct": "Qwen2.5 0.5B",
        "qwen/qwen2.5-1.5b-instruct": "Qwen2.5 1.5B",
        "qwen/qwen2.5-3b-instruct": "Qwen2.5 3B",
        "qwen/qwen2.5-7b-instruct": "Qwen2.5 7B",
        "qwen/qwen2.5-32b-instruct": "Qwen2.5 32B",
        "qwen/qwen2.5-72b-instruct": "Qwen2.5 72B",
        "internlm/internlm2_5-1_8b-chat": "InternLM2.5 1.8B",
        "internlm/internlm2_5-7b-chat": "InternLM2.5 7B",
        "internlm/internlm2_5-20b-chat": "InternLM2.5 20B",
        "cohereforai/aya-expanse-8b": "Aya Expanse 8B",
        "microsoft/phi-3.5-mini-instruct": "Phi 3.5 Mini",
        # vlms
        "qwen/qwen2-vl-2b-instruct": "Qwen2 VL 2B",
        "qwen/qwen2-vl-7b-instruct": "Qwen2 VL 7B",
        "qwen/qwen2-vl-72b-instruct": "Qwen2 VL 72B",
        "opengvlab/internvl2_5-1b": "InternVL2.5 1B",
        "opengvlab/internvl2_5-2b": "InternVL2.5 2B",
        "opengvlab/internvl2_5-4b": "InternVL2.5 4B",
        "opengvlab/internvl2_5-8b": "InternVL2.5 8B",
        "opengvlab/internvl2_5-26b": "InternVL2.5 26B",
        "opengvlab/internvl2_5-38b": "InternVL2.5 38B",
        "opengvlab/internvl2_5-78b": "InternVL2.5 78B",
        "wuenlp/centurio_qwen": "Centurio Qwen",
        "wuenlp/centurio_aya": "Centurio Aya",
        "openbmb/minicpm-v-2_6": "MiniCPM V 2.6",
        "microsoft/phi-3.5-vision-instruct": "Phi 3.5 Vision",
        "meta-llama/llama-3.2-11b-vision-instruct": "Llama 3.2 11B Vision",
    }

    return paper_names


def get_llm_backbones() -> dict[str, str]:
    return {
        "qwen/qwen2-vl-2b-instruct": "qwen/qwen2.5-1.5b-instruct",
        "qwen/qwen2-vl-7b-instruct": "qwen/qwen2.5-7b-instruct",
        "qwen/qwen2-vl-72b-instruct": "qwen/qwen2.5-72b-instruct",
        "opengvlab/internvl2_5-1b": "qwen/qwen2.5-0.5b-instruct",
        "opengvlab/internvl2_5-2b": "internlm/internlm2_5-1_8b-chat",
        "opengvlab/internvl2_5-4b": "qwen/qwen2.5-3b-instruct",
        "opengvlab/internvl2_5-8b": "internlm/internlm2_5-7b-chat",
        "opengvlab/internvl2_5-26b": "internlm/internlm2_5-20b-chat",
        "opengvlab/internvl2_5-38b": "qwen/qwen2.5-32b-instruct",
        "opengvlab/internvl2_5-78b": "qwen/qwen2.5-72b-instruct",
        "wuenlp/centurio_aya": "cohereforai/aya-expanse-8b",
        "wuenlp/centurio_qwen": "qwen/qwen2.5-7b-instruct",
        "microsoft/phi-3.5-vision-instruct": "microsoft/phi-3.5-mini-instruct",
    }


def get_model_sizes() -> dict[str, float]:
    model_sizes = {
        # api models
        "claude-3-5-sonnet-20241022": 1e12,
        "gpt-4o-2024-11-20": 1 * 1e12,
        "gpt-4o-mini-2024-07-18": 5 * 1e11,
        "gemini-1.5-pro-002": 1 * 1e12,
        "gemini-1.5-flash-002": 5 * 1e11,
        # vlms
        "qwen/qwen2-vl-2b-instruct": 2208985600,
        "qwen/qwen2-vl-7b-instruct": 8291375616,
        "qwen/qwen2-vl-72b-instruct": 73405560320,
        "microsoft/phi-3.5-vision-instruct": 4146621440,
        "openbmb/minicpm-v-2_6": 8099175152,
        "meta-llama/llama-3.2-11b-vision-instruct": 10670220835,
        "opengvlab/internvl2_5-1b": 938193024,
        "opengvlab/internvl2_5-2b": 2205754368,
        "opengvlab/internvl2_5-4b": 3712637952,
        "opengvlab/internvl2_5-8b": 8075365376,
        "opengvlab/internvl2_5-26b": 25514186112,
        "opengvlab/internvl2_5-38b": 38388164992,
        "opengvlab/internvl2_5-78b": 78408318336,
        "wuenlp/centurio_aya": 8054255616,
        "wuenlp/centurio_qwen": 7636726272,
        # llms
        "qwen/qwen2.5-0.5b-instruct": 494032768,
        "qwen/qwen2.5-1.5b-instruct": 1543714304,
        "qwen/qwen2.5-3b-instruct": 3085938688,
        "qwen/qwen2.5-7b-instruct": 7615616512,
        "qwen/qwen2.5-32b-instruct": 32763876352,
        "qwen/qwen2.5-72b-instruct": 72706203648,
        "internlm/internlm2_5-1_8b-chat": 1889110016,
        "internlm/internlm2_5-7b-chat": 7737708544,
        "internlm/internlm2_5-20b-chat": 19861149696,
        "cohereforai/aya-expanse-8b": 8028033024,
        "microsoft/phi-3.5-mini-instruct": 3821079552,
    }

    return model_sizes


def get_model_families() -> dict[str, str]:
    model_family = {
        # api models
        "claude-3-5-sonnet-20241022": "claude",
        "gpt-4o-2024-11-20": "gpt-4o",
        "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
        "gemini-1.5-pro-002": "gemini-pro",
        "gemini-1.5-flash-002": "gemini-flash",
        # vlms
        "qwen/qwen2-vl-2b-instruct": "qwen2-vl",
        "qwen/qwen2-vl-7b-instruct": "qwen2-vl",
        "qwen/qwen2-vl-72b-instruct": "qwen2-vl",
        "microsoft/phi-3.5-vision-instruct": "phi-3.5-vision",
        "openbmb/minicpm-v-2_6": "minicpm-v",
        "meta-llama/llama-3.2-11b-vision-instruct": "llama-3.2-vision",
        "opengvlab/internvl2_5-1b": "internvl2.5",
        "opengvlab/internvl2_5-2b": "internvl2.5",
        "opengvlab/internvl2_5-4b": "internvl2.5",
        "opengvlab/internvl2_5-8b": "internvl2.5",
        "opengvlab/internvl2_5-26b": "internvl2.5",
        "opengvlab/internvl2_5-38b": "internvl2.5",
        "opengvlab/internvl2_5-78b": "internvl2.5",
        "wuenlp/centurio_aya": "centurio",
        "wuenlp/centurio_qwen": "centurio",
        # llms
        "qwen/qwen2.5-0.5b-instruct": "qwen2.5",
        "qwen/qwen2.5-1.5b-instruct": "qwen2.5",
        "qwen/qwen2.5-3b-instruct": "qwen2.5",
        "qwen/qwen2.5-7b-instruct": "qwen2.5",
        "qwen/qwen2.5-32b-instruct": "qwen2.5",
        "qwen/qwen2.5-72b-instruct": "qwen2.5",
        "internlm/internlm2_5-1_8b-chat": "internlm2.5",
        "internlm/internlm2_5-7b-chat": "internlm2.5",
        "internlm/internlm2_5-20b-chat": "internlm2.5",
        "cohereforai/aya-expanse-8b": "aya",
        "microsoft/phi-3.5-mini-instruct": "phi-3.5-mini",
    }
    return model_family


def get_model_size_groups(
    return_vlms: bool = True,
    return_llms: bool = True,
) -> dict[str, list[str]]:
    vlms = {
        "S": [  # up to 4B
            "qwen/qwen2-vl-2b-instruct",
            "opengvlab/internvl2_5-1b",
            "opengvlab/internvl2_5-2b",
            "opengvlab/internvl2_5-4b",
            "microsoft/phi-3.5-vision-instruct",
        ],
        "M": [  # up to 12B
            "qwen/qwen2-vl-7b-instruct",
            "opengvlab/internvl2_5-8b",
            "wuenlp/centurio_aya",
            "wuenlp/centurio_qwen",
            "openbmb/minicpm-v-2_6",
            "meta-llama/llama-3.2-11b-vision-instruct",
        ],
        "L": [  # up to 40B
            "opengvlab/internvl2_5-26b",
            "opengvlab/internvl2_5-38b",
        ],
        "XL": [  # up to 80B
            "qwen/qwen2-vl-72b-instruct",
            "opengvlab/internvl2_5-78b",
        ],
    }

    llms = {
        "S": [  # up to 4B
            "qwen/qwen2.5-0.5b-instruct",
            "qwen/qwen2.5-1.5b-instruct",
            "qwen/qwen2.5-3b-instruct",
            "internlm/internlm2_5-1_8b-chat",
            "microsoft/phi-3.5-mini-instruct",
        ],
        "M": [  # up to 12B
            "qwen/qwen2.5-7b-instruct",
            "internlm/internlm2_5-7b-chat",
            "cohereforai/aya-expanse-8b",
        ],
        "L": [  # up to 40B
            "internlm/internlm2_5-20b-chat",
            "qwen/qwen2.5-32b-instruct",
        ],
        "XL": [  # up to 80B
            "qwen/qwen2.5-72b-instruct",
        ],
    }

    ret = {}
    if return_vlms:
        ret.update(vlms)
    if return_llms:
        if len(ret) > 0:
            for size, models in llms.items():
                ret[size].extend(models)
        else:
            ret.update(llms)

    return ret


def get_model_id_from_name_or_id(model_name_or_id: str) -> str:
    model_name_or_id = model_name_or_id.lower()
    if "__" in model_name_or_id:
        model_name_or_id = model_name_or_id.replace("__", "/")
    model_paper_name_2_id = {v.lower(): k for k, v in get_model_paper_names().items()}
    if model_name_or_id in model_paper_name_2_id:
        return model_paper_name_2_id[model_name_or_id]

    if model_name_or_id in get_all_supported_baseline_models(return_list=False):
        return model_name_or_id

    raise KeyError(f"Model Name or ID {model_name_or_id} not supported")


def get_llm_backbone(model_id: str) -> str | None:
    model_id = get_model_id_from_name_or_id(model_id)
    backbones = get_llm_backbones()
    return backbones.get(model_id, None)


def get_model_size_group(model_id: str) -> str:
    model_id = get_model_id_from_name_or_id(model_id)

    api_models = get_all_supported_baseline_models(
        return_api=True, return_llms=False, return_vlms=False
    )
    if model_id in api_models:
        return "A"
    size_groups = get_model_size_groups()
    for group, models in size_groups.items():
        if model_id in models:
            return group
    raise KeyError(f"Model ID {model_id} not supported")


def get_model_family(model_id: str) -> str:
    model_id = get_model_id_from_name_or_id(model_id)
    model_families = get_model_families()

    if model_id not in model_families:
        raise KeyError(f"Model ID {model_id} not supported")

    return model_families[model_id]


def get_model_size(model_id: str) -> float:
    # compute via
    # from huggingface_hub import HfApi
    # from tqdm.auto import tqdm
    # api = HfApi()
    # model_sizes = {}
    # for repo_id in tqdm(get_all_supported_baseline_models(return_api=False)):
    #     metadata = api.get_safetensors_metadata(repo_id)
    #     try:
    #         pc = metadata.parameter_count
    #         parameter_count = pc.get("BF16", pc.get("F16", -1))
    #         params[repo_id.split("/")[-1].lower] = parameter_count
    #     except Exception as e:
    #         print(repo_id)
    #         print(e)
    model_id = get_model_id_from_name_or_id(model_id)
    model_sizes = get_model_sizes()

    if model_id not in model_sizes:
        raise KeyError(f"Model ID {model_id} not supported")

    return model_sizes[model_id]


def get_model_paper_name(model_id: str) -> str:
    model_id = get_model_id_from_name_or_id(model_id)
    paper_names = get_model_paper_names()
    if model_id not in paper_names:
        raise KeyError(
            f"Model ID {model_id} not supported. Supported models: {list(paper_names.keys())}"
        )

    return paper_names[model_id]


def is_model_supported(model_id: str, raise_error: bool = True) -> bool:
    model_id = get_model_id_from_name_or_id(model_id)
    is_supported = model_id in get_all_supported_baseline_models(return_list=False)
    if raise_error and not is_supported:
        raise KeyError(f"Model ID {model_id} not supported")

    return is_supported


def load_model(model_id: str) -> "BaselineModel":
    model_id = get_model_id_from_name_or_id(model_id)
    is_model_supported(model_id, raise_error=True)
    supported = get_all_supported_baseline_models(return_list=False)
    model_id = supported[model_id]  # type: ignore

    if model_id.startswith("Qwen/Qwen2-VL-"):
        from gimmick.models.qwen2_vl import Qwen2VL

        return Qwen2VL(model_id=model_id)
    elif model_id.startswith("Qwen/Qwen2.5-"):
        from gimmick.models.qwen2_5 import Qwen2_5

        return Qwen2_5(model_id=model_id)
    elif model_id == "microsoft/Phi-3.5-vision-instruct":
        from gimmick.models.phi3_5_vision import Phi3_5_Vision

        return Phi3_5_Vision()
    elif model_id == "microsoft/Phi-3.5-mini-instruct":
        from gimmick.models.phi3_5_mini import Phi3_5_Mini

        return Phi3_5_Mini()
    elif model_id == "neulab/Pangea-7B-hf":
        from gimmick.models.pangea import Pangea

        return Pangea()
    elif model_id == "openbmb/MiniCPM-V-2_6":
        from gimmick.models.minicpm_v_2_6 import MiniCPM_V_2_6

        return MiniCPM_V_2_6()
    elif model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        from gimmick.models.llama3_2_vision import Llama3_2_Vision

        return Llama3_2_Vision()
    elif model_id == "mistralai/Pixtral-12B-2409":
        from gimmick.models.pixtral import Pixtral

        return Pixtral()
    elif model_id.startswith("OpenGVLab/InternVL2_5-"):
        from gimmick.models.internvl_2_5 import Intern_V_2_5

        return Intern_V_2_5(model_id=model_id)
    elif model_id.startswith("WueNLP/centurio_"):
        from gimmick.models.centurio import Centurio

        return Centurio(model_id=model_id)
    elif model_id.startswith("internlm/internlm2_5-"):
        from gimmick.models.internlm_2_5 import Intern_LM_2_5

        return Intern_LM_2_5(model_id=model_id)
    elif model_id.startswith("google/paligemma2-"):
        from gimmick.models.paligemma2 import PaliGemma2

        return PaliGemma2(model_id=model_id)
    elif model_id == "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf":
        from gimmick.models.llava_ov import LlavaOneVision

        return LlavaOneVision()
    elif model_id == "CohereForAI/aya-expanse-8b":
        from gimmick.models.aya_expanse import AyaExpanse

        return AyaExpanse()
    elif model_id.startswith("gpt-4o-") or model_id.startswith("o1-"):
        from gimmick.models.openai_model import OAIModel

        return OAIModel(model_id=model_id)
    elif model_id.startswith("gemini-1.5-"):
        from gimmick.models.gemini import Gemini

        return Gemini(model_id=model_id)
    elif model_id == "claude-3-5-sonnet-20241022":
        from gimmick.models.claude import ClaudeSonnetModel

        return ClaudeSonnetModel(model_id=model_id)
    else:
        raise NotImplementedError(f"Model ID {model_id} not supported")
