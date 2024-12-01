# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial
from types import MethodType
from typing import Any, Dict, Tuple

import torch
from modelscope import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func
from .qwen import get_model_tokenizer_qwen

logger = get_logger()


def get_model_tokenizer_mplug_owl2(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
    local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
    sys.path.append(os.path.join(local_repo_path))

    # register
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
    from mplug_owl2 import MPLUGOwl2LlamaForCausalLM
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = kwargs.pop('vocab_size', None)
    if vocab_size is not None:
        model_config.vocab_size = vocab_size
    get_model_tokenizer_function = kwargs.pop('get_model_tokenizer_function')
    model, tokenizer = get_model_tokenizer_function(
        model_dir, model_info, model_kwargs, load_model, model_config=model_config, **kwargs)
    logger.info('Please ignore the unimported warning.')
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    tokenizer.processor = processor
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl2, [
            ModelGroup([
                Model('iic/mPLUG-Owl2', 'MAGAer13/mplug-owl2-llama2-7b'),
                Model('iic/mPLUG-Owl2.1', 'Mizukiluke/mplug_owl_2_1'),
            ],
                       requires=['transformers<4.35'],
                       tags=['vision']),
        ],
        TemplateType.mplug_owl2,
        get_model_tokenizer_mplug_owl2,
        model_arch=ModelArch.mplug_owl2))


def get_model_tokenizer_mplug_owl3(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    get_class_from_dynamic_module('configuration_hyper_qwen2.HyperQwen2Config', model_dir)
    model_cls = get_class_from_dynamic_module('modeling_mplugowl3.mPLUGOwl3Model', model_dir)
    model_cls._no_split_modules = ['SiglipEncoderLayer']
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor = model.init_processor(tokenizer)
    if model is not None:
        func_list = ['generate', 'forward']
        use_submodel_func(model, 'language_model', func_list)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl3, [
            ModelGroup([
                Model('iic/mPLUG-Owl3-1B-241014', 'mPLUG/mPLUG-Owl3-1B-241014'),
                Model('iic/mPLUG-Owl3-2B-241014', 'mPLUG/mPLUG-Owl3-2B-241014'),
                Model('iic/mPLUG-Owl3-7B-240728', 'mPLUG/mPLUG-Owl3-7B-240728'),
            ],
                       requires=['transformers>=4.36', 'icecream'],
                       tags=['multi-modal', 'vision', 'video']),
        ],
        TemplateType.mplug_owl3,
        get_model_tokenizer_mplug_owl3,
        architectures=['mPLUGOwl3Model'],
        model_arch=ModelArch.mplug_owl3))