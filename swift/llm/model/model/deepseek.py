# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from typing import Any, Dict

from modelscope import AutoConfig
from transformers import PretrainedConfig

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import git_clone_github, use_submodel_func


def get_model_tokenizer_deepseek_moe(model_dir: str,
                                     config: PretrainedConfig,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, config, model_kwargs, load_model, **kwargs)
    if model is not None:
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__

        def _dtype_hook(module, input, output):
            return output.to(input[0].dtype)

        for module in model.modules():
            if isinstance(module, mlp_cls):
                module.register_forward_hook(_dtype_hook)
    return model, tokenizer


def get_model_tokenizer_deepseek2(model_dir: str,
                                  config: PretrainedConfig,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_deepseek_moe(model_dir, config, model_kwargs, load_model, **kwargs)
    if model is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.deepseek_moe,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-moe-16b-chat', 'deepseek-ai/deepseek-moe-16b-chat'),
                    Model('deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-base'),
                ],
                tags=['moe'],
            ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_deepseek_moe,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/DeepSeek-Coder-V2-Base', 'deepseek-ai/DeepSeek-Coder-V2-Base'),
                    Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base'),
                    Model('deepseek-ai/DeepSeek-Coder-V2-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Instruct'),
                    Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'),
                    Model('deepseek-ai/DeepSeek-V2-Lite', 'deepseek-ai/DeepSeek-V2-Lite'),
                    Model('deepseek-ai/DeepSeek-V2-Lite-Chat', 'deepseek-ai/DeepSeek-V2-Lite-Chat'),
                    Model('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2'),
                    Model('deepseek-ai/DeepSeek-V2-Chat', 'deepseek-ai/DeepSeek-V2-Chat'),
                ],
                requires=['transformers>=4.39.3'],
                tags=['moe'],
            ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_deepseek_moe,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek2_5,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/DeepSeek-V2.5', 'deepseek-ai/DeepSeek-V2.5'),
                ],
                requires=['transformers>=4.39.3'],
                tags=['moe'],
            ),
        ],
        TemplateType.deepseek2_5,
        get_model_tokenizer_deepseek_moe,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_vllm=True,
    ))


def get_model_tokenizer_deepseek_vl(model_dir: str,
                                    config: PretrainedConfig,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    # compat with python==3.10
    if sys.version_info.minor >= 10:
        import collections
        import collections.abc
        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
    sys.path.append(os.path.join(local_repo_path))
    from deepseek_vl.models import VLChatProcessor
    processor = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    # flash_attn
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionImpl(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    attn_type.update_config(model_config)
    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, config, model_kwargs, load_model, model_config=model_config, tokenizer=tokenizer, **kwargs)
    tokenizer.processor = processor
    if load_model:
        patch_output_clone(model.language_model.model.embed_tokens)
        patch_output_to_input_device(model.language_model.model.embed_tokens)
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        use_submodel_func(model, 'language_model', func_list)
        model.generation_config = model.language_model.generation_config
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl,
        [
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-vl-7b-chat', 'deepseek-ai/deepseek-vl-7b-chat'),
                    Model('deepseek-ai/deepseek-vl-1.3b-chat', 'deepseek-ai/deepseek-vl-1.3b-chat'),
                ],
                tags=['multi-modal', 'vision'],
            ),
        ],
        TemplateType.deepseek_vl,
        get_model_tokenizer_deepseek_vl,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        support_flash_attn=True,
        support_lmdeploy=True,
    ))