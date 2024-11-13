# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Type

import torch
import transformers
from modelscope import AutoConfig
from packaging import version
from modelscope import AutoTokenizer
from transformers import PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from ..model_arch import ModelArch
from swift.llm import TemplateType
from swift.utils import get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_output_to_input_device
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import AttnImpl, ModelInfo

logger = get_logger()


def remove_property(tokenizer_cls: Type[PreTrainedTokenizerBase], tokenizer_config: Dict[str, Any]) -> None:
    for k, v in tokenizer_cls.__dict__.items():
        if k.endswith('_token') and isinstance(v, property) and k in tokenizer_config:
            setattr(tokenizer_cls, k, tokenizer_config[k])


def get_model_tokenizer_chatglm(model_dir: str,
                                model_info: ModelInfo,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if model_kwargs.get('quantization_config') is not None:
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['output_layer']
    # fix transformers>=4.34 bug
    if version.parse(transformers.__version__) >= version.parse('4.34'):
        tokenizer_config = get_tokenizer_config(model_dir)
        class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
        tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
        tokenizer_cls._auto_class = 'AutoTokenizer'
        remove_property(tokenizer_cls, tokenizer_config)
        kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward

    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.chatglm2, [
            ModelGroup([
                Model('ZhipuAI/chatglm2-6b', 'THUDM/chatglm2-6b'),
                Model('ZhipuAI/chatglm2-6b-32k', 'THUDM/chatglm2-6b-32k')
            ]),
            ModelGroup(
                [Model('ZhipuAI/codegeex2-6b', 'THUDM/codegeex2-6b')],
                requires=['transformers<4.42'],
                tags=['coding'],
            ),
        ],
        TemplateType.chatglm2,
        get_model_tokenizer_chatglm,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.chatglm))

register_model(
    ModelMeta(
        LLMModelType.codefuse_codegeex2, [
            ModelGroup(
                [Model('codefuse-ai/CodeFuse-CodeGeeX2-6B', 'codefuse-ai/CodeFuse-CodeGeeX2-6B')],
                tags=['coding'],
            ),
        ],
        TemplateType.codefuse,
        get_model_tokenizer_chatglm,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers==4.33.0'],
        model_arch=ModelArch.chatglm))

register_model(
    ModelMeta(
        LLMModelType.chatglm3, [
            ModelGroup([
                Model('ZhipuAI/chatglm3-6b-base', 'THUDM/chatglm3-6b-base'),
                Model('ZhipuAI/chatglm3-6b', 'THUDM/chatglm3-6b'),
                Model('ZhipuAI/chatglm3-6b-32k', 'THUDM/chatglm3-6b-32k'),
                Model('ZhipuAI/chatglm3-6b-128k', 'THUDM/chatglm3-6b-128k'),
            ])
        ],
        TemplateType.chatglm3,
        get_model_tokenizer_chatglm,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.chatglm))


def get_model_tokenizer_glm4(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'))
    kwargs['model_config'] = model_config
    model, tokenizer = get_model_tokenizer_chatglm(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if len(tokenizer.encode('<|user|>', add_special_tokens=False)) > 1:
        for k in tokenizer.special_tokens.keys():
            tokenizer.add_tokens(k)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.glm4, [
            ModelGroup([
                Model('ZhipuAI/glm-4-9b', 'THUDM/glm-4-9b'),
                Model('ZhipuAI/glm-4-9b-chat', 'THUDM/glm-4-9b-chat'),
                Model('ZhipuAI/glm-4-9b-chat-1m', 'THUDM/glm-4-9b-chat-1m'),
            ]),
            ModelGroup([
                Model('ZhipuAI/LongWriter-glm4-9b', 'THUDM/LongWriter-glm4-9b'),
            ])
        ],
        TemplateType.chatglm4,
        get_model_tokenizer_glm4,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers==4.43.0'],
        model_arch=ModelArch.chatglm))

register_model(
    ModelMeta(
        LLMModelType.codegeex4,
        [ModelGroup([
            Model('ZhipuAI/codegeex4-all-9b', 'THUDM/codegeex4-all-9b'),
        ], tags=['coding'])],
        TemplateType.codegeex4,
        get_model_tokenizer_glm4,
        requires=['transformers<4.42'],
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm))


def get_model_tokenizer_glm4v(model_dir: str,
                              model_info: ModelInfo,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model, tokenizer = get_model_tokenizer_glm4(model_dir, model_info, model_kwargs, load_model, **kwargs)
    # fix merge-lora
    tokenizer.init_kwargs['image_size'] = 1120
    if load_model:
        # fix device_map 4
        n_gpu = torch.cuda.device_count()
        local_world_size = get_dist_setting()[3]
        if n_gpu // local_world_size >= 4:
            for layer in model.transformer.vision.transformer.layers:
                patch_output_to_input_device(layer.mlp)
                patch_output_to_input_device(layer.post_attention_layernorm)
            device = next(model.transformer.vision.linear_proj.parameters()).device
            model.transformer.vision.boi.data = model.transformer.vision.boi.to(device)
            model.transformer.vision.eoi.data = model.transformer.vision.eoi.to(device)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.glm4v, [ModelGroup([
            Model('ZhipuAI/glm-4v-9b', 'THUDM/glm-4v-9b'),
        ])],
        TemplateType.glm4v,
        get_model_tokenizer_glm4v,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        requires=['transformers==4.43.0'],
        model_arch=ModelArch.glm4v))


def get_model_tokenizer_cogvlm(model_dir: str,
                               model_info: ModelInfo,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/vicuna-7b-v1.5', revision='master', trust_remote_code=True)
    if load_model:
        logger.warning('CogAgent with FusedLayerNorm will cause an training loss of NAN, '
                       'to avoid this, please uninstall apex.')
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    logger.info('Please ignore the unimported warning.')
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.cogvlm, [
            ModelGroup([
                Model('ZhipuAI/cogvlm-chat', 'THUDM/cogvlm-chat-hf'),
            ]),
        ],
        TemplateType.cogvlm,
        get_model_tokenizer_cogvlm,
        architectures=['CogVLMForCausalLM'],
        support_gradient_checkpointing=False,
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))

register_model(
    ModelMeta(
        MLLMModelType.cogagent_chat, [
            ModelGroup([
                Model('ZhipuAI/cogagent-chat', 'THUDM/cogagent-chat-hf'),
            ]),
        ],
        TemplateType.cogagent_chat,
        get_model_tokenizer_cogvlm,
        architectures=['CogAgentForCausalLM'],
        support_gradient_checkpointing=False,
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))

register_model(
    ModelMeta(
        MLLMModelType.cogagent_vqa, [ModelGroup([
            Model('ZhipuAI/cogagent-vqa', 'THUDM/cogagent-vqa-hf'),
        ])],
        TemplateType.cogagent_vqa,
        get_model_tokenizer_cogvlm,
        architectures=['CogAgentForCausalLM'],
        support_gradient_checkpointing=False,
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm))


def get_model_tokenizer_cogvlm2(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        # fix device map 4
        for layer in model.model.vision.transformer.layers:
            patch_output_to_input_device(layer.mlp)
            patch_output_to_input_device(layer.post_attention_layernorm)

        device = next(model.model.vision.linear_proj.parameters()).device
        model.model.vision.boi.data = model.model.vision.boi.to(device)
        model.model.vision.eoi.data = model.model.vision.eoi.to(device)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.cogvlm2, [
            ModelGroup([
                Model('ZhipuAI/cogvlm2-llama3-chat-19B', 'THUDM/cogvlm2-llama3-chat-19B'),
                Model('ZhipuAI/cogvlm2-llama3-chinese-chat-19B', 'THUDM/cogvlm2-llama3-chinese-chat-19B'),
            ]),
        ],
        TemplateType.cogvlm,
        get_model_tokenizer_cogvlm2,
        architectures=['CogVLMForCausalLM'],
        requires=['transformers<4.42'],
        model_arch=ModelArch.cogvlm,
        support_gradient_checkpointing=False))

register_model(
    ModelMeta(
        MLLMModelType.cogvlm2_video, [
            ModelGroup([
                Model('ZhipuAI/cogvlm2-video-llama3-chat', 'THUDM/cogvlm2-video-llama3-chat'),
            ],
                       tags=['video']),
        ],
        TemplateType.cogvlm2_video,
        get_model_tokenizer_cogvlm2,
        architectures=['CogVLMVideoForCausalLM'],
        requires=['transformers>=4.42'],
        support_gradient_checkpointing=False,
        model_arch=ModelArch.cogvlm))