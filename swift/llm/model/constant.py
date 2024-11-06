# Copyright (c) Alibaba, Inc. and its affiliates.
# Classification criteria for model_type: same model architecture, tokenizer (get function), template.
from typing import List


class LLMModelType:
    # dense
    qwen = 'qwen'
    codefuse_qwen = 'codefuse_qwen'
    modelscope_agent = 'modelscope_agent'
    qwen2 = 'qwen2'
    qwen2_5 = 'qwen2_5'

    llama = 'llama'
    llama3 = 'llama3'
    longwriter_llama3 = 'longwriter_llama3'
    llama3_2 = 'llama3_2'
    yi = 'yi'
    yi_coder = 'yi_coder'

    reflection_llama3_1 = 'reflection_llama3_1'

    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    codefuse_codegeex2 = 'codefuse_codegeex2'
    codegeex4 = 'codegeex4'
    glm4 = 'glm4'

    internlm = 'internlm'
    internlm2 = 'internlm2'

    longwriter_llama3_1 = 'longwriter_llama3_1'
    longwriter_glm4 = 'longwriter_glm4'

    telechat = 'telechat'
    telechat2 = 'telechat2'

    yuan2 = 'yuan2'

    atom = 'atom'

    grok = 'grok'

    mamba = 'mamba'

    polylm = 'polylm'

    skywork = 'skywork'

    codefuse_codellama = 'codefuse_codellama'

    baichuan = 'baichuan'
    baichuan2 = 'baichuan2'
    baichuan2_int4 = 'baichuan2_int4'

    phi3_small = 'phi3_small'

    deepseek = 'deepseek'
    deepseek2_5 = 'deepseek2_5'

    # moe
    qwen2_moe = 'qwen2_moe'
    minicpm_moe = 'minicpm_moe'
    deepseek_moe = 'deepseek_moe'


class MLLMModelType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'
    llama3_2_vision = 'llama3_2_vision'

    glm4v = 'glm4v'
    cogvlm = 'cogvlm'
    cogagent_vqa = 'cogagent_vqa'
    cogagent_chat = 'cogagent_chat'
    cogvlm2 = 'cogvlm2'
    cogvlm2_video = 'cogvlm2_video'
    cogvlm_chat = 'cogvlm_chat'

    xcomposer2 = 'xcomposer2'
    xcomposer2_4khd = 'xcomposer2_4khd'
    xcomposer2_5 = 'xcomposer2_5'

    llama3_1_omni = 'llama3_1_omni'
    idefics3_llama3 = 'idefics3_llama3'

    llava1_5 = 'llava1_5'
    llava1_6_mistral = 'llava1_6_mistral'
    llava1_6_vicuna = 'llava1_6_vicuna'
    llava1_6_yi = 'llava1_6_yi'
    llava1_6_llama3_1 = 'llava1_6_llama3_1'
    llava_next = 'llava_next'
    llava_llama = 'llava_llama'
    llama3_llava_next_hf = 'llama3_llava_next_hf'
    llama3_llava_next = 'llama3_llava_next'
    llava_next_video = 'llava_next_video'
    llava_next_video_yi = 'llava_next_video_yi'

    # internvl
    internvl = 'internvl'
    internvl_mini = 'internvl_mini'
    internvl2 = 'internvl2'
    internvl2_phi3 = 'internvl2_phi3'

    deepseek_vl = 'deepseek_vl'

    minicpmv = 'minicpmv'
    minicpmv2_6 = 'minicpmv2_6'
    minicpmv2_5 = 'minicpmv2_5'

    mplug3 = 'mplug3'
    mplug2 = 'mplug2'
    mplug2_1 = 'mplug2_1'

    phi3_vl = 'phi3_vl'

    florence = 'florence'

    emu3_gen = 'emu3_gen'

    idefics3 = 'idefics3'

    yi_vl = 'yi_vl'


class ModelType(LLMModelType, MLLMModelType):

    @classmethod
    def get_model_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__'):
                continue
            value = cls.__dict__[k]
            if isinstance(value, str):
                res.append(value)
        return res