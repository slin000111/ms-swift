# Copyright (c) Alibaba, Inc. and its affiliates.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

from transformers import PreTrainedTokenizerBase

from .base import Template
from .utils import Prompt, Word


@dataclass
class TemplateMeta:
    """
    Examples:
        chatml (with bos):
            prefix: <s>
            prompt: <|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n
            chat_sep: <|im_end|>\n
            suffix: <|im_end|>
            system_prefix: <s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n

        <s><|im_start|>system  # prefix or system_prefix
        {{SYSTEM}}<|im_end|>
        <|im_start|>user  # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>  # chat_sep
        <|im_start|>user  # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>  # suffix
    """
    template_type: str
    prefix: Prompt
    prompt: Prompt
    chat_sep: Optional[Prompt]
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    template_cls: Type[Template] = Template
    system_prefix: Optional[Prompt] = None
    default_system: Optional[str] = None
    response_prefix: str = ''

    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=list)
    agent_template: str = 'react_en'

    def to_generate_template_meta(self) -> 'TemplateMeta':
        self = deepcopy(self)
        return TemplateMeta(
            self.template_type,
            prefix=[],
            prompt=['{{QUERY}}'],
            chat_sep=None,
            template_cls=self.template_cls,
            auto_add_bos=True,
            stop_words=self.stop_words,
        )

    @staticmethod
    def _has_system(prefix_or_prompt: Prompt) -> bool:
        return any(['{{SYSTEM}}' in p for p in prefix_or_prompt])

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        return [p.replace('{{SYSTEM}}', '') for p in prefix if isinstance(p, str)]

    def _check_template_meta(self):
        # check
        for x in [self.prefix, self.prompt, self.suffix]:
            assert isinstance(x, list)
        for x in [self.chat_sep, self.system_prefix]:
            assert x is None or isinstance(x, list)

    def __post_init__(self):
        # system
        if self._has_system(self.prefix):
            assert self.system_prefix is None, 'The prefix already contains {{SYSTEM}}.'
            self.system_prefix = self.prefix
            self.prefix = self._replace_system(self.prefix)

        self.is_post_system = self._has_system(self.prompt)  # mistral_nemo
        if self.is_post_system:
            self.system_prompt = self.prompt
            self.prompt = [context for context in self.prompt if '{{SYSTEM}}' not in context]

        if self.system_prefix is None and not self.is_post_system:
            self.support_system = False
        else:
            self.support_system = True
        self.check_system(self.default_system)

        self.support_multi_round = self.chat_sep is not None

    @staticmethod
    def _token_attr_to_id(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """Turn `eos_token_id` to token id

        e.g. [['eos_token_id']] -> [[2]]
        """
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                v = [getattr(tokenizer, sub_v) if isinstance(sub_v, str) else sub_v for sub_v in v]
            res_value.append(v)
        return res_value

    def init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            value = getattr(self, key)
            value = self._token_attr_to_id(tokenizer, value)
            setattr(self, key, value)

        if self.suffix and self.suffix[-1] not in self.stop_words:
            self.stop_words.append(self.suffix[-1])
        if tokenizer.eos_token not in self.stop_words:
            self.stop_words.append(tokenizer.eos_token)

        self.stop_token_id = tokenizer.eos_token_id
        if self.suffix:
            suffix_tokens = self.suffix[-1]
            if isinstance(suffix_tokens, str):
                stop_token_id = tokenizer.convert_tokens_to_ids(suffix_tokens)
            elif isinstance(suffix_tokens, list) and len(suffix_tokens) == 1:
                stop_token_id = suffix_tokens[0]
            else:
                stop_token_id = None
            if stop_token_id is not None:
                self.stop_token_id = stop_token_id

    def check_system(self, system: Optional[str]) -> None:
        if system is not None:
            assert self.support_system, (
                f'The template does not support `system`, template_type: {self.template_type}, system: {system}')
