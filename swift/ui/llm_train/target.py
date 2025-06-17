# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Target(BaseUI):

    group = 'llm_train'

    locale_dict = {
        'target_params': {
            'label': {
                'zh': 'target模块参数',
                'en': 'Tuner modules params'
            }
        },
        'freeze_llm': {
            'label': {
                'zh': '冻结llm',
                'en': 'freeze llm'
            },
        },
        'freeze_aligner': {
            'label': {
                'zh': '冻结aligner',
                'en': 'freeze aligner'
            },
        },
        'freeze_vit': {
            'label': {
                'zh': '冻结vit',
                'en': 'freeze vit'
            },
        },
        'target_modules': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
        'target_regex': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
        'modules_to_save': {
            'label': {
                'zh': 'Tuner参数',
                'en': 'Tuner params'
            }
        },
        'init_weights': {
            'label': {
                'zh': 'lora初始化方法',
                'en': 'init lora weights'
            },
            'info': {
                'zh': 'gaussian/pissa/pissa_niter_[n]/olora/loftq/true/false',
                'en': 'gaussian/pissa/pissa_niter_[n]/olora/loftq/true/false',
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='target_params', open=True):
            with gr.Blocks():
                with gr.Row():
                    gr.Textbox(elem_id='target_modules', lines=1, value='all-linear', is_list=True, scale=5)
                    gr.Checkbox(elem_id='freeze_llm', scale=5)
                    gr.Checkbox(elem_id='freeze_aligner', scale=5)
                    gr.Checkbox(elem_id='freeze_vit', scale=5)
                with gr.Row():
                    gr.Textbox(elem_id='target_regex', scale=8)
                    gr.Textbox(elem_id='modules_to_save', scale=8)
                    gr.Textbox(elem_id='init_weights', scale=5)
