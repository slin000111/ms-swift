# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Dict, Type

import gradio as gr

from swift.llm import RLHFArguments
from swift.llm.argument.base_args.base_args import get_supported_tuners
from swift.ui.base import BaseUI
from swift.ui.llm_grpo.advanced import GRPOAdvanced
from swift.ui.llm_grpo.dataset import GRPODataset
from swift.ui.llm_grpo.grpo_advanced import GrpoAdvanced
from swift.ui.llm_grpo.hyper import GRPOHyper
from swift.ui.llm_grpo.model import GRPOModel
from swift.ui.llm_grpo.optimizer import GRPOOptimizer
from swift.ui.llm_grpo.quantization import GRPOQuantization
from swift.ui.llm_grpo.ref_model import RefModel
from swift.ui.llm_grpo.report_to import GRPOReportTo
from swift.ui.llm_grpo.reward import Reward
from swift.ui.llm_grpo.rollout import Rollout
from swift.ui.llm_grpo.runtime import GRPORuntime
from swift.ui.llm_grpo.save import GRPOSave
from swift.ui.llm_grpo.tuner import GRPOTuner
from swift.ui.llm_train.llm_train import LLMTrain
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMGRPO(LLMTrain):
    group = 'llm_grpo'

    sub_ui = [
        GRPOModel,
        GRPODataset,
        Reward,
        GRPORuntime,
        Rollout,
        GRPOSave,
        GRPOTuner,
        GRPOOptimizer,
        GRPOHyper,
        GRPOQuantization,
        GRPOAdvanced,
        RefModel,
        GrpoAdvanced,
        GRPOReportTo,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_grpo': {
            'label': {
                'zh': 'LLM GRPO',
                'en': 'LLM GRPO',
            }
        },
        'submit_alert': {
            'value': {
                'zh':
                '任务已开始，请查看tensorboard或日志记录，关闭本页面不影响训练过程',
                'en':
                'Task started, please check the tensorboard or log file, '
                'closing this page does not affect training'
            }
        },
        'dataset_alert': {
            'value': {
                'zh': '请选择或填入一个数据集',
                'en': 'Please input or select a dataset'
            }
        },
        'submit': {
            'value': {
                'zh': '🚀 开始训练',
                'en': '🚀 Begin'
            }
        },
        'dry_run': {
            'label': {
                'zh': '仅生成运行命令',
                'en': 'Dry-run'
            },
            'info': {
                'zh': '仅生成运行命令，开发者自行运行',
                'en': 'Generate run command only, for manually running'
            }
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to train'
            }
        },
        'train_type': {
            'label': {
                'zh': '训练方式',
                'en': 'Train type'
            },
            'info': {
                'zh': '选择训练的方式',
                'en': 'Select the training type'
            }
        },
        'seed': {
            'label': {
                'zh': '随机数种子',
                'en': 'Seed'
            },
            'info': {
                'zh': '选择随机数种子',
                'en': 'Select a random seed'
            }
        },
        'torch_dtype': {
            'label': {
                'zh': '训练精度',
                'en': 'Training Precision'
            },
            'info': {
                'zh': '选择训练精度',
                'en': 'Select the training precision'
            }
        },
        'envs': {
            'label': {
                'zh': '环境变量',
                'en': 'Extra env vars'
            },
        },
        'use_ddp': {
            'label': {
                'zh': '使用DDP',
                'en': 'Use DDP'
            },
            'info': {
                'zh': '是否使用数据并行训练',
                'en': 'Use Distributed Data Parallel to train'
            }
        },
        'ddp_num': {
            'label': {
                'zh': 'DDP分片数量',
                'en': 'Number of DDP sharding'
            },
            'info': {
                'zh': '启用多少进程的数据并行',
                'en': 'The data parallel size of DDP'
            }
        },
        'tuner_backend': {
            'label': {
                'zh': 'Tuner backend',
                'en': 'Tuner backend'
            },
            'info': {
                'zh': 'tuner实现框架',
                'en': 'The tuner backend'
            }
        },
        'use_liger_kernel': {
            'label': {
                'zh': '使用Liger kernel',
                'en': 'Use Liger kernel'
            },
            'info': {
                'zh': 'Liger kernel可以有效降低显存使用',
                'en': 'Liger kernel can reduce memory usage'
            }
        },
        'train_param': {
            'label': {
                'zh': '训练参数设置',
                'en': 'Train settings'
            },
        },
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_grpo', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                GRPOModel.build_ui(base_tab)
                GRPODataset.build_ui(base_tab)
                Reward.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='train_type', scale=4, choices=list(get_supported_tuners()))
                        gr.Dropdown(elem_id='tuner_backend', scale=4)
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='torch_dtype', scale=4)
                    with gr.Row():
                        gr.Checkbox(elem_id='use_liger_kernel', scale=4)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='1', scale=4)
                GRPOHyper.build_ui(base_tab)
                GRPORuntime.build_ui(base_tab)
                with gr.Row(equal_height=True):
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(device_count)] + ['cpu'],
                        value=default_device,
                        scale=8)
                    gr.Textbox(elem_id='envs', scale=8)
                    gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                Rollout.build_ui(base_tab)
                GRPOTuner.build_ui(base_tab)
                RefModel.build_ui(base_tab)
                GRPOQuantization.build_ui(base_tab)
                GRPOSave.build_ui(base_tab)
                GRPOReportTo.build_ui(base_tab)
                GrpoAdvanced.build_ui(base_tab)
                GrpoAdvanced.build_ui(base_tab)

                cls.element('train_type').change(
                    GRPOHyper.update_lr,
                    inputs=[base_tab.element('train_type')],
                    outputs=[cls.element('learning_rate')])

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))

                submit.click(
                    cls.train_local,
                    list(cls.valid_elements().values()), [
                        cls.element('running_cmd'),
                        cls.element('logging_dir'),
                        cls.element('runtime_tab'),
                        cls.element('running_tasks'),
                        cls.element('train_record'),
                    ],
                    queue=True)

                base_tab.element('running_tasks').change(
                    partial(GRPORuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')] + GRPORuntime.all_plots)
                GRPORuntime.element('kill_task').click(
                    GRPORuntime.kill_task,
                    [GRPORuntime.element('running_tasks')],
                    [GRPORuntime.element('running_tasks')] + [GRPORuntime.element('log')] + GRPORuntime.all_plots,
                ).then(GRPORuntime.reset, [], [GRPORuntime.element('logging_dir')] + [GRPOHyper.element('output_dir')])

    @classmethod
    def prepare_sub_to_filter(cls):
        return ['train_type', 'opimizer',
                'vllm_mode'], GRPOTuner.tabs_to_filter + GRPOOptimizer.tabs_to_filter + Rollout.tabs_to_filter
