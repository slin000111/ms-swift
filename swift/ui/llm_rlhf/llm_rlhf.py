# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import os
import re
import sys
import time
from functools import partial
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, Type

import gradio as gr
import json
import torch
from json import JSONDecodeError
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from swift.llm import RLHFArguments
from swift.llm.argument.base_args.base_args import get_supported_tuners
from swift.ui.base import BaseUI
from swift.ui.llm_rlhf.advanced import RLHFAdvanced
from swift.ui.llm_rlhf.dataset import RLHFDataset
from swift.ui.llm_rlhf.hyper import RLHFHyper
from swift.ui.llm_rlhf.model import RLHFModel
from swift.ui.llm_rlhf.optimizer import RLHFOptimizer
from swift.ui.llm_rlhf.quantization import RLHFQuantization
from swift.ui.llm_rlhf.report_to import RLHFReportTo
from swift.ui.llm_rlhf.rlhf import RLHF
from swift.ui.llm_rlhf.runtime import RLHFRuntime
from swift.ui.llm_rlhf.save import RLHFSave
from swift.ui.llm_rlhf.tuner import RLHFTuner
from swift.ui.llm_train.llm_train import LLMTrain
from swift.utils import get_device_count, get_logger

logger = get_logger()


class LLMRLHF(LLMTrain):

    group = 'llm_rlhf'

    sub_ui = [
        RLHFModel, RLHFDataset, RLHFRuntime, RLHFSave, RLHFHyper, RLHFQuantization, RLHFAdvanced, RLHFOptimizer, RLHF,
        RLHFReportTo, RLHFTuner
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_rlhf': {
            'label': {
                'zh': 'LLM RLHF',
                'en': 'LLM RLHF',
            }
        },
        'train_stage': {
            'label': {
                'zh': '训练Stage',
                'en': 'Train Stage'
            },
            'info': {
                'zh': '请注意选择与此匹配的数据集，人类对齐配置在页面下方',
                'en': 'Please choose matched dataset, RLHF settings is at the bottom of the page'
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
        with gr.TabItem(elem_id='llm_rlhf', label=''):
            default_device = 'cpu'
            device_count = get_device_count()
            if device_count > 0:
                default_device = '0'
            with gr.Blocks():
                RLHFModel.build_ui(base_tab)
                RLHFDataset.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(elem_id='train_type', scale=2, choices=list(get_supported_tuners()))
                        gr.Dropdown(elem_id='tuner_backend', scale=2)
                    with gr.Row():
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='torch_dtype', scale=4)
                        gr.Checkbox(elem_id='use_liger_kernel', scale=4)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='1', scale=4)
                RLHFHyper.build_ui(base_tab)
                RLHFRuntime.build_ui(base_tab)
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

                RLHFTuner.build_ui(base_tab)
                RLHFOptimizer.build_ui(base_tab)
                RLHF.build_ui(base_tab)
                RLHFQuantization.build_ui(base_tab)
                RLHFSave.build_ui(base_tab)
                RLHFReportTo.build_ui(base_tab)
                RLHFAdvanced.build_ui(base_tab)

                base_tab.element('gpu_id').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                base_tab.element('use_ddp').change(
                    cls.update_ddp_num,
                    [base_tab.element('gpu_id'), base_tab.element('use_ddp')], base_tab.element('ddp_num'))
                cls.element('train_type').change(
                    RLHFHyper.update_lr,
                    inputs=[base_tab.element('train_type')],
                    outputs=[cls.element('learning_rate')])

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
                    partial(RLHFRuntime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                    list(base_tab.valid_elements().values()) + [cls.element('log')] + RLHFRuntime.all_plots)
                RLHFRuntime.element('kill_task').click(
                    RLHFRuntime.kill_task,
                    [RLHFRuntime.element('running_tasks')],
                    [RLHFRuntime.element('running_tasks')] + [RLHFRuntime.element('log')] + RLHFRuntime.all_plots,
                ).then(RLHFRuntime.reset, [], [RLHFRuntime.element('logging_dir')] + [RLHFHyper.element('output_dir')])

    @classmethod
    def train(cls, *args):
        ignore_elements = ('logging_dir', 'more_params', 'train_stage', 'envs')
        default_args = cls.get_default_value_from_dataclass(RLHFArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = cls.valid_element_keys()
        if cls.group == 'llm_grpo':
            train_stage = 'rlhf'
        else:
            train_stage = 'sft'
        for key, value in zip(keys, args):
            compare_value = default_args.get(key)
            if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                value = int(value)
            elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                value = float(value)
            elif isinstance(value, str) and re.fullmatch(cls.bool_regex, value):
                value = True if value.lower() == 'true' else False
            if key not in ignore_elements and key in default_args and compare_value != value and value:
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list) or getattr(cls.element(key), 'is_list', False)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                try:
                    more_params = json.loads(value)
                except (JSONDecodeError or TypeError):
                    more_params_cmd = value

            if key == 'train_stage':
                train_stage = value

        kwargs.update(more_params)
        if 'dataset' not in kwargs and 'custom_train_dataset_path' not in kwargs:
            raise gr.Error(cls.locale('dataset_alert', cls.lang)['value'])

        model = kwargs.get('model')
        if os.path.exists(model) and os.path.exists(os.path.join(model, 'args.json')):
            kwargs['resume_from_checkpoint'] = kwargs.pop('model')

        cmd = train_stage
        if kwargs.get('deepspeed'):
            more_params_cmd += f' --deepspeed {kwargs.pop("deepspeed")} '
        use_liger_kernel = kwargs.get('use_liger_kernel', None)
        if use_liger_kernel:
            kwargs.pop('use_liger_kernel')
        tabs_relation_dict = cls.prepare_sub_to_filter()
        cls.remove_useless_args(kwargs, tabs_relation_dict)
        try:
            sft_args = RLHFArguments(
                **{
                    key: value.split(' ') if kwargs_is_list.get(key, False) and isinstance(value, str) else value
                    for key, value in kwargs.items()
                })
        except Exception as e:
            if 'using `--model`' in str(e):  # TODO a dirty fix
                kwargs['model'] = kwargs.pop('resume_from_checkpoint')
                sft_args = RLHFArguments(
                    **{
                        key: value.split(' ') if kwargs_is_list.get(key, False) and isinstance(value, str) else value
                        for key, value in kwargs.items()
                    })
            else:
                raise e
        params = ''

        if cls.group == 'llm_grpo' and sys.platform != 'win32':
            params += f'--rlhf_type {cls.quote}grpo{cls.quote} '

        sep = f'{cls.quote} {cls.quote}'
        for e in kwargs:
            if isinstance(kwargs[e], list):
                params += f'--{e} {cls.quote}{sep.join(kwargs[e])}{cls.quote} '
            elif e in kwargs_is_list and kwargs_is_list[e]:
                all_args = [arg for arg in kwargs[e].split(' ') if arg.strip()]
                params += f'--{e} {cls.quote}{sep.join(all_args)}{cls.quote} '
            else:
                params += f'--{e} {cls.quote}{kwargs[e]}{cls.quote} '
        if use_liger_kernel:
            params += f'--use_liger_kernel {cls.quote}{use_liger_kernel}{cls.quote}'
        params += more_params_cmd + ' '
        params += f'--add_version False --output_dir {sft_args.output_dir} ' \
                  f'--logging_dir {sft_args.logging_dir} --ignore_args_error True'
        ddp_param = ''
        devices = other_kwargs['gpu_id']
        envs = other_kwargs['envs'] or ''
        envs = envs.strip()
        devices = [d for d in devices if d]
        if other_kwargs['use_ddp']:
            assert int(other_kwargs['ddp_num']) > 0
            ddp_param = f'NPROC_PER_NODE={int(other_kwargs["ddp_num"])}'
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            if is_torch_npu_available():
                cuda_param = f'ASCEND_RT_VISIBLE_DEVICES={gpus}'
            elif is_torch_cuda_available():
                cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'
            else:
                cuda_param = ''

        log_file = os.path.join(sft_args.logging_dir, 'run.log')
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            if ddp_param:
                ddp_param = f'set {ddp_param} && '
            if envs:
                envs = [env.strip() for env in envs.split(' ') if env.strip()]
                _envs = ''
                for env in envs:
                    _envs += f'set {env} && '
                envs = _envs
            run_command = f'{cuda_param}{ddp_param}{envs}start /b swift sft {params} > {log_file} 2>&1'
        else:
            run_command = f'{cuda_param} {ddp_param} {envs} nohup swift {cmd} {params} > {log_file} 2>&1 &'
        logger.info(f'Run training: {run_command}')
        if model:
            record = {}
            for key, value in zip(keys, args):
                if key in default_args or key in ('more_params', 'train_stage', 'use_ddp', 'ddp_num', 'gpu_id', 'envs'):
                    record[key] = value or None
            cls.save_cache(model, record)
        return run_command, sft_args, other_kwargs

    @classmethod
    def prepare_sub_to_filter(cls):
        tabs_relation_dict = {
            key: val
            for key, val in zip(['train_type', 'opimizer'], [RLHFTuner.tabs_to_filter, RLHFOptimizer.tabs_to_filter])
        }
        return tabs_relation_dict
