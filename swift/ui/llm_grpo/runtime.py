# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.ui.llm_train.runtime import Runtime
from swift.utils import get_logger

logger = get_logger()


class GRPORuntime(Runtime):

    group = 'llm_grpo'

    locale_dict = {
        'runtime_tab': {
            'label': {
                'zh': '运行时',
                'en': 'Runtime'
            },
        },
        'tb_not_found': {
            'value': {
                'zh': 'tensorboard未安装,使用pip install tensorboard进行安装',
                'en': 'tensorboard not found, install it by pip install tensorboard',
            }
        },
        'running_cmd': {
            'label': {
                'zh': '运行命令',
                'en': 'Command line'
            },
            'info': {
                'zh': '执行的实际命令',
                'en': 'The actual command'
            }
        },
        'show_log': {
            'value': {
                'zh': '展示运行状态',
                'en': 'Show running status'
            },
        },
        'stop_show_log': {
            'value': {
                'zh': '停止展示运行状态',
                'en': 'Stop showing running status'
            },
        },
        'logging_dir': {
            'label': {
                'zh': '日志路径',
                'en': 'Logging dir'
            },
            'info': {
                'zh': '支持手动传入文件路径',
                'en': 'Support fill custom path in'
            }
        },
        'log': {
            'label': {
                'zh': '日志输出',
                'en': 'Logging content'
            },
            'info': {
                'zh': '如果日志无更新请再次点击"展示日志内容"',
                'en': 'Please press "Show log" if the log content is not updating'
            }
        },
        'running_tasks': {
            'label': {
                'zh': '运行中任务',
                'en': 'Running Tasks'
            },
            'info': {
                'zh': '运行中的任务（所有的swift rlhf命令）',
                'en': 'All running tasks(started by swift rlhf)'
            }
        },
        'refresh_tasks': {
            'value': {
                'zh': '找回运行时任务',
                'en': 'Find running tasks'
            },
        },
        'kill_task': {
            'value': {
                'zh': '杀死任务',
                'en': 'Kill running task'
            },
        },
        'tb_url': {
            'label': {
                'zh': 'Tensorboard链接',
                'en': 'Tensorboard URL'
            },
            'info': {
                'zh': '仅展示，不可编辑',
                'en': 'Not editable'
            }
        },
        'start_tb': {
            'value': {
                'zh': '打开TensorBoard',
                'en': 'Start TensorBoard'
            },
        },
        'close_tb': {
            'value': {
                'zh': '关闭TensorBoard',
                'en': 'Close TensorBoard'
            },
        },
    }
