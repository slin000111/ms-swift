# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.llm_train.target import Target


class GRPOTarget(Target):

    group = 'llm_grpo'
