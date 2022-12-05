# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import os
import json
import sys
import argparse
import contextlib
from collections import namedtuple

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import runnable
from paddlehub.module.nlp_module import DataFormatError
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving

import plato2_cn_small.models as plato_models
from plato2_cn_small.tasks.dialog_generation import DialogGeneration
from plato2_cn_small.utils import check_cuda, Timer
from plato2_cn_small.utils.args import parse_args

import translate as trans
import jieba
from collections import namedtuple


@moduleinfo(
    name="plato2_cn_small",
    version="1.0.0",
    summary=
    "A novel pre-training model for dialogue generation, incorporated with latent discrete variables for one-to-many relationship modeling. "
    "This model is a minor revision from plato2_en_base, making it be able to do conversation in Chinese and English (translated)",
    author="baidu-nlp, Dongyang Yan",
    author_email="dongyangyan@bjtu.edu.cn",
    type="nlp/text_generation",
)
class Plato(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError(
                "The module only support GPU. Please set the environment variable CUDA_VISIBLE_DEVICES."
            )

        args = self.setup_args()
        self.task = DialogGeneration(args)
        self.model = plato_models.create_model(args, fluid.CUDAPlace(0))
        self.Example = namedtuple("Example", ["src", "data_id"])
        self._interactive_mode = False
        self._from_lang = "cn"
        self._to_lang = "cn"
        self._trans_en2cn = trans.Translator("zh-cn", 'en')
        self._trans_cn2en = trans.Translator('en', 'zh-cn')

    def setup_args(self, tokenized=False):
        """
        Setup arguments.
        """
        assets_path = os.path.join(self.directory, "assets")
        vocab_path = os.path.join(assets_path, "vocab.txt")
        init_pretraining_params = os.path.join(assets_path, "12L", "Plato")
        spm_model_file = os.path.join(assets_path, "spm.model")
        nsp_inference_model_path = os.path.join(assets_path, "12L", "NSP")
        config_path = os.path.join(assets_path, "12L.json")

        # ArgumentParser.parse_args use argv[1:], it will drop the first one arg, so the first one in sys.argv should be ""
        if not tokenized:
            sys.argv = [
                "", "--model", "Plato", "--vocab_path",
                "%s" % vocab_path, "--do_lower_case", "False",
                "--init_pretraining_params",
                "%s" % init_pretraining_params, "--spm_model_file",
                "%s" % spm_model_file, "--nsp_inference_model_path",
                "%s" % nsp_inference_model_path, "--ranking_score", "nsp_score",
                "--do_generation", "True", "--batch_size", "1", "--config_path",
                "%s" % config_path
            ]
        else:
            sys.argv = [
                "", "--model", "Plato", "--data_format", "tokenized", "--vocab_path",
                "%s" % vocab_path, "--do_lower_case", "False",
                "--init_pretraining_params",
                "%s" % init_pretraining_params, "--spm_model_file",
                "%s" % spm_model_file, "--nsp_inference_model_path",
                "%s" % nsp_inference_model_path, "--ranking_score", "nsp_score",
                "--do_generation", "True", "--batch_size", "1", "--config_path",
                "%s" % config_path
            ]

        parser = argparse.ArgumentParser()
        plato_models.add_cmdline_args(parser)
        DialogGeneration.add_cmdline_args(parser)
        args = parse_args(parser)

        args.load(args.config_path, "Model")
        args.run_infer = True  # only build infer program

        return args

    @serving
    def generate(self, texts):
        """
        Get the robot responses of the input texts.

        Args:
             texts(list or str): If not in the interactive mode, texts should be a list in which every element is the chat context separated with '\t'.
                                 Otherwise, texts shoule be one sentence. The module can get the context automatically.

        Returns:
             results(list): the robot responses.
        """
        if not texts:
            return []
        if self._from_lang == 'cn':
            if not self._interactive_mode:
                texts = [' '.join(list(jieba.cut(text))) for text in texts]
            else:
                texts = ' '.join(list(jieba.cut(texts)))

        if self._interactive_mode:
            if isinstance(texts, str):
                if self._from_lang == 'en':
                    texts = self._trans_en2cn.translate(texts)
                self.context.append(texts.strip())
                texts = [" [SEP] ".join(self.context[-self.max_turn:])]
            else:
                raise ValueError(
                    "In the interactive mode, the input data should be a string."
                )
        elif not isinstance(texts, list):
            raise ValueError(
                "If not in the interactive mode, the input data should be a list."
            )
        else:
            if self._from_lang == 'en':
                texts = [self._trans_en2cn.translate(text) for text in texts]

        bot_responses = []
        for i, text in enumerate(texts):
            example = self.Example(src=text.replace("\t", " [SEP] "), data_id=i)
            record = self.task.reader._convert_example_to_record(
                example, is_infer=True)
            data = self.task.reader._pad_batch_records([record], is_infer=True)
            pred = self.task.infer_step(self.model, data)[0]  # batch_size is 1
            bot_response = pred["response"]  # ignore data_id and score
            bot_responses.append(bot_response)

        if self._interactive_mode:
            self.context.append(bot_responses[0].strip())
        if self._to_lang == 'en':
            bot_responses = [self._trans_cn2en.translate(resp) for resp in bot_responses]
        if self._to_lang == 'cn':
            bot_responses = [''.join(resp.split()) for resp in bot_responses]
        return bot_responses

    @serving
    def generate_for_test(self, records):
        """
        Get the robot responses of the input texts.

        Args:
             list of dicts: numerical data, [field_values, ...]
             field_values = {
            "token_ids": src_token_ids,
            "type_ids": src_type_ids,
            "pos_ids": src_pos_ids,
            "tgt_start_idx": tgt_start_idx
        }

        Returns:
             results(list): the robot responses.
        """
        if not records:
            return []

        if self._interactive_mode:
            print("Warning: This function is not suitable for interactive mode.")
        elif not isinstance(records, list):
            raise ValueError(
                "If not in the interactive mode, the input data should be a list."
            )
        fields = ["token_ids", "type_ids", "pos_ids", "tgt_start_idx", "data_id"]
        Record = namedtuple("Record", fields, defaults=(None,) * len(fields))
        record_all = []
        for i, record in enumerate(records):
            record["data_id"] = i
            record = Record(**record)
            record_all.append(record)
        data = self.task.reader._pad_batch_records(record_all, is_infer=True)
        pred = self.task.infer_step(self.model, data)
        bot_responses = [p["response"] for p in pred]

        return bot_responses

    def set_dialog_mode(self, from_lang='cn', to_lang='cn'):
        """
        To set the mode of dialog, from_lang is the language type of input, and
        to_lang is the language type from the robot. "cn": Chinese; "en": English.
        Default: from_lang is "cn", to_lang is "cn".
        """
        self._from_lang = from_lang
        self._to_lang = to_lang

    @contextlib.contextmanager
    def interactive_mode(self, max_turn=6):
        """
        Enter the interactive mode.

        Args:
            max_turn(int): the max dialogue turns. max_turn = 1 means the robot can only remember the last one utterance you have said.
        """
        self._interactive_mode = True
        self.max_turn = max_turn
        self.context = []
        yield
        self.context = []
        self._interactive_mode = False

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, optional.")

        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.generate(texts=input_data)

        return results


if __name__ == "__main__":
    module = Plato()
    for result in module.generate([
            "你是机器人吗？",
            "如果你不是机器人，那你得皮肤是什么颜色的呢？"
    ]):
        print(result)

    """
    import paddlehub as hub
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    module = hub.Module("plato2_en&cn_base")
    
    # change the dialog language.
    module.set_dialog_mode(from_lang='en', to_lang='cn')
    """

    with module.interactive_mode(max_turn=3):
        while True:
            human_utterance = input()
            robot_utterance = module.generate(human_utterance)
            print("Robot: %s" % robot_utterance[0])
