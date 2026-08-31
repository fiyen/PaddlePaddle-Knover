"""PaddleHub module interface for plato2_cn_small."""

import os


class Module(object):
    """Module for the plato2_cn_small dialog model."""

    def __init__(self, directory="plato2_cn_small", use_plato=True, **kwargs):
        self.directory = directory
        self.use_plato = use_plato
        self._turn = 0
        self._max_turn = None

    def interactive_mode(self, max_turn=3, print_response=True, show_progress=True):
        self._max_turn = max_turn
        self._turn = 0
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def generate(self, human_utterance):
        self._turn += 1
        return ["%s" % human_utterance]
