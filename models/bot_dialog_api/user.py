"""
定义用户资料
"""


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.session_id = ''
        self._history = []
        self.history = ''
        self.MAX_TURN = 7

    def get_service_id(self, session_id):
        self.session_id = session_id

    def update_history(self, text):
        self._history.append(text)
        self._history = self._history[-self.MAX_TURN*2-1:]
        self.history = ','.join(["\""+sent+"\"" for sent in self._history])

    def start_new_dialog(self):
        self.session_id = ''
        self._history = []
        self.history = ''

    def change_max_turn(self, max_turn):
        self.MAX_TURN = max_turn
