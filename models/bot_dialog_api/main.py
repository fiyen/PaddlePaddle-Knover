"""
进行对话
"""
from dialog import dialog
from user import User


user_id = 'test_user'
user = User(user_id)

while True:
    human_ans = input()
    if len(human_ans) > 0:
        user.update_history(human_ans)
        robot_resp, session_id = dialog(human_ans, user.user_id, user.session_id, user.history)
        user.session_id = session_id
        user.update_history(robot_resp)
        print("Robot: %s" % robot_resp)
    else:
        break
user.start_new_dialog()
