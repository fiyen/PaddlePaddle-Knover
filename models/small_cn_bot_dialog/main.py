import paddlehub as hub
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
module = hub.Module(directory="plato2_cn_small", use_plato=True)

with module.interactive_mode(max_turn=3):
    while True:
        human_utterance = input()
        robot_utterance = module.generate(human_utterance)
        print("Robot: %s" % robot_utterance[0])
