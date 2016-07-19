from ale_python_interface import ALEInterface
import numpy as np
import cv2

ale = ALEInterface()
ale.setInt('random_seed', 123)
ale.setInt('frame_skip', 4)
rom_name = 'freeway'
ale.loadROM('roms/%s.bin' % rom_name)
actions = np.array(ale.getMinimalActionSet())
w, h = ale.getScreenDims()


def _act(action, rgb=False):
    ale.act(action)
    if rgb:
        frame = np.zeros(h*w*3, dtype=np.uint8)
        ale.getScreenRGB(frame)
    else:
        frame = np.zeros(h*w, dtype=np.uint8)
        ale.getScreen(frame)
    if ale.game_over():
        ale.reset_game()
    if rgb:
        return frame.reshape((h, w, 3))
    else:
        return prepareFrame(frame.reshape((h, w)))

def prepareFrame(frame):
    return cv2.resize(frame, (84, 110))[26:110, :]

def getFrames(n, rgb=False):
    random_action_indices = np.random.randint(0, len(actions), n)
    random_actions = actions[random_action_indices]
    frames = []
    for random_action in random_actions:
        frames.append(_act(random_action, rgb))
    return frames

if __name__ == '__main__':
    while True:
        img = getFrames(1)[0]
        cv2.imshow('blah', img)
        cv2.waitKey(1)
