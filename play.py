import cv2
import numpy as np
import mss
import pyautogui
import torch
import torch.nn as nn
import os
import time

# -----------------------------
# Parameters
# -----------------------------
ROI = {"top": 250, "left": 550, "width": 820, "height": 750}
STATE_SHAPE = (84, 84)
ACTIONS = ['w', 'a', 's', 'd']
MODEL_PATH = "dqn_snake.pth" # add the path of your saved model or use the default model provided in the repo

# -----------------------------
# DQN Network (same as training)
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Utilities
# -----------------------------
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, STATE_SHAPE)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def select_action(state, net, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = net(state)
    return q_values.argmax().item()

def perform_action(action_index):
    key = ACTIONS[action_index]
    pyautogui.press(key)
    print(f"[ACTION] {key}")

# -----------------------------
# Main Loop (Play only)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DQN((1, *STATE_SHAPE), len(ACTIONS)).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval()
print(f"[MODEL] Loaded pretrained model from {MODEL_PATH}")

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(ROI))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        state = preprocess(frame)

        # Agent picks best action
        action_index = select_action(state, net, device)
        perform_action(action_index)

        cv2.imshow("Snake Bot Playing", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
