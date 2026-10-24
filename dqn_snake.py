import cv2
import json
import numpy as np
import mss
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import time
from typing import Optional

# -----------------------------
# Parameters
# -----------------------------
ROI = {"top": 250, "left": 550, "width": 820, "height": 750}
SCORE_ROI = {"top": 150, "left": 600, "width": 100, "height": 80}  # score area
SCORE_BLUR_KERNEL = (5, 5)
SCORE_DIFF_THRESHOLD = 15
SCORE_DIFF_MIN_PIXELS = 120
SCORE_DIFF_MEAN_THRESHOLD = 2.5
END_ROI = {"top": 150, "left": 1000, "width": 100, "height": 50}
END_FRAME_PATH = "end.txt"
SCORE_PATH = "st_score.txt"

STATE_SHAPE = (84, 84)
ACTIONS = ['w', 'a', 's', 'd']
MEMORY_SIZE = 5000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 1e-4
MODEL_PATH = "dqn_snake.pth"

# -----------------------------
# Load zero score template
# -----------------------------
def load_score_zero(path=SCORE_PATH) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[WARN] Zero score template '{path}' not found.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return np.array(data, dtype=np.uint8)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load zero score from '{path}': {exc}")
        return None

ZERO_SCORE_TEMPLATE = load_score_zero()


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Score ROI template must be 2D or 3D array")


def _blur_score_roi(gray_roi: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray_roi, SCORE_BLUR_KERNEL, 0)


def reset_score_roi() -> Optional[np.ndarray]:
    if ZERO_SCORE_TEMPLATE is None:
        return None
    template = _ensure_gray(ZERO_SCORE_TEMPLATE)
    expected_shape = (SCORE_ROI["height"], SCORE_ROI["width"])
    if template.shape != expected_shape:
        template = cv2.resize(template, expected_shape[::-1])
    return _blur_score_roi(template)


current_score_roi = reset_score_roi()

# -----------------------------
# Load game over template
# -----------------------------
def load_end_frame(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[WARN] Game over template '{path}' not found. Falling back to frame-difference detection.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        array = np.array(data, dtype=np.uint8)
        return array
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load template from '{path}': {exc}. Falling back to frame-difference detection.")
        return None

END_FRAME_TEMPLATE = load_end_frame(END_FRAME_PATH)

# -----------------------------
# DQN Network
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
# Replay Memory
# -----------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -----------------------------
# Utilities
# -----------------------------
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, STATE_SHAPE)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def select_action(state, epsilon, net, device):
    if random.random() < epsilon:
        return random.randint(0, len(ACTIONS)-1)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = net(state)
    return q_values.argmax().item()

def perform_action(action_index):
    key = ACTIONS[action_index]
    pyautogui.press(key)
    print(f"[ACTION] Move: {key}")

# --------- Game Over Detection ----------
last_end_frame = None
no_change_counter = 0
def check_game_over(end_frame: np.ndarray) -> bool:
    global last_end_frame, no_change_counter
    if END_FRAME_TEMPLATE is not None:
        if end_frame.shape != END_FRAME_TEMPLATE.shape:
            return False
        return np.array_equal(end_frame, END_FRAME_TEMPLATE)

    gray = cv2.cvtColor(end_frame, cv2.COLOR_BGR2GRAY)
    if last_end_frame is None:
        last_end_frame = gray
        return False

    diff = cv2.absdiff(last_end_frame, gray)
    non_zero_count = np.count_nonzero(diff)
    last_end_frame = gray

    if non_zero_count < 1000:
        no_change_counter += 1
    else:
        no_change_counter = 0
    return no_change_counter > 2

# --------- Food Detection (using zero-score + variable) ----------
def food_eaten_check(full_frame, game_over: bool) -> bool:
    global current_score_roi

    x, y, w, h = SCORE_ROI["left"], SCORE_ROI["top"], SCORE_ROI["width"], SCORE_ROI["height"]
    score_frame = full_frame[y:y+h, x:x+w]
    score_gray = cv2.cvtColor(score_frame, cv2.COLOR_BGR2GRAY)
    score_blur = _blur_score_roi(score_gray)

    if current_score_roi is None:
        current_score_roi = score_blur
        return False

    if current_score_roi is None or score_blur.shape != current_score_roi.shape:
        current_score_roi = score_blur
        return False

    diff = cv2.absdiff(score_blur, current_score_roi)
    change_pixels = np.count_nonzero(diff > SCORE_DIFF_THRESHOLD)
    mean_diff = float(diff.mean())

    if not game_over and (change_pixels > SCORE_DIFF_MIN_PIXELS or mean_diff > SCORE_DIFF_MEAN_THRESHOLD):
        print("[EVENT] Food eaten detected by score change.")
        current_score_roi = score_blur  # update variable only
        return True

    if not game_over:
        current_score_roi = score_blur

    return False

# -----------------------------
# Main Loop
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DQN((1, *STATE_SHAPE), len(ACTIONS)).to(device)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

if os.path.exists(MODEL_PATH):
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    print("Loaded existing model.")

epsilon = EPSILON_START
last_food_time = time.time()
prev_frame = None

os.makedirs("food_frames", exist_ok=True)

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(ROI))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        end_img = np.array(sct.grab(END_ROI))
        end_frame = cv2.cvtColor(end_img, cv2.COLOR_BGRA2BGR)
        state = preprocess(frame)

        alive = not check_game_over(end_frame)
        reward = 0

        # --- Reward logic ---
        if food_eaten_check(frame, not alive):
            reward = 1
            last_food_time = time.time()
            print("[EVENT] Food eaten! +1 reward")

            timestamp = int(time.time() * 1000)
            if prev_frame is not None:
                cv2.imwrite(f"food_frames/food_prev_{timestamp}.png", prev_frame)
            cv2.imwrite(f"food_frames/food_curr_{timestamp}.png", frame)

        elif not alive:
            reward = -3
            print("[GAME OVER] Restarting... (ESC + SPACE)")
            pyautogui.press("esc")
            pyautogui.press("space")
            last_food_time = time.time()
            current_score_roi = reset_score_roi()
            time.sleep(1)
            continue

        elif time.time() - last_food_time > 10:
            reward = -3
            print("[TIMEOUT] Too slow! -3 reward (ESC + SPACE)")
            pyautogui.press("esc")
            pyautogui.press("space")
            last_food_time = time.time()
            current_score_roi = reset_score_roi()
            time.sleep(1)
            continue

        # --- Agent Action ---
        action_index = select_action(state, epsilon, net, device)
        perform_action(action_index)

        next_state = preprocess(frame)
        done = not alive
        memory.push(state, action_index, reward, next_state, done)

        # Training
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + GAMMA * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        prev_frame = frame.copy()

        cv2.imshow("Snake", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

torch.save(net.state_dict(), MODEL_PATH)
cv2.destroyAllWindows()
