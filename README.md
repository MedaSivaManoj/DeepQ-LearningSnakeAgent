# Reinforcement Learning Snake Bot 🐍🤖

This project implements a **Deep Q-Network (DQN)** agent that learns to play the **Google Snake Game** using **screen capture** and **keyboard simulation**.

## Features
- **Screen Capture:** Uses `mss` and `OpenCV` to grab the game screen in real-time.
- **State Preprocessing:** Frames are resized (84x84), converted to grayscale, and normalized.
- **Deep Q-Network (DQN):**
  - Convolutional neural network (CNN) approximates Q-values.
  - Epsilon-greedy exploration strategy.
  - Replay memory for stable training.
- **Reward System:**
  - `+1` reward for eating food (detected via score change).
  - `-1` reward for inactivity/timeouts.
  - `-3` penalty for game over.
- **Game Over Detection:** Either via saved template (`end.txt`) or fallback frame-difference detection.
- **Model Saving & Reuse:** Best-performing model automatically saved and reused across sessions.

## Requirements
Install dependencies:
```bash
pip install opencv-python mss pyautogui torch numpy
```

## How It Works
1. Captures the **Snake game screen** within a fixed Region of Interest (ROI).
2. Preprocesses the frame into a suitable state representation.
3. Selects an action (`w`, `a`, `s`, `d`) using epsilon-greedy strategy.
4. Executes the action via `pyautogui`.
5. Detects **food eaten** (via score change) and **game over** (via template/difference).
6. Updates replay memory and trains the DQN.

## Training
- Model: CNN-based DQN (`dqn_snake.pth`).
- Optimizer: Adam (lr=1e-4).
- Discount factor: γ = 0.99.
- Epsilon decay from 1.0 → 0.1.
- Replay buffer size: 5000, batch size: 32.

## File Structure
```
├── dqn_snake.py         # Main training loop
├── st_score.txt         # Zero-score reference template
├── end.txt              # Game over screen template
├── dqn_snake.pth        # Saved DQN model (auto-generated)
├── food_frames/         # Saved frames around food events (optional)
```

## Demo
The bot continuously plays Snake, learning over multiple episodes.  
- Eats food → +1 reward  
- Crashes → negative reward  
- Saves best model automatically.  

## Tech Stack
- **Languages:** Python  
- **Libraries:** PyTorch, OpenCV, MSS, PyAutoGUI, NumPy  
