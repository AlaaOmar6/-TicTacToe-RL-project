
# 🎮 Tic-Tac-Toe Reinforcement Learning

### *Comparing Minimax, Dynamic Programming, Monte Carlo, and Temporal Difference Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()


## 📋 Table of Contents

- [Overview](#-overview)
- [Algorithms Implemented](#-algorithms-implemented)
- [Key Results](#-key-results)
- [State Space Analysis](#-state-space-analysis)
- [Installation](#-installation)
- [Usage](#-usage)
- [Learning Curves](#-learning-curves)
- [Theoretical Comparison](#-theoretical-comparison)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements and rigorously compares **four fundamental approaches** to solving Tic-Tac-Toe through the lens of Reinforcement Learning and Game Theory. The agent learns to play optimally as **Player X** against an optimal **Minimax opponent (Player O)**.

### What Makes This Project Unique?

- 🔍 **Complete state-space enumeration** — all 5,478 valid game states generated via DFS
- ✅ **Verified correctness** — DP and Minimax agree on **all 2,423 non-terminal X-states**
- 📊 **Comprehensive evaluation** — win/draw/loss tracking with ε-decay exploration
- 🧠 **True TD bootstrapping** — proper Q-Learning with `max_a' Q(s',a')` updates

---

## 🤖 Algorithms Implemented

| Algorithm | Type | Training Mode | Update Rule |
|-----------|------|---------------|-------------|
| **Minimax** | Model-based | N/A (analytical) | `V(s) = γ · max/min V(s')` |
| **Dynamic Programming** | Model-based | Full MDP knowledge | `V(s) = Σ π(a\|s)[R + γ·V(s')]` |
| **Monte Carlo** | Model-free | vs Minimax opponent | `Q(s,a) += (1/N)[G - Q(s,a)]` |
| **Q-Learning** | Model-free | Self-play | `Q += α[r + γ·max Q(s',a') - Q]` |
| **SARSA** | Model-free | Self-play | `Q += α[r + γ·Q(s',a') - Q]` |

### Design Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Discount factor (γ) | **0.9** | Balances immediate vs. future rewards |
| Rewards | **Win: +1, Loss: -1, Draw: 0** | Sparse terminal rewards |
| MC learning rate | **1/N** (incremental mean) | Unbiased sample average |
| TD learning rate (α) | **0.1** | Stable convergence |
| ε-decay | **0.99995** → min **0.1** | Sufficient exploration |

---

## 🏆 Key Results

### Optimal Play Verification

```
✅ V(empty board) = 0.0000        (draw under optimal play)
✅ DP-Minimax mismatches: 0       (verified on all 2,423 states)
✅ DP converged in 6 iterations
```

### Greedy Policy Performance (ε = 0)

| Agent | Win Rate | Draw Rate | Loss Rate | Status |
|-------|----------|-----------|-----------|--------|
| **Monte Carlo** | ~0% | **>90%** | <5% | ✅ Optimal |
| **Q-Learning** | ~0% | **>90%** | <5% | ✅ Optimal |
| **SARSA** | ~0% | **>90%** | <5% | ✅ Optimal |

> All agents successfully learn to **force a draw** against an unbeatable Minimax opponent — the theoretical best outcome in Tic-Tac-Toe.

---

## 🔬 State Space Analysis

The project rigorously verifies the state space composition:

```
Total valid states (DFS):        5,478
├── Terminal states:               958
└── Non-terminal states:         4,520
    ├── X-player states:         2,599
    │   ├── Terminal X-states:     176
    │   └── Non-terminal X-states: 2,423  ← Decision states for the agent
    └── O-player states:         1,921
        ├── Terminal O-states:     782
        └── Non-terminal O-states: 1,139
```

> **Note:** The project's reference to "2,423 states" specifically denotes the **non-terminal X-player decision states** where the agent must select an action. Terminal states have fixed rewards and require no decisions.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- `matplotlib` for visualization

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/tictactoe-rl-project.git
cd tictactoe-rl-project

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
matplotlib>=3.5.0
```

---

## 🚀 Usage

### Run the Full Pipeline

```bash
python tictactoe_rl_project2.py
```

This executes the complete workflow:

1. **State Space Verification** — generates and validates all 5,478 states
2. **Minimax Computation** — recursive value function with gamma discounting
3. **Dynamic Programming** — policy iteration until convergence
4. **Consistency Check** — verifies DP ≡ Minimax on all 2,423 states
5. **Monte Carlo Training** — 100,000 episodes vs. optimal opponent
6. **Q-Learning Training** — 100,000 episodes via self-play
7. **SARSA Training** — 100,000 episodes via self-play
8. **Greedy Evaluation** — deterministic policy assessment
9. **Learning Curves** — matplotlib visualization saved to `learning_curves_final.png`

### Expected Output

```
============================================================
STATE SPACE VERIFICATION
============================================================
Total valid states (DFS): 5478
  Terminal states: 958
  Non-terminal states: 4520
...
Minimax V(empty board) = 0.0000
DP converged in 6 iterations
DP V(empty board) = 0.0000
DP-Minimax mismatches: 0 (must be 0)
DP and Minimax are 100% consistent on all 2,423 non-terminal X-states!
...
```

---

## 📈 Learning Curves

The project generates comparative learning curves showing the evolution of **win**, **draw**, and **loss** rates across training episodes:

| Panel | Algorithm | Training Mode |
|-------|-----------|---------------|
| Left | Monte Carlo | vs. Minimax Opponent |
| Center | Q-Learning | Self-Play (Off-Policy TD) |
| Right | SARSA | Self-Play (On-Policy TD) |

![Learning Curves](learning_curves_final.png)

> All three methods converge to near-optimal play with draw rates exceeding 90%.

---

## 📚 Theoretical Comparison

### Algorithm Characteristics

| Aspect | Minimax | DP | Monte Carlo | Q-Learning | SARSA |
|--------|---------|-----|-------------|------------|-------|
| **Requires Model** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Bootstrapping** | N/A | N/A | ❌ No | ✅ Yes | ✅ Yes |
| **Exploration** | N/A | N/A | ε-greedy | ε-greedy | ε-greedy |
| **Bias/Variance** | Zero/Zero | Zero/Zero | Low/High | Med/Low | Med/Low |
| **Sample Efficiency** | N/A | N/A | Low | High | High |
| **Convergence** | Immediate | ~6 iterations | ~50k episodes | ~30k episodes | ~40k episodes |

### Why Self-Play for TD Methods?

Training against a perfect Minimax opponent provides **limited exploration** — the opponent always responds optimally, restricting the agent's exposure to diverse game trajectories. **Self-play** enables the agent to explore both sides of the game, discovering winning strategies through interaction with its own evolving policy.

### Why TRUE Q-Learning Bootstrapping?

True Q-Learning employs the **max operator** in its TD target:

```
TD Target = r + γ · max_a' Q(s', a')
```

This allows the agent to learn from incomplete episodes and propagate value information more efficiently than Monte Carlo methods, which must wait until episode termination.

---

## 📁 Project Structure

```
tictactoe-rl-project/
├── tictactoe_rl_project2.py      # Main implementation (all algorithms)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── learning_curves_final.png      # Generated visualization
└── .gitignore                     # Git ignore rules
```

### Code Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SECTION 1: Constants                 │
├─────────────────────────────────────────────────────────┤
│                    SECTION 2: Board Utilities           │
│  • State encoding/decoding                              │
│  • Move generation & validation                         │
│  • Win/draw detection                                   │
│  • State-space DFS generator                            │
├─────────────────────────────────────────────────────────┤
│                    SECTION 3: State Verification        │
│  • 2,423 non-terminal X-state assertion                 │
├─────────────────────────────────────────────────────────┤
│  SECTION 4: Minimax    │  SECTION 5: DP Policy Iteration│
│  • Recursive search    │  • Value iteration             │
│  • Gamma discounting   │  • Policy improvement          │
├─────────────────────────────────────────────────────────┤
│                    SECTION 6: Consistency Check         │
│  • Verify DP ≡ Minimax on all 2,423 states              │
├─────────────────────────────────────────────────────────┤
│  SECTION 7: Monte Carlo │  SECTION 8: Q-Learning        │
│  • Every-Visit MC       │  • Off-Policy TD              │
│  • Incremental mean     │  • True bootstrapping         │
│  • vs Minimax opponent  │  • Self-play                  │
├─────────────────────────────────────────────────────────┤
│                    SECTION 9: SARSA (Optional)          │
│  • On-Policy TD                                         │
├─────────────────────────────────────────────────────────┤
│                    SECTION 10: Greedy Evaluation        │
│  • ε=0 deterministic policy testing                     │
├─────────────────────────────────────────────────────────┤
│                    SECTION 11: Learning Curves          │
│  • Matplotlib visualization                             │
├─────────────────────────────────────────────────────────┤
│                    SECTION 12: Theoretical Comparison   │
│  • Algorithm analysis & design rationale                │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Acknowledgments

- **Instructor:** Eng. Abdelrahman Shehata — for the project requirements and guidance
- **Course:** Reinforcement Learning
- **References:**
  - Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd Edition)
  - Russell & Norvig — *Artificial Intelligence: A Modern Approach*

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

</div>
