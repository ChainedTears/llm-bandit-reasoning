# TextBandit: Evaluating Decision-Making of Large Language Models in Uncertain Environments

## Abstract
Large language models (LLMs) have shown to be increasingly capable of performing reasoning tasks, but their ability to make sequential decisions under uncertainty only using natural language remains underexplored. We introduce a novel benchmark in which LLMs interact with multi-armed bandit environments using purely textual feedback, “you earned a token”, without access to numerical cues or explicit probabilities, resulting in the model to infer latent reward structures purely off linguistic cues and to adapt accordingly. We evaluated the  performance of four open-source LLMs and compare their performance to standard decision-making algorithms such as Thompson Sampling, Epsilon Greedy, Upper Confidence Bound (UCB), and random choice. While most of the LLMs underperformed compared to the baselines, Qwen3-4B, achieved the best-arm selection rate of 89.2\% , which significantly outperformed both the larger LLMs and traditional methods. Our findings suggest that probabilistic reasoning is able to emerge from language alone, and we present this benchmark as a step towards evaluating decision-making capabilities in naturalistic, non-numeric contexts.
\end{abstract}

## Features
- Evaluate LLMs in natural language-based multi-armed bandit tasks.
- Perform experiments using different LLMs, including Qwen3-4B, Llama-3.1-8B, Phi-2, and Qwen3-8B.
- Compare LLM performance with baseline decision-making algorithms.
- Track key metrics: Cumulative Reward, Best-Arm Selection Rate, and Cumulative Regret.
