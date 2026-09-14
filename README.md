# Fine-tuning an LLM on PubMedQA using GRPO

> **Note:** This notebook was created and run in Spring 2025. The code may be outdated due to changes in the Unsloth, TRL, or vLLM libraries. You may need to adjust imports and configuration parameters to work with current versions.

This repository contains a Jupyter notebook (`main.ipynb`) that demonstrates how to fine-tune a large language model (LLM) on the **PubMedQA** dataset using **Group Relative Policy Optimization (GRPO)**. The goal is to train a biomedical research assistant capable of reading scientific abstracts and answering yes/no/maybe questions with reasoned, evidence-based explanations.

## Overview

The notebook fine-tunes **Qwen2.5-1.5B-Instruct** (4-bit quantized) using the [Unsloth](https://github.com/unslothai/unsloth) library for efficient training. It employs GRPO — a reinforcement learning from human feedback (RLHF) algorithm implemented in [TRL](https://github.com/huggingface/trl) — with a custom set of reward functions designed to encourage:

- **Correct final answers** (yes / no / maybe)
- **Well-structured, paragraph-style reasoning** inside XML tags
- **Faithfulness** to the source abstract (keyword overlap and length matching)
- **Proper formatting** (avoiding bullet points, lists, and leaking the answer into the reasoning)

### Task Format

The model is prompted to respond in the following structured format:

```xml
<long_answer>
[Detailed, step-by-step reasoning with evidence from the abstract]
</long_answer>
<answer>
[yes | no | maybe]
</answer>
```

## Notebook Structure

| Section | Description |
|---------|-------------|
| **Setup & Model Loading** | Loads the 4-bit quantized Qwen2.5-1.5B-Instruct model via Unsloth with LoRA adaptation (rank 64). Enables fast inference and bfloat16 support. |
| **Data Processing** | Loads the `pragmaticcs/pubmedqa-prompts` dataset from Hugging Face, combines labeled and artificial-balanced subsets, and creates an 80/20 train/test split. Data is formatted into conversation prompts with a system prompt. |
| **Training Preparation** | Imports NLP utilities (NLTK tokenization, stopword removal, lemmatization) and defines **7 reward functions** used during GRPO training. |
| **Training** | Configures and runs the `GRPOTrainer` with vLLM-accelerated generation. Training runs for 1 epoch with a learning rate of 5e-6 and 6 generations per prompt. |
| **Evaluation** | Generates responses on the test set and prints examples showing the question, expected answer, model response, and correctness flag. |

## Reward Functions

The GRPO trainer uses a weighted combination of 7 reward functions (weights sum to 1.0):

| Reward Function | Weight | Purpose |
|-----------------|--------|---------|
| `answer_correctness_reward_func` | 0.25 | +1.0 for correct yes/no/maybe answer, -1.0 for incorrect |
| `keyword_overlap_reward_func` | 0.15 | Rewards overlap of lemmatized keywords between generated reasoning and ground-truth long answer |
| `correct_reasoning_length_reward_func` | 0.15 | Rewards similarity in token count between generated reasoning and ground-truth long answer |
| `no_answer_in_reasoning_reward_func` | 0.05 | 0.0 penalty if "yes", "no", or "maybe" appears in the reasoning section |
| `xmlcount_reward_func` | 0.15 | Rewards presence of all four XML tags (`<long_answer>`, `</long_answer>`, `<answer>`, `</answer>`) |
| `strict_format_reward_func` | 0.15 | 1.0 only if the response exactly matches the expected strict XML pattern |
| `paragraph_format_reward_func` | 0.10 | Rewards paragraph-style prose; penalizes bullet points, numbered lists, and excessive newlines |

## Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | 4-bit quantized Qwen2.5 1.5B instruct model |
| LoRA rank | 64 | Rank for LoRA adaptation |
| Max sequence length | 2048 | Maximum token length for completions |
| Learning rate | 5e-6 | Optimizer learning rate |
| Batch size | 6 | Per-device training batch size |
| Generations | 6 | Number of responses sampled per prompt in GRPO |
| Training epochs | 1 | Number of passes over the training data |
| Output directory | `outputs/` | Directory for checkpoints and logs |

## Dataset

The notebook uses the [`pragmaticcs/pubmedqa-prompts`](https://huggingface.co/datasets/pragmaticcs/pubmedqa-prompts) dataset, which contains:

- **`labeled`**: PubMedQA instances with labeled answers and long-form explanations
- **`artificial_balanced`**: Artificially balanced subset used to augment training data (first 1600 samples are used)

Each sample includes a scientific abstract, a yes/no/maybe question, the expected answer, and a ground-truth long answer.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
