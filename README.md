# 🏏 AMD AI Premier League (AAIPL) — Adversarial Agent Orchestration
### *High-Performance LLM Optimization & Reinforcement Learning on AMD Instinct™ MI300X*

<p align="center">
  <img src="./assets/AMDAAIPL.png" alt="AMD AAIPL Banner" width="800">
</p>

[![Hardware: AMD MI300X](https://img.shields.io/badge/Hardware-AMD_Instinct_MI300X-ED1C24.svg?style=flat-square&logo=amd&logoColor=white)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![Framework: ROCm](https://img.shields.io/badge/Framework-ROCm_6.x-blue.svg?style=flat-square)](https://rocm.docs.amd.com/)
[![Event: IIT Delhi ScAI](https://img.shields.io/badge/Event-IIT_Delhi_ScAI-006a4e.svg?style=flat-square)](https://scai.iitd.ac.in/)
[![Optimization: Unsloth](https://img.shields.io/badge/Optimization-Unsloth-7b2cbf.svg?style=flat-square)](https://github.com/unslothai/unsloth)

---

## 📖 1. Project Abstract
The **AMD AI Premier League (AAIPL)** is a high-stakes competitive track within the AMD AI Reinforcement Learning Hackathon hosted at **IIT Delhi (Yardi School of AI)**. This project focuses on the development of dual-agent systems—a **Q-Agent** for adversarial question generation and an **A-Agent** for deductive reasoning—optimized for the **AMD Instinct™ MI300X** ecosystem. 

Unlike standard fine-tuning projects, this required **System-Level Optimization**: managing high-bandwidth memory (HBM3), implementing Reinforcement Learning (GRPO) to align agent behavior, and ensuring sub-9-second inference latencies for complex logical deduction.

---

## 🏗 2. System Architecture

The project follows a "Pitcher-Batter" adversarial loop. The goal is to maximize the "stumping" rate of the Q-Agent while maintaining 100% accuracy on the A-Agent.

### 2.1 Q-Agent: Adversarial Generator
The Q-Agent (Question Model) is fine-tuned to identify logical edge cases in four primary domains:
* **Logical Reasoning:** Syllogisms and multi-step deductive patterns.
* **Puzzles:** Linear/Circular seating arrangements with high constraint density.
* **Blood Relations:** Multi-generational family tree puzzles.
* **Alphanumeric Series:** Complex mixed-series logic.

### 2.2 A-Agent: The Deductive Solver
The A-Agent (Answer Model) uses a specialized "Chain-of-Thought" (CoT) prompt architecture, optimized via **PEFT (Parameter-Efficient Fine-Tuning)** to handle adversarial inputs without falling into hallucination traps.

---

## ⚡ 3. Deep-Tech Engineering & Optimizations

### 3.1 Hardware Acceleration: AMD Instinct™ MI300X
We optimized for the MI300X's **192GB HBM3** and **5.3 TB/s memory bandwidth**:
* **HSA_OVERRIDE_GFX_VERSION:** Fine-tuned the ROCm environment for GFX942 (MI300X) to ensure maximum compute utilization.
* **Batch Sizing:** Exploited the massive 192GB VRAM to run large-batch Reinforcement Learning updates, reducing training time by 40% compared to standard A100-80GB configurations.

### 3.2 Unsloth & ROCm Kernel Optimization
Integration of **Unsloth** provided a 2x speedup in fine-tuning by:
* Rewriting standard PyTorch kernels into highly efficient **Triton/ROCm** kernels.
* Implementing **4-bit/8-bit Quantization** via bitsandbytes-rocm to fit multiple agent replicas on a single GPU node.
* Reducing VRAM fragmentation during long-context puzzle generation.

### 3.3 Reinforcement Learning via GRPO
We implemented **Group Relative Policy Optimization (GRPO)** for agent alignment. GRPO is uniquely suited for this competition as it eliminates the need for a separate Critic model, saving significant VRAM:
* **Reward Function 1 (Correctness):** Rewards the A-Agent for matching the "Golden Answer."
* **Reward Function 2 (Complexity):** Rewards the Q-Agent when the opponent model fails, encouraging the generation of "hard but solvable" puzzles.
* **Reward Function 3 (Format):** Strict penalty for non-JSON compliant outputs.

---

## 🏏 4. Tournament Mechanics & Scoring

Matches are **1v1 knockout** format. Teams switch sides between innings.

### 4.1 Scoring Logic
$$\text{A-agent Score} = \left( \frac{\text{Questions correctly answered with expected format}}{N} \right) \times 100$$
$$\text{Q-agent Score} = \left( \frac{\text{Questions incorrectly answered by opponent}}{N} \right) \times 100$$

### 4.2 Competitive Constraints (SLA)
To mimic production environments, strict **Service Level Agreements (SLAs)** were enforced:
* **Q-Generation:** < 13.0 Seconds (includes prompt overhead + generation).
* **A-Inference:** < 9.0 Seconds (requires efficient KV caching and optimized decoding).

---

## 📋 5. Guidelines & JSON Formats

### Q-Agent Structure
```json
{
    "topic": "Logical Reasoning",
    "question": "If all A are B and some B are C...",
    "choices": ["A) Choice 1", "B) Choice 2", "C) Choice 3", "D) Choice 4"],
    "answer": "A",
    "explanation": "Brief reasoning path under 100 words."
}
