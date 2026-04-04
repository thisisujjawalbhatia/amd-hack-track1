# 🏏 AMD AI Premier League (AAIPL) — Adversarial Agent Orchestration
### *Advanced Reinforcement Learning & LLM Optimization on AMD Instinct™ MI300X*

[![Hardware: AMD MI300X](https://img.shields.io/badge/Hardware-AMD_Instinct_MI300X-ED1C24.svg?style=flat-square&logo=amd&logoColor=white)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![Framework: ROCm](https://img.shields.io/badge/Framework-ROCm_6.x-blue.svg?style=flat-square)](https://rocm.docs.amd.com/)
[![Event: IIT Delhi ScAI](https://img.shields.io/badge/Event-IIT_Delhi_ScAI-006a4e.svg?style=flat-square)](https://scai.iitd.ac.in/)
[![Optimization: Unsloth](https://img.shields.io/badge/Optimization-Unsloth-7b2cbf.svg?style=flat-square)](https://github.com/unslothai/unsloth)

---

## 📖 1. Project Abstract
The **AMD AI Premier League (AAIPL)** is a high-stakes competitive track within the AMD AI Reinforcement Learning Hackathon hosted at **IIT Delhi (Yardi School of AI)**. This project focuses on the development of dual-agent systems—a **Q-Agent** for adversarial question generation and an **A-Agent** for deductive reasoning—optimized for the **AMD ROCm** ecosystem. 

In this "Cricket-style" tournament, models compete in head-to-head innings where survival depends on a model's ability to stump opponents with complex puzzles while accurately decoding adversarial inputs under strict latency constraints.

---

## 🏗 2. The Task: Dual-Agent Architecture

### 2.1 The Q-Agent (Questioner)
The Q-Agent acts as the "Pitcher." Its primary objective is to generate $N$ puzzle-based questions based on specific domains that are difficult for other LLMs to solve but remain factually grounded.

- **Implementation:** Custom model in `agents/question_model.py`.
- **Target Topics:** 1.  **Logical Reasoning:** Syllogisms and complex deductive logic.
    2.  **Puzzles:** Linear and Circular seating arrangements.
    3.  **Blood Relations:** Multi-generational family tree puzzles.
    4.  **Alphanumeric Series:** Mixed-logic sequence challenges.
- **Output Format:** Must strictly follow the schema in `assets/sample_question.json`.

### 2.2 The A-Agent (Answerer)
The A-Agent is the "Batter." It must parse the adversarial questions provided by the opponent's Q-Agent and provide the correct choice letter along with logical reasoning.

- **Implementation:** Custom model in `agents/answer_model.py`.
- **Performance Metric:** Accuracy and reasoning fidelity against unseen adversarial logic.
- **Output Format:** Must strictly follow the schema in `assets/sample_answer.json`.

---

## 🏏 3. Tournament Overview & Scoring

Matches are conducted in a **1v1 knockout** format. Each match consists of two innings where teams switch roles.

### 3.1 Scoring Formulas
The final winner is determined by the summation of the Q-Agent score and the A-Agent score across both innings.

$$\text{A-agent Score} = \left( \frac{\text{Questions correctly answered with expected format}}{N} \right) \times 100$$

$$\text{Q-agent Score} = \left( \frac{\text{Questions incorrectly answered by opponent}}{N} \right) \times 100$$

### 3.2 Elimination & Disqualification Rules
- **Seeding Round:** Before the knockouts, A-Agents are tested against a hidden set of questions.
- **Format Compliance:** Your Q-Agent **must** generate at least **50% valid, format-correct questions** (based on $N$). Failure to meet this threshold results in automatic disqualification.
- **Tie-Breakers:** In case of a tie, closed benchmark questions will be used to evaluate the answer agents (A-agent) and rank the teams.

---

## ⚡ 4. Technical Implementation & Hardware Optimization

### 4.1 AMD Instinct™ MI300X Integration
The project is built to leverage the **192GB HBM3** capacity and massive throughput of the MI300X.
- **ROCm 6.x Kernels:** Using optimized PyTorch builds for AMD.
- **VRAM Utilization:** Exploiting large memory footprints to train with higher batch sizes for Reinforcement Learning stability.

### 4.2 Training Strategy: GRPO & Unsloth
- **Reinforcement Learning:** Utilizing **Group Relative Policy Optimization (GRPO)** to reward models for "Hard-but-Solvable" question generation.
- **Unsloth Optimization:** Implementing Unsloth kernels to achieve 2x faster fine-tuning on the MI300X.

---

## 📋 5. Guidelines & Constraints

### 5.1 Format Requirements
**Q-Agent JSON Structure:**
```json
{
    "topic": "<Topic Name>",
    "question": "<full question text>",
    "choices": [
        "A) <choice A>",
        "B) <choice B>",
        "C) <choice C>",
        "D) <choice D>"
    ],
    "answer": "<choice letter only>",
    "explanation": "<brief explanation within 100 words>"
}
