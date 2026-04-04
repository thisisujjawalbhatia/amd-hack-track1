# 🏏 AMD AI Premier League (AAIPL) — High-Performance Agent Engineering
### *Adversarial Reasoning & LLM Orchestration on AMD Instinct™ Hardware*

<p align="center">
  <img src="./assets/AMDAAIPL.png" alt="AMD AAIPL Banner" width="800">
</p>

---

## 📋 The Mission
This project involves building a dual-agent system designed for adversarial competition:
* **Question Agent (Q-Agent):** Generates $N$ puzzle-based questions based on specified topics.
    * **Logic:** `question_model.py`.
    * **Output:** Must strictly follow the `sample_question.json` format.
* **Answer Agent (A-Agent):** Solves questions posed by opposing Q-agents.
    * **Logic:** `answer_model.py`.
    * **Output:** Must strictly follow the `sample_answer.json` format.

---

## 🔌 Workstation Operations
* **Initiation:** Access the workstation at `dev.amd-ai-academy.com` using the provided Team ID and Password.
* **Model Protocols:** Only models located in `/root/.cache/huggingface/hub` are authorized.
    * These models are read-only; you must **copy** them into `AAIPL/hf_models` for editing.
    * Attempting to modify the original source folder results in immediate disqualification.
* **Code Management:** Push all code (excluding `hf_models`) to GitHub using the `git.sh` script.

---

## 🏟️ Tournament Dynamics
The competition follows a **1v1 knockout** format structured like a cricket match where teams switch sides.

* **Seeding Round:** An initial elimination stage tests A-agents against hidden questions to determine eligibility for the knockout phase.
* **1st Inning:** Team-A (Q-Agent) pitches questions $\rightarrow$ Team-B (A-Agent) answers.
* **2nd Inning:** Team-B (Q-Agent) pitches questions $\rightarrow$ Team-A (A-Agent) answers.
* **Final Scoring:** The winner is determined by the total points from both pitching (Q-Agent) and batting (A-Agent) phases.

---

## 📄 Data Protocols & Formats

### Q-Agent Generation (JSON)
```json
{
    "topic": "<Topic of the Question>",
    "question": "<full question text>",
    "choices": [
        "A) <choice A text>",
        "B) <choice B text>",
        "C) <choice C text>",
        "D) <choice D text>"
    ],
    "answer": "<correct choice letter only>",
    "explanation": "brief explanation within 100 words"
}
```
*The Topic, Question, Choices, and Answer will be verified for absolute correctness.*

### A-Agent Solving (JSON)
```json
{
    "answer": "<correct choice letter only>",
    "reasoning": "brief reasoning within 100 words"
}
```
*The Answer key is directly compared against the opponent's Q-agent answer for scoring.*

---

## 🚫 Engagement Rules & Restrictions
* **No RAG:** Retrieval Augmented Generation techniques are strictly forbidden.
* **Zero Adversarial Poisoning:** Approaches designed to make A-agents hallucinate lead to disqualification.
* **Language & Model Integrity:** Only English is permitted, and only provided models may be used.
* **SLA Constraints:**
    * **Q-Generation:** Must be under **13 seconds** per question.
    * **A-Inference:** Must be under **9 seconds** per answer.
* **Token Limits:** Adhere strictly to the `max_tokens` settings in `agen.yaml` and `qgen.yaml`.

---

## 🗃️ Directory Architecture
```plaintext
.
├── agents
│   ├── question_model.py      # Core Q-agent logic
│   ├── question_agent.py     # Inference wrapper for Q
│   ├── answer_model.py        # Core A-agent logic
│   └── answer_agent.py       # Inference wrapper for A
├── assets
│   ├── topics.json           # Target domains for generation
│   ├── sample_question.json   # Q-format specification
│   └── sample_answer.json     # A-format specification
├── utils
│   └── build_prompt.py       # Prompt-tuning scripts
├── qgen.yaml / agen.yaml      # Generation parameters
└── tutorial.ipynb            # Unsloth & Synthetic Data Guide
```

---

## 🥇 Scoring & Evaluation

Performance is measured by competitive accuracy:

$$\text{A-agent Score} = \dfrac{\# \text{ questions correctly answered}}{N} \times 100$$
$$\text{Q-agent Score} = \dfrac{\# \text{ questions incorrectly answered by opponent}}{N} \times 100$$

> **Disqualification Warning:** Q-agents must maintain a minimum **50% format-correctness rate** to remain in the tournament. Tie-breakers are resolved using closed benchmark question sets.

---

## 🕹️ Getting Started
To verify your system locally before submission:

**Question Generation:**
```bash
python -m agents.question_agent --output_file "outputs/questions.json" --num_questions 20 --verbose
```

**Answer Deduction:**
```bash
python -m agents.answer_agent --input_file "outputs/filtered_questions.json" --output_file "outputs/answers.json" --verbose
```

---
<p align="center">
  <i>"Hardware-optimized adversarial reasoning at the edge of innovation."</i>
</p>
