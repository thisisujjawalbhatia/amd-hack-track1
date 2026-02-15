#!/usr/bin/python3
"""
Build fine-tuning datasets for Q-Agent (Mistral) and A-Agent (Qwen2.5)
from the seed data in assets/.

Usage:
    python -m agents.build_finetune_data

Outputs:
    agents/data/q_train.jsonl   — question generation training data
    agents/data/a_train.jsonl   — answer generation training data
"""

import json
import random
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR = Path(__file__).parent / "data"

# === System prompts (must match question_agent.py / answer_agent.py) ===

Q_SYS_PROMPT = (
    "You are an **expert-level examiner** with deep expertise in designing "
    "**highly challenging and conceptually rigorous multiple-choice questions (MCQs)** "
    "for the **Quantitative Aptitude and Analytical Reasoning** sections of top-tier "
    "competitive exams.\n"
    "Think step by step to generate the question and solve the same, but only output "
    "the final answer. Do not show your thinking process.\n"
    "**Please DO NOT reveal the solution steps or any intermediate reasoning.**"
)

A_SYS_PROMPT = (
    "You are an expert answer agent specializing in solving multiple-choice questions "
    "(MCQs) that test quantitative aptitude skills, as seen in top-tier competitive exams. "
    "You have a deep understanding of logical reasoning, puzzles, and analytical "
    "problem-solving under exam conditions. For each question, think step by step using "
    "a clear chain-of-thought approach. Break down the problem, analyze all options, "
    "eliminate distractors, and then confidently select the correct answer. "
    "Always explain your reasoning before finalizing your choice."
)

# === Topic mapping (maps subtopic -> parent category) ===

TOPIC_MAP = {
    "Syllogisms": "Logical Reasoning",
    "Seating Arrangements (Linear, Circular)": "Puzzles",
    "Family tree logic": "Blood Relations and Family Tree",
    "Mixed Series (Alphanumeric)": "Series and Patterns",
}


def build_q_user_prompt(topic_full: str, correct_option: str) -> str:
    """Build a user prompt matching question_agent.py's tmpl format."""
    subtopic = topic_full.split("/")[-1]
    distractors = ", ".join(
        [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
    )
    tmpl = (
        "Generate an EXTREMELY DIFFICULT MCQ on topic: {0}.\n\n"
        "**CRITICAL REQUIREMENTS:**\n"
        '1.  **Topic Alignment**: The "question" must be strictly relevant to the topic: {1}.\n'
        "2.  **Question Quality**: The question must be EXTREMELY DIFFICULT, clear, and test deep conceptual understanding. Avoid trivial or ambiguous questions.\n"
        '3.  **Choices (4 total)**: Generate exactly FOUR multiple-choice options, labeled "A)", "B)", "C)", and "D)".\n'
        "4.  **Single Correct Answer**: Ensure that option {2} is only factually correct.\n"
        "5.  **Plausible Distractors**: While option {3} are three incorrect UNIQUE choices which are highly plausible and common misconceptions related to the topic, designed to mislead someone without expert knowledge.\n"
        '6.  **Answer Key**: The "answer" field in the JSON should be ONLY the letter {4}.\n'
        '7.  **Explanation**: The "explanation" field provides a concise (under 100 words) and clear justification for why the correct answer is correct.\n\n'
        "RESPONSE FORMAT: Strictly generate a valid JSON object ensuring proper syntax and structure as shown below.\n\n"
        "EXAMPLE: {5}\n"
        "{{\n"
        '  "topic": "{6}",\n'
        '  "question": "...",\n'
        '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
        '  "answer": "{7}",\n'
        '  "explanation": "Provide a brief explanation why {8} is correct within 100 words."\n'
        "}}"
    )
    return tmpl.format(
        topic_full,
        topic_full,
        correct_option,
        distractors,
        correct_option,
        topic_full,
        subtopic,
        correct_option,
        correct_option,
    )


def build_a_user_prompt(question: str, choices: list) -> str:
    """Build a user prompt matching answer_agent.py's tmpl format."""
    formatted_choices = " ".join(c.strip() for c in choices)
    tmpl = (
        "INSTRUCTIONS FOR ANSWERING:\n"
        "1. Carefully read and understand what is being asked.\n"
        "2. Consider why each choice might be correct or incorrect.\n"
        "3. There is only **ONE OPTION** correct.\n"
        "4. Provide reasoning within 100 words\n\n"
        "Now answer the following question:\n"
        "Question: {}\n"
        "Choices: {}\n\n"
        "RESPONSE FORMAT: Strictly generate a valid JSON object as shown below:\n"
        "{{\n"
        '    "answer": "One of the letter from [A, B, C, D]",\n'
        '    "reasoning": "Brief explanation within 100 words"\n'
        "}}"
    )
    return tmpl.format(question, formatted_choices)


def remap_answer(example: dict, target_option: str) -> dict:
    """
    Remap a question example so that the correct answer is at target_option.
    This augments the dataset by varying correct-option position.
    """
    original_answer = example.get("expected_answer", example.get("answer", "A"))
    choices = list(example["choices"])

    if len(choices) != 4:
        return None

    option_letters = ["A", "B", "C", "D"]
    orig_idx = option_letters.index(original_answer)
    target_idx = option_letters.index(target_option)

    if orig_idx == target_idx:
        return example.copy()

    # Swap the choices
    new_choices = list(choices)
    new_choices[orig_idx], new_choices[target_idx] = (
        new_choices[target_idx],
        new_choices[orig_idx],
    )

    # Fix the labels
    for i, c in enumerate(new_choices):
        c_stripped = c.strip()
        if len(c_stripped) > 2 and c_stripped[1] == ")":
            c_stripped = c_stripped[2:].strip()
        new_choices[i] = f"{option_letters[i]}) {c_stripped}"

    result = dict(example)
    result["choices"] = new_choices
    if "expected_answer" in result:
        result["expected_answer"] = target_option
    if "answer" in result:
        result["answer"] = target_option
    return result


def build_q_dataset():
    """Build question-generation fine-tuning dataset."""
    examples_path = ASSETS_DIR / "topics_example.json"
    sample_q_path = ASSETS_DIR / "sample_question.json"

    with open(examples_path, "r", encoding="utf-8") as f:
        topic_examples = json.load(f)

    with open(sample_q_path, "r", encoding="utf-8") as f:
        sample_questions = json.load(f)

    training_data = []

    # From topics_example.json
    for subtopic, examples in topic_examples.items():
        parent = TOPIC_MAP.get(subtopic, subtopic)
        topic_full = f"{parent}/{subtopic}"

        for example in examples:
            answer = example.get("expected_answer", example.get("answer", "A"))

            # Create multiple variants with different correct-option positions
            for target_opt in ["A", "B", "C", "D"]:
                remapped = remap_answer(example, target_opt)
                if remapped is None:
                    continue

                user_prompt = build_q_user_prompt(topic_full, target_opt)

                # Build the ideal JSON response
                response_obj = {
                    "topic": subtopic,
                    "question": remapped["question"],
                    "choices": remapped["choices"],
                    "answer": target_opt,
                    "explanation": remapped.get("explanation", ""),
                }

                training_data.append(
                    {
                        "messages": [
                            {"role": "system", "content": Q_SYS_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {
                                "role": "assistant",
                                "content": json.dumps(response_obj, ensure_ascii=False),
                            },
                        ]
                    }
                )

    # From sample_question.json
    for sq in sample_questions:
        subtopic = sq.get("topic", "")
        parent = TOPIC_MAP.get(subtopic, subtopic)
        topic_full = f"{parent}/{subtopic}" if parent != subtopic else subtopic
        answer = sq.get("expected_answer", sq.get("answer", "A"))

        for target_opt in ["A", "B", "C", "D"]:
            remapped = remap_answer(sq, target_opt)
            if remapped is None:
                continue

            user_prompt = build_q_user_prompt(topic_full, target_opt)
            response_obj = {
                "topic": subtopic,
                "question": remapped["question"],
                "choices": remapped["choices"],
                "answer": target_opt,
                "explanation": remapped.get("explanation", ""),
            }

            training_data.append(
                {
                    "messages": [
                        {"role": "system", "content": Q_SYS_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "assistant",
                            "content": json.dumps(response_obj, ensure_ascii=False),
                        },
                    ]
                }
            )

    return training_data


def build_a_dataset():
    """Build answer-generation fine-tuning dataset."""
    examples_path = ASSETS_DIR / "topics_example.json"
    sample_q_path = ASSETS_DIR / "sample_question.json"
    sample_a_path = ASSETS_DIR / "sample_answer.json"

    with open(examples_path, "r", encoding="utf-8") as f:
        topic_examples = json.load(f)

    with open(sample_q_path, "r", encoding="utf-8") as f:
        sample_questions = json.load(f)

    with open(sample_a_path, "r", encoding="utf-8") as f:
        sample_answers = json.load(f)

    training_data = []

    # From topics_example.json
    for subtopic, examples in topic_examples.items():
        for example in examples:
            answer = example.get("expected_answer", example.get("answer", "A"))
            explanation = example.get("explanation", "")

            user_prompt = build_a_user_prompt(example["question"], example["choices"])
            response_obj = {
                "answer": answer,
                "reasoning": explanation,
            }

            training_data.append(
                {
                    "messages": [
                        {"role": "system", "content": A_SYS_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "assistant",
                            "content": json.dumps(response_obj, ensure_ascii=False),
                        },
                    ]
                }
            )

    # From sample_question.json + sample_answer.json
    for sq, sa in zip(sample_questions, sample_answers):
        answer = sa.get("answer", sq.get("expected_answer", "A"))
        reasoning = sa.get("reasoning", sq.get("explanation", ""))

        user_prompt = build_a_user_prompt(sq["question"], sq["choices"])
        response_obj = {
            "answer": answer,
            "reasoning": reasoning,
        }

        training_data.append(
            {
                "messages": [
                    {"role": "system", "content": A_SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": json.dumps(response_obj, ensure_ascii=False),
                    },
                ]
            }
        )

    return training_data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build Q-agent dataset
    q_data = build_q_dataset()
    q_path = OUTPUT_DIR / "q_train.jsonl"
    with open(q_path, "w", encoding="utf-8") as f:
        for item in q_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Q-Agent training data: {len(q_data)} examples -> {q_path}")

    # Build A-agent dataset
    a_data = build_a_dataset()
    a_path = OUTPUT_DIR / "a_train.jsonl"
    with open(a_path, "w", encoding="utf-8") as f:
        for item in a_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"A-Agent training data: {len(a_data)} examples -> {a_path}")


if __name__ == "__main__":
    main()
