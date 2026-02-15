#!/usr/bin/python3

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from answer_model import AAgent


class AnsweringAgent:
    """Agent responsible for answering multiple-choice questions."""

    def __init__(self, use_detailed_prompt: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.use_detailed_prompt = use_detailed_prompt

    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompt for answering a question."""
        
        # System prompt
        if self.use_detailed_prompt:
            sys_prompt = (
                "You are an expert in solving multiple-choice questions. "
                "Think step-by-step and provide clear reasoning before selecting the answer."
            )
        else:
            sys_prompt = "You are a helpful expert assistant for answering questions."

        # Format choices
        choices_text = " ".join(question_data.get("choices", []))

        # User prompt
        user_prompt = (
            "Answer the following multiple-choice question:\n\n"
            f"Question: {question_data.get('question', '')}\n"
            f"Choices: {choices_text}\n\n"
            "Provide your answer in the following JSON format:\n"
            "{\n"
            '  "answer": "[A, B, C, or D]",\n'
            '  "reasoning": "[Brief explanation under 100 words]"\n'
            "}"
        )

        return user_prompt, sys_prompt

    def answer_question(
        self,
        question_data: Dict[str, Any] | List[Dict[str, Any]],
        **gen_kwargs,
    ) -> Tuple[str | List[str], int | None, float | None]:
        """Generate answer(s) for question(s)."""
        
        if isinstance(question_data, list):
            prompts = []
            sys_prompt = None
            for q in question_data:
                p, sp = self.build_prompt(q)
                prompts.append(p)
                sys_prompt = sp
        else:
            prompts, sys_prompt = self.build_prompt(question_data)

        response, token_len, generation_time = self.agent.generate_response(
            prompts, sys_prompt, **gen_kwargs
        )
        return response, token_len, generation_time

    def answer_batch(
        self,
        questions: List[Dict[str, Any]],
        batch_size: int = 5,
        **gen_kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        """Answer questions in batches."""
        
        answers = []
        token_lengths = []
        generation_times = []

        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Answering questions")

        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch, **gen_kwargs)

            if isinstance(batch_answers, list):
                answers.extend(batch_answers)
            else:
                answers.append(batch_answers)

            token_lengths.append(tl)
            generation_times.append(gt)
            pbar.update(1)

        pbar.close()
        return answers, token_lengths, generation_times

    def filter_answers(
        self, answers: List[str | Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Filter and validate answers."""
        
        def is_valid(a: Dict[str, str]) -> bool:
            if not isinstance(a, dict):
                return False

            required_keys = ["answer"]
            if "answer" not in a:
                return False

            answer = a["answer"].strip().upper()
            if len(answer) != 1 or answer not in "ABCD":
                return False

            return True

        filtered = []
        for a in answers:
            if isinstance(a, dict):
                if is_valid(a):
                    filtered.append(a)
            elif isinstance(a, str):
                try:
                    a_dict = json.loads(a)
                    if is_valid(a_dict):
                        filtered.append(a_dict)
                except json.JSONDecodeError:
                    continue

        return filtered

    def save_answers(
        self, answers: List[Dict | str], output_path: str | Path
    ) -> None:
        """Save answers to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(answers, f, indent=2)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Answer MCQ questions")
    parser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json")
    parser.add_argument("--output_file", type=str, default="outputs/answers.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load questions
    with open(args.input_file, "r") as f:
        questions = json.load(f)

    # Load generation config
    gen_kwargs = {"tgps_show": True}
    try:
        with open("agen.yaml", "r") as f:
            gen_kwargs.update(yaml.safe_load(f))
    except FileNotFoundError:
        pass

    # Generate answers
    agent = AnsweringAgent()
    answers, token_lengths, generation_times = agent.answer_batch(
        questions, batch_size=args.batch_size, **gen_kwargs
    )

    print(f"Generated {len(answers)} answers")

    if args.verbose:
        for i, (q, a) in enumerate(zip(questions, answers)):
            print(f"\n[{i+1}] Question: {q.get('question', 'N/A')}")
            print(f"    Expected: {q.get('answer', 'N/A')}")
            print(f"    Answer: {a}")

        if gen_kwargs.get("tgps_show"):
            total_time = sum(t for t in generation_times if t)
            total_tokens = sum(t for t in token_lengths if t)
            print(f"\nTotal time: {total_time:.3f}s")
            print(f"Total tokens: {total_tokens}")
            print(f"TGPS: {total_tokens/total_time:.3f}")

    # Filter and save
    filtered_answers = agent.filter_answers(answers)
    agent.save_answers(answers, args.output_file)
    agent.save_answers(
        filtered_answers,
        args.output_file.replace("answers.json", "filtered_answers.json"),
    )

    print(f"Saved {len(filtered_answers)} valid answers")