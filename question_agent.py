#!/usr/bin/python3

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from question_model import QAgent


class QuestioningAgent:
    """Agent responsible for generating multiple-choice questions."""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_prompt(
        self,
        topic: str,
        use_advanced_system: bool = True,
        include_samples: bool = True,
        samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        """Build prompt for question generation."""
        
        # System prompt
        if use_advanced_system:
            sys_prompt = (
                "You are an expert-level examiner designing highly challenging MCQs "
                "for quantitative aptitude and analytical reasoning in competitive exams. "
                "Generate only the final JSON output without showing your thinking process."
            )
        else:
            sys_prompt = "You are an examiner creating difficult multiple-choice questions."

        # Random correct answer to avoid bias
        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join([o for o in ["A", "B", "C", "D"] if o != correct_option])

        # Build main prompt
        user_prompt = (
            f"Generate an EXTREMELY DIFFICULT MCQ on topic: {topic}\n\n"
            f"**CRITICAL REQUIREMENTS:**\n"
            f"1. **Topic Alignment**: Question must be strictly relevant to {topic}.\n"
            f"2. **Question Quality**: Must be extremely difficult and test deep understanding.\n"
            f"3. **Choices**: Generate exactly FOUR options labeled A), B), C), D).\n"
            f"4. **Correct Answer**: Only option {correct_option} is factually correct.\n"
            f"5. **Distractors**: Options {distractors} are plausible but incorrect.\n"
            f"6. **Answer Key**: The answer field should contain ONLY the letter {correct_option}.\n"
            f"7. **Explanation**: Provide brief (under 100 words) justification.\n\n"
            f"RESPONSE FORMAT: Return ONLY a valid JSON object:\n"
            f"{{\n"
            f'  "topic": "{topic}",\n'
            f'  "question": "...",\n'
            f'  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            f'  "answer": "{correct_option}",\n'
            f'  "explanation": "..."\n'
            f"}}"
        )

        return user_prompt, sys_prompt

    def generate_question(
        self,
        topic: str | List[str],
        use_advanced_system: bool = True,
        include_samples: bool = True,
        samples: Dict[str, List[Dict[str, str]]] | None = None,
        **gen_kwargs,
    ) -> Tuple[str | List[str], int | None, float | None]:
        """Generate a question for given topic(s)."""
        
        if isinstance(topic, list):
            prompts = []
            sys_prompt = None
            for t in topic:
                p, sp = self.build_prompt(t, use_advanced_system, include_samples, samples)
                prompts.append(p)
                sys_prompt = sp
        else:
            prompts, sys_prompt = self.build_prompt(
                topic, use_advanced_system, include_samples, samples
            )

        response, token_len, generation_time = self.agent.generate_response(
            prompts, sys_prompt, **gen_kwargs
        )
        return response, token_len, generation_time

    def generate_batch(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        use_advanced_system: bool = True,
        include_samples: bool = True,
        samples: Dict[str, List[Dict[str, str]]] | None = None,
        **gen_kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        """Generate questions in batches."""
        
        # Expand topics to match num_questions
        all_topics = [
            (topic, subtopic)
            for topic, subtopics in topics.items()
            for subtopic in subtopics
        ]
        selected_topics = random.choices(all_topics, k=num_questions)

        questions = []
        token_lengths = []
        generation_times = []

        total_batches = (len(selected_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Generating questions")

        for i in range(0, len(selected_topics), batch_size):
            batch_topics = selected_topics[i : i + batch_size]
            batch_questions, tl, gt = self.generate_question(
                batch_topics, use_advanced_system, include_samples, samples, **gen_kwargs
            )

            if isinstance(batch_questions, list):
                questions.extend(batch_questions)
            else:
                questions.append(batch_questions)

            token_lengths.append(tl)
            generation_times.append(gt)
            pbar.update(1)

        pbar.close()
        return questions, token_lengths, generation_times

    def filter_questions(
        self, questions: List[str | Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and validate questions to ensure correct format."""
        
        def is_valid(q: Dict[str, Any]) -> bool:
            required_keys = ["topic", "question", "choices", "answer"]
            if not all(key in q for key in required_keys):
                return False

            if not isinstance(q["choices"], list) or len(q["choices"]) != 4:
                return False

            choices_valid = all(
                isinstance(c, str) and len(c) > 2 and c[0].upper() in "ABCD"
                for c in q["choices"]
            )
            if not choices_valid:
                return False

            if not isinstance(q["answer"], str) or q["answer"].upper() not in "ABCD":
                return False

            return True

        filtered = []
        for q in questions:
            if isinstance(q, dict) and is_valid(q):
                filtered.append(q)
            elif isinstance(q, str):
                try:
                    q_dict = json.loads(q)
                    if is_valid(q_dict):
                        filtered.append(q_dict)
                except json.JSONDecodeError:
                    continue

        return filtered

    def save_questions(
        self, questions: List[Dict | str], output_path: str | Path
    ) -> None:
        """Save questions to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(questions, f, indent=2)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate MCQ questions")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="outputs/questions.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load topics
    with open("assets/topics.json", "r") as f:
        topics = json.load(f)

    # Load generation config
    gen_kwargs = {"tgps_show": True}
    try:
        with open("qgen.yaml", "r") as f:
            gen_kwargs.update(yaml.safe_load(f))
    except FileNotFoundError:
        pass

    # Generate questions
    agent = QuestioningAgent()
    questions, token_lengths, generation_times = agent.generate_batch(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        **gen_kwargs,
    )

    print(f"Generated {len(questions)} questions")

    if args.verbose:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}] {q}")

        if gen_kwargs.get("tgps_show"):
            total_time = sum(t for t in generation_times if t)
            total_tokens = sum(t for t in token_lengths if t)
            print(f"\nTotal time: {total_time:.3f}s")
            print(f"Total tokens: {total_tokens}")
            print(f"TGPS: {total_tokens/total_time:.3f}")

    # Filter and save
    filtered_questions = agent.filter_questions(questions)
    agent.save_questions(questions, args.output_file)
    agent.save_questions(
        filtered_questions,
        args.output_file.replace("questions.json", "filtered_questions.json"),
    )

    print(f"Saved {len(filtered_questions)} valid questions")