import json
import torch
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/AAIPL/hf_models/mistral_7b_base-qlora/"

# -----------------------------
# Load Model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = True


# -----------------------------
# Topic Guidance
# -----------------------------
def topic_guidance(topic: str):

    if topic == "Syllogisms":
        return """
Use formal logical structure:
All A are B.
All B are C.
Ask which conclusion must be true.
Ensure correct answer follows valid transitive logic.
Avoid real-world contradictions.
"""

    elif topic == "Seating Arrangements (Linear, Circular)":
        return """
Use 4-6 people with deterministic constraints.
Ensure one unique valid arrangement.
"""

    elif topic == "Family tree logic":
        return """
Use 3-step family relationships.
Clearly define gender and generations.
"""

    elif topic == "Mixed Series (Alphanumeric)":
        return """
Create a short alphanumeric pattern.
Ensure one clear progression rule.
"""

    return "Create a logical reasoning puzzle."


# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(topic: str):
    extra = topic_guidance(topic)

    return f"""
Generate ONE logical reasoning multiple choice question.

Topic: {topic}
Guidelines: {extra}

Rules:
- Exactly 4 options labeled A), B), C), D)
- Only ONE correct answer
- The answer must be a SINGLE LETTER: A, B, C, or D
- No ambiguity
- English only
- Output STRICTLY in JSON

JSON FORMAT:
{{
  "topic": "{topic}",
  "question": "...",
  "choices": [
    "A) ...",
    "B) ...",
    "C) ...",
    "D) ..."
  ],
  "answer": "A",
  "explanation": "Brief explanation under 100 words."
}}

Return ONLY the JSON object.
"""


# -----------------------------
# JSON Extraction (FIXED)
# -----------------------------
def extract_json(text: str):
    try:
        # Take last JSON object only
        last_open = text.rfind("{")
        json_text = text[last_open:]
        return json.loads(json_text)
    except:
        return None


# -----------------------------
# Validation
# -----------------------------
def validate_question(q):

    if not isinstance(q, dict):
        return False

    required_keys = ["topic", "question", "choices", "answer"]
    if not all(k in q for k in required_keys):
        return False

    if not isinstance(q["choices"], list) or len(q["choices"]) != 4:
        return False

    valid_labels = {"A)", "B)", "C)", "D)"}
    seen = set()

    for choice in q["choices"]:
        if not isinstance(choice, str):
            return False

        label = choice.strip()[:2]

        if label not in valid_labels:
            return False

        seen.add(label)

    if len(seen) != 4:
        return False

    if not isinstance(q["answer"], str):
        return False

    if q["answer"].upper() not in ["A", "B", "C", "D"]:
        return False

    return True


# -----------------------------
# Generate Single Question
# -----------------------------
def generate_question(topic: str, max_new_tokens=140):

    prompt = build_prompt(topic)

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    )

    input_ids = inputs.to(model.device)
    attention_mask = torch.ones(
    input_ids.shape,
    dtype=torch.long,
    device=model.device)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.3,        # More deterministic
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return extract_json(text)


# -----------------------------
# Retry Wrapper
# -----------------------------
def get_question(topic: str):

    for _ in range(3):
        q = generate_question(topic)
        if q and validate_question(q):
            return q

    return None


# -----------------------------
# Multi-Topic Generation
# -----------------------------
TOPICS = [
    "Syllogisms",
    "Seating Arrangements (Linear, Circular)",
    "Family tree logic",
    "Mixed Series (Alphanumeric)"
]


def is_duplicate(new_q, existing):
    for q in existing:
        if q["question"] == new_q["question"]:
            return True
    return False


def generate_multiple_questions(num_questions=20):

    results = []
    topic_index = 0
    attempts = 0
    max_attempts = num_questions * 6

    total_start = time.time()

    while len(results) < num_questions and attempts < max_attempts:

        topic = TOPICS[topic_index % len(TOPICS)]

        start_time = time.time()

        q = get_question(topic)

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        if q and not is_duplicate(q, results):
            results.append(q)
            print(f"Generated {len(results)}/{num_questions} | Topic: {topic} | Time: {duration}s")

        topic_index += 1
        attempts += 1

    total_end = time.time()
    print("\nTotal Generation Time:", round(total_end - total_start, 2), "seconds")

    if len(results) < num_questions:
        print("Warning: Could not generate enough valid questions.")

    return results


# -----------------------------
# Save Output
# -----------------------------
import os

def save_questions(questions, filename="outputs/questions.json"):
    os.makedirs("outputs", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(questions, f, indent=4)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    questions = generate_multiple_questions(20)
    save_questions(questions)
    print("Total questions generated:", len(questions))
