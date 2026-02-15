# Mistral-7B-Instruct-v0.3 for Question Generation
import json
import re
import time
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)


class QAgent(object):
    def __init__(self, **kwargs):
        base_dir = Path(__file__).parent.parent / "hf_models"
        model_path = str(base_dir / "mistral_7b_base")
        lora_path = base_dir / "mistral_7b_base-qlora"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model, then apply LoRA if available
        if lora_path.exists() and (lora_path / "adapter_config.json").exists():
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base_model, str(lora_path))
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
            )
        self.model.eval()

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text that may contain markdown or extra content."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0).strip()
        return text.strip()

    def _normalize_choices(self, data: dict) -> dict:
        """Normalize choices to list format ['A) ...', 'B) ...', 'C) ...', 'D) ...'].
        Handles both object format {"A": "...", ...} and list format."""
        choices = data.get("choices")
        if choices is None:
            return data

        if isinstance(choices, dict):
            # Convert {"A": "text", "B": "text", ...} -> ["A) text", "B) text", ...]
            normalized = []
            for letter in ["A", "B", "C", "D"]:
                val = choices.get(letter, choices.get(letter.lower(), ""))
                # Strip existing prefix like "A) " if present
                val = re.sub(r"^[A-Da-d]\)\s*", "", str(val).strip())
                normalized.append(f"{letter}) {val}")
            data["choices"] = normalized

        elif isinstance(choices, list):
            # Ensure each choice has proper "X) " prefix
            normalized = []
            for i, c in enumerate(choices):
                c_str = str(c).strip()
                letter = chr(65 + i)  # A, B, C, D
                if not re.match(r"^[A-Da-d]\)", c_str):
                    c_str = f"{letter}) {c_str}"
                normalized.append(c_str)
            data["choices"] = normalized

        return data

    def _validate_question(self, text: str) -> bool:
        """Validate that generated text is a properly formatted question JSON."""
        try:
            data = json.loads(text)
            if not all(k in data for k in ["topic", "question", "choices", "answer"]):
                return False
            # Normalize choices before validation
            data = self._normalize_choices(data)
            if not isinstance(data["choices"], list) or len(data["choices"]) != 4:
                return False
            ans = data["answer"].strip().upper()
            if len(ans) != 1 or ans not in "ABCD":
                return False
            return True
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            return False

    # Hard limits enforced regardless of external config
    _MAX_NEW_TOKENS = 350
    _TEMPERATURE = 0.3
    _TOP_P = 0.9

    def _build_gen_params(self, **kwargs) -> dict:
        """Build generation parameters with enforced optimal values."""
        # Cap max_new_tokens to keep generation under 13s
        max_tokens = min(kwargs.get("max_new_tokens", self._MAX_NEW_TOKENS), self._MAX_NEW_TOKENS)
        gen_params = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": self._TEMPERATURE,
            "top_p": self._TOP_P,
        }
        if "repetition_penalty" in kwargs:
            gen_params["repetition_penalty"] = kwargs["repetition_penalty"]
        return gen_params

    def _prepare_inputs(self, message: List[str], system_prompt: str):
        """Tokenize messages using the chat template."""
        texts = []
        for msg in message:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        return model_inputs

    def _decode_batch(self, model_inputs, generated_ids, track_tokens: bool):
        """Decode generated token ids into text, extracting and normalizing JSON."""
        batch_outs = []
        token_len = 0
        for input_ids, gen_seq in zip(model_inputs.input_ids, generated_ids):
            output_ids = gen_seq[len(input_ids) :]
            if track_tokens:
                token_len += len(output_ids)
            content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            content = self._extract_json(content)
            # Normalize choices format (dict -> list) if needed
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "choices" in data:
                    data = self._normalize_choices(data)
                    content = json.dumps(data, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass
            batch_outs.append(content)
        return batch_outs, token_len

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> Tuple:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]

        model_inputs = self._prepare_inputs(message, system_prompt)
        gen_params = self._build_gen_params(**kwargs)
        tgps_show_var = kwargs.get("tgps_show", False)

        if tgps_show_var:
            start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_params)

        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs, token_len = self._decode_batch(
            model_inputs, generated_ids, tgps_show_var
        )

        # Validation + single retry for items that failed JSON validation
        retry_indices = [
            i for i, out in enumerate(batch_outs) if not self._validate_question(out)
        ]
        if retry_indices:
            retry_msgs = [message[i] for i in retry_indices]
            retry_inputs = self._prepare_inputs(retry_msgs, system_prompt)
            with torch.no_grad():
                retry_ids = self.model.generate(**retry_inputs, **gen_params)
            retry_outs, extra_tokens = self._decode_batch(
                retry_inputs, retry_ids, tgps_show_var
            )
            for j, idx in enumerate(retry_indices):
                batch_outs[idx] = retry_outs[j]
            if tgps_show_var:
                token_len += extra_tokens
                generation_time = time.time() - start_time

        result = batch_outs[0] if len(batch_outs) == 1 else batch_outs

        if tgps_show_var:
            return result, token_len, generation_time
        return result, None, None


if __name__ == "__main__":
    model = QAgent()
    response, tl, tm = model.generate_response(
        "Generate an extremely difficult MCQ on topic: Logical Reasoning/Syllogisms.",
        system_prompt="You are an expert examiner.",
        tgps_show=True,
        max_new_tokens=350,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
    )
    print("Response:", response)
    if tl and tm:
        print(f"Tokens: {tl}, Time: {tm:.2f}s, TGPS: {tl/tm:.2f}")
