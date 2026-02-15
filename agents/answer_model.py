# Qwen2.5-14B-Instruct for Answer Generation
import json
import re
import time
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.random.manual_seed(0)


class AAgent(object):
    def __init__(self, **kwargs):
        base_dir = Path(__file__).parent.parent / "hf_models"
        model_path = str(base_dir / "qwen_14b_base")
        lora_path = base_dir / "qwen_14b_base-qlora"

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

    def _validate_answer(self, text: str) -> bool:
        """Validate that generated text is a properly formatted answer JSON."""
        try:
            data = json.loads(text)
            if "answer" not in data:
                return False
            ans = data["answer"].strip().upper()
            if len(ans) != 1 or ans not in "ABCD":
                return False
            return True
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            return False

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> Tuple:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]

        # Prepare messages using chat template
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

        tgps_show_var = kwargs.get("tgps_show", False)

        # Enforced greedy decoding — ignore external do_sample/temperature
        gen_params = {
            "max_new_tokens": min(kwargs.get("max_new_tokens", 180), 180),
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": False,
        }

        if tgps_show_var:
            start_time = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_params)

        if tgps_show_var:
            generation_time = time.time() - start_time

        # Decode batch
        batch_outs = []
        token_len = 0 if tgps_show_var else None
        for input_ids, gen_seq in zip(model_inputs.input_ids, generated_ids):
            output_ids = gen_seq[len(input_ids) :]
            if tgps_show_var:
                token_len += len(output_ids)
            content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            content = self._extract_json(content)
            batch_outs.append(content)

        result = batch_outs[0] if len(batch_outs) == 1 else batch_outs

        if tgps_show_var:
            return result, token_len, generation_time
        return result, None, None


if __name__ == "__main__":
    agent = AAgent()
    response, tl, gt = agent.generate_response(
        "Solve: 2x + 5 = 15",
        system_prompt="You are a math tutor.",
        tgps_show=True,
        max_new_tokens=180,
    )
    print(f"Response: {response}")
    if tl and gt:
        print(f"Tokens: {tl}, Time: {gt:.2f}s, TGPS: {tl/gt:.2f}")
