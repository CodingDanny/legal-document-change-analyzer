from __future__ import annotations
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

MAX_WORKERS = 4
MAX_RETRIES = 3
BASE_BACKOFF = 0.5
RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}

client = OpenAI()

def openai_call(instructions: str, prompt:str, text_format_class, model: str = "gpt-5-mini", reasoning_effort: str = "low") -> Dict[str, str]:
    backoff = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.parse(
                model=model,
                reasoning={"effort": reasoning_effort},
                instructions=instructions,
                input=prompt,
                text_format=text_format_class,
            )
            return response.output_parsed

        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in RETRY_STATUSES and attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("Failed to classify change")

def parallel_openai_calls(instructions: str, prompts: List[str], text_format_class) -> List[Dict[str, str]]:
    results: List[Dict[str, str] | None] = [None] * len(prompts)
    max_workers = min(MAX_WORKERS, max(1, len(prompts)))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_idx = {pool.submit(openai_call, instructions, prompt, text_format_class): i for i, prompt in enumerate(prompts)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()

    return results

