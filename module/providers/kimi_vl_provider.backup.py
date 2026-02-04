# Backup of kimi_vl_provider.py before modifications
# Timestamp placeholder
#
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from utils.parse_display import (
    display_caption_and_rate,
    display_pair_image_description,
    extract_code_block_content,
)


def _collect_stream_kimi(completion: Any, console: Console) -> str:
    chunks: list[str] = []
    for chunk in completion:
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                chunks.append(delta.content)
                console.print(".", end="", style="blue")
        except Exception:
            pass
    console.print("\n")
    return "".join(chunks)


def attempt_kimi_vl(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    uri: str,
    image_pixels: Optional[Pixels] = None,
    pair_pixels: Optional[Pixels] = None,
) -> str:
    start_time = time.time()

    completion = client.chat.completions.create(
        model=model_path,
        messages=messages,
        temperature=1,
        top_p=0.95,
        max_tokens=8192,
        stream=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")

    response_text = _collect_stream_kimi(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    if pair_pixels is not None and image_pixels is not None:
        display_pair_image_description(
            title=Path(uri).name,
            description=response_text,
            pixels=image_pixels,
            pair_pixels=pair_pixels,
            panel_height=32,
            console=console,
        )
        return response_text

    if image_pixels is not None:
        display_caption_and_rate(
            title=Path(uri).name,
            tag_description="",
            long_description=response_text,
            pixels=image_pixels,
            rating=[],
            average_score=0,
            panel_height=32,
            console=console,
        )

    return response_text
