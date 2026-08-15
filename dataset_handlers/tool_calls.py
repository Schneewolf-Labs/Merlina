"""Flatten tool-calling conversations without inventing a dialect.

Datasets in the OpenAI tool-calling shape carry a `tools` column and assistant turns whose
`content` is null with a `tool_calls` list beside it. The messages converter flattens a
conversation to (system, prompt, chosen) strings, which has nowhere to put either -- so tool
schemas were dropped, tool calls were dropped, and a null content crashed the join with
"sequence item 0: expected str instance, NoneType found".

The temptation is to render tool calls as `<tool_call>{"name": ..., "arguments": ...}</tool_call>`,
which is the Hermes/Qwen2-era convention and is what most published tool-calling data looks like.
That is not safe. Qwen3.5 renders the same call as

    <tool_call>
    <function=take_action>
    <parameter=action_id>
    a7
    </parameter>
    </function>
    </tool_call>

A model trained on the JSON form and prompted with the XML form emits a dialect nothing parses,
and the loss curve looks perfectly healthy while it happens.

So nothing here hardcodes a format. Every piece is rendered by the target tokenizer's own chat
template, and the flattened strings are exact: re-templating them reproduces the native rendering
byte for byte. `verify_roundtrip` asserts precisely that, and the converter refuses to guess when
no tokenizer is available.

The mechanism is a content slot. Rendering one message whose content is a sentinel reveals the
prefix and suffix the template wraps around it; anything that starts with that prefix and ends
with that suffix can be turned back into a content string by slicing them off.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SENTINEL = "\u2063MERLINA_CONTENT_SLOT\u2063"


class ToolFlattenError(ValueError):
    """Raised when tool-calling data cannot be flattened faithfully."""


def row_has_tools(row: dict) -> bool:
    """True if this row carries tool schemas or any tool-call / tool-result turn."""
    if row.get("tools"):
        return True
    for msg in row.get("messages") or []:
        if msg.get("tool_calls") or msg.get("role") == "tool":
            return True
    return False


def dataset_has_tools(dataset: Any) -> bool:
    try:
        if "tools" in getattr(dataset, "column_names", []):
            return True
        return row_has_tools(dataset[0])
    except (IndexError, TypeError, KeyError):
        return False


# Templates reject conversations that contain no user turn ("No user query found in messages"),
# so a single message cannot be rendered on its own. Both the real render and the slot render use
# the same scaffold, which therefore cancels when the prefix is sliced off.
SCAFFOLD_USER = {"role": "user", "content": "\u2063scaffold\u2063"}


def _extract(tokenizer: Any, real: list, slot: list, label: str, tools=None) -> str:
    """The content string that, in `slot`'s position, reproduces `real`'s rendering."""
    rendered = tokenizer.apply_chat_template(real, tools=tools, tokenize=False)
    probe = tokenizer.apply_chat_template(slot, tokenize=False)
    if SENTINEL not in probe:
        raise ToolFlattenError(f"Template dropped the {label} content; cannot flatten faithfully.")
    head, _, tail = probe.partition(SENTINEL)
    if not (rendered.startswith(head) and rendered.endswith(tail)):
        raise ToolFlattenError(
            f"Rendered {label} turn does not match the template's own wrapper. This template "
            "needs explicit support rather than slot inference."
        )
    return rendered[len(head): len(rendered) - len(tail)]


def _normalise_call(call: dict) -> dict:
    """OpenAI tool calls nest under `function` and may carry arguments as a JSON string."""
    fn = call.get("function", call)
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {"name": fn.get("name"), "arguments": args if args is not None else {}}


def system_with_tools(tokenizer: Any, system_text: str, tools: list) -> str:
    """The system content that carries `tools`, in whatever shape the template uses."""
    return _extract(
        tokenizer,
        [{"role": "system", "content": system_text}, SCAFFOLD_USER],
        [{"role": "system", "content": SENTINEL}, SCAFFOLD_USER],
        "system",
        tools=tools,
    )


def assistant_with_calls(tokenizer: Any, message: dict) -> str:
    """Assistant content including its tool calls, in the template's own dialect."""
    msg = {"role": "assistant", "content": message.get("content") or ""}
    calls = message.get("tool_calls")
    if calls:
        msg["tool_calls"] = [
            {"type": "function", "function": _normalise_call(c)} for c in calls
        ]
    return _extract(
        tokenizer,
        [SCAFFOLD_USER, msg],
        [SCAFFOLD_USER, {"role": "assistant", "content": SENTINEL}],
        "assistant",
    )


def tool_result_as_user(tokenizer: Any, message: dict) -> str:
    """A tool result expressed as user content (templates fold tool turns into the user side)."""
    return _extract(
        tokenizer,
        [SCAFFOLD_USER, {"role": "tool", "content": message.get("content") or ""}],
        [SCAFFOLD_USER, {"role": "user", "content": SENTINEL}],
        "tool",
    )


def verify_roundtrip(tokenizer: Any, row: dict, flat: dict) -> bool:
    """True if the flattened strings re-template to exactly the native rendering.

    The whole safety argument rests on this, so it is a real comparison rather than a spot check.
    """
    # Datasets in the wild store tool-call arguments as a JSON string; templates index them as a
    # mapping and raise "Can only get item pairs from a mapping." Normalise before comparing so
    # this checks the flattening, not the source encoding.
    native_messages = []
    for m in row["messages"]:
        if m.get("tool_calls"):
            native_messages.append({
                "role": m["role"],
                "content": m.get("content") or "",
                "tool_calls": [
                    {"type": "function", "function": _normalise_call(c)} for c in m["tool_calls"]
                ],
            })
        else:
            native_messages.append({"role": m["role"], "content": m.get("content") or ""})
    native = tokenizer.apply_chat_template(
        native_messages, tools=row.get("tools"), tokenize=False,
    )
    rebuilt = []
    if flat.get("system"):
        rebuilt.append({"role": "system", "content": flat["system"]})
    rebuilt.append({"role": "user", "content": flat["prompt"]})
    rebuilt.append({"role": "assistant", "content": flat["chosen"]})
    return tokenizer.apply_chat_template(rebuilt, tokenize=False) == native


def flatten_row(row: dict, tokenizer: Optional[Any]) -> dict:
    """Flatten a tool-calling conversation to (system, prompt, chosen).

    Multi-turn conversations join with blank lines, matching the plain-text converter.
    """
    if tokenizer is None:
        raise ToolFlattenError(
            "This dataset uses tool calls, whose text form differs between model families "
            "(Qwen3.5 nests <function=...> inside <tool_call>, Hermes uses JSON). Flattening it "
            "without the target tokenizer would train a dialect the model is never prompted with. "
            "Set model_name on the dataset config so the tokenizer's own template can be used."
        )

    messages = row.get("messages") or []
    tools = row.get("tools")

    system_parts, user_parts, assistant_parts = [], [], []
    for msg in messages:
        role = (msg.get("role") or "").lower()
        if role == "system":
            system_parts.append(msg.get("content") or "")
        elif role == "user":
            user_parts.append(msg.get("content") or "")
        elif role == "tool":
            user_parts.append(tool_result_as_user(tokenizer, msg))
        elif role == "assistant":
            assistant_parts.append(assistant_with_calls(tokenizer, msg))
        else:
            logger.warning("Unknown role '%s' in messages, skipping", role)

    system = "\n\n".join(p for p in system_parts if p)
    if tools:
        system = system_with_tools(tokenizer, system, tools)

    prompt = "\n\n".join(p for p in user_parts if p)
    chosen = "\n\n".join(p for p in assistant_parts if p)

    if not prompt:
        raise ValueError("No user messages found in messages list")
    if not chosen:
        raise ValueError("No assistant messages found in messages list")

    return {"system": system, "prompt": prompt, "chosen": chosen}
