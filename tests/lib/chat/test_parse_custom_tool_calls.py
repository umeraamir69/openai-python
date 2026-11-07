from __future__ import annotations

from openai.lib._parsing._completions import parse_chat_completion
from openai.types.chat import ChatCompletion


def test_parse_chat_completion_keeps_custom_tool_calls() -> None:
    completion = ChatCompletion.construct(
        id="chatcmpl_test",
        object="chat.completion",
        created=0,
        model="gpt-4o",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fn",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"SF"}',
                            },
                        },
                        {
                            "id": "call_custom",
                            "type": "custom",
                            "custom": {
                                "name": "code_exec",
                                "input": "print(1)",
                            },
                        },
                    ],
                },
            }
        ],
    )

    parsed = parse_chat_completion(
        response_format=None,
        chat_completion=completion,
        input_tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
    )

    tool_calls = parsed.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[1].type == "custom"
    assert tool_calls[1].custom.name == "code_exec"
    assert tool_calls[1].custom.input == "print(1)"
