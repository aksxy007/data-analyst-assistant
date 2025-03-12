from dataclasses import dataclass

@dataclass
class ToolInvocation:
    """Data model for tool invocation."""
    tool: str
    tool_inputs: dict