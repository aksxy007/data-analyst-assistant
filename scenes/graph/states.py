from typing_extensions import TypedDict
from typing import Annotated, Sequence, List
from langchain_core.messages import BaseMessage
import operator

from scenes.DataModels.InputData import InputData
class AgentState(TypedDict):
    """Data model for agent state."""
    messages: Annotated[Sequence[BaseMessage],operator.add]
    input_data: Annotated[List[InputData],operator.add]
    intermediate_outputs: Annotated[List[dict],operator.add]
    current_variables: dict
    output_image_paths: Annotated[List[str], operator.add]