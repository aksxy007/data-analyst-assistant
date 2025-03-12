from dataclasses import dataclass
from typing import List
@dataclass
class InputData:
    """Data model for input data."""
    variable_name: str
    data_path:str
    data_description: str