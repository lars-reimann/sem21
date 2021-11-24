from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Parameter:
    type: List[type]
    description: str
    import_path: str

    default_value = None
