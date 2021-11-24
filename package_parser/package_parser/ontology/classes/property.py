from dataclasses import dataclass
from typing import Optional

from package_parser.package_parser.ontology.classes.parameter import Parameter


class Property:
    description: Optional[str]
    domain: Parameter
    range: Parameter
