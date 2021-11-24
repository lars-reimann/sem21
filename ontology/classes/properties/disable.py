from classes.parameter import Parameter
from classes.dependency import Dependency


class Disables(Dependency):
    """
    The domain Parameter disables the range Parameter.
    E.g. if domain == val, range is an ignored parameter.
    """
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available.") -> None:
        super().__init__(domain, range, description=description)
