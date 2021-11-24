from classes.parameter import Parameter
from classes.dependency import Dependency


class Arithmetic(Dependency):
    """
    The domain Parameter requires the range parameter to calculate its own value.
    E.g. domain = range * 10.
    """
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available.") -> None:
        super().__init__(domain, range, description=description)
