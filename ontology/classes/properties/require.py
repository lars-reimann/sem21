from classes.parameter import Parameter
from classes.dependency import Dependency


class Requires(Dependency):
    """
    The domain Parameter requires that the range Parameter is initialized.
    E.g. domain is initialized only when range = val.
    """
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available.") -> None:
        super().__init__(domain, range, description=description)
