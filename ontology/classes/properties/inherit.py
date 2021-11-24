from classes.parameter import Parameter
from classes.dependency import Dependency


class InheritsValueFrom(Dependency):
    """
    The domain Parameter inherits its value from an attribute of the range Parameter.
    E.g. if domain == None, attribute of range is used.
    """
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available.") -> None:
        super().__init__(domain, range, description=description)
