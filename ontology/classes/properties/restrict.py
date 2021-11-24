from classes.parameter import Parameter
from classes.property import Property


class Restricts(Property):
    """
    The domain Parameter restricts the value of the range Parameter.
    E.g. Range value cannot be larger than domain value.
    """
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available.") -> None:
        super().__init__(domain, range, description=description)
