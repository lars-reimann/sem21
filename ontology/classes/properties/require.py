from classes.property import Property
from classes.property import Property


class Require(Property):
    """
    The domain Parameter requires that the range Parameter is initialized.
    E.g. If domain = val, then range is used.
    """
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available.") -> None:
        super().__init__(domain, range, description=description)
