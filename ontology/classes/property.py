from classes.parameter import Parameter


class Property:
    def __init__(self, domain: Parameter, range: Parameter, description: str = "No description available." ) -> None:
        self.range = range
        self.domain = domain
        self.description = description
