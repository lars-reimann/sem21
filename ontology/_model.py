from typing import Any, Optional

from enum import Enum, auto
from typing import Any, Dict, Optional


class ParameterAssignment(Enum):
    POSITION_ONLY = (auto(),)
    POSITION_OR_NAME = (auto(),)
    NAME_ONLY = (auto(),)


class ParameterAndResultDocstring:
    @classmethod
    def from_json(cls, json: Any):
        return cls(json["type"], json["description"])

    def __init__(
        self,
        type_: str,
        description: str,
    ) -> None:
        self.type: str = type_
        self.description: str = description

    def to_json(self) -> Dict:
        return {"type": self.type, "description": self.description}


class Parameter:
    @classmethod
    def from_json(cls, json: Any):
        return cls(
            json["name"],
            json["default_value"],
            json["is_public"],
            ParameterAssignment[json["assigned_by"]],
            ParameterAndResultDocstring.from_json(json["docstring"]),
        )

    def __init__(
        self,
        name: str,
        default_value: Optional[str],
        is_public: bool,
        assigned_by: ParameterAssignment,
        docstring: ParameterAndResultDocstring,
    ) -> None:
        self.name: str = name
        self.default_value: Optional[str] = default_value
        self.is_public: bool = is_public
        self.assigned_by: ParameterAssignment = assigned_by
        self.docstring = docstring

    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "default_value": self.default_value,
            "is_public": self.is_public,
            "assigned_by": self.assigned_by.name,
            "docstring": self.docstring.to_json(),
        }




class Action:
    @classmethod
    def from_json(cls, json: Any):
        return cls(
            json['action']
        )

    def __init__(
        self,
        action: str
    ) -> None:
        self.action = action
    
    def to_json(self) -> Dict:
        return {
            "action": self.action
        }

class RuntimeAction(Action):
    def __init__(self, action: str) -> None:
        super().__init__(action)


class StaticAction(Action):
    def __init__(self, action: str) -> None:
        super().__init__(action)


class ParameterIsIgnored(StaticAction):
    def __init__(self, action: str) -> None:
        super().__init__(action)


class ParameterIsIllegal(StaticAction):
    def __init__(self, action: str) -> None:
        super().__init__(action)




class Condition:
    @classmethod
    def from_json(cls, json: Any):
        return cls(
            json['condition']
        )

    def __init__(
        self,
        condition: str
    ) -> None:
        self.condition = condition
    
    def to_json(self) -> Dict:
        return {
            "condition": self.condition
        }


class RuntimeCondition(Condition):
    def __init__(self, condition: str) -> None:
        super().__init__(condition)


class StaticCondition(Condition):
    def __init__(self, condition: str) -> None:
        super().__init__(condition)


class ParameterHasValue(StaticCondition):
    def __init__(self, condition: str) -> None:
        super().__init__(condition)


class ParameterIsSet(StaticCondition):
    def __init__(self, condition: str) -> None:
        super().__init__(condition)




class Dependency:
    @classmethod
    def from_json(cls, json: Any):
        return cls(
            Parameter.from_json(['hasDependentParameter']),
            Parameter.from_json(['isDependingOn']),
            Condition.from_json(['hasCondition']),
            Action.from_json(['hasAction'])
        )

    def __init__(
        self,
        hasDependentParameter: Parameter,
        isDependingOn: Parameter,
        hasCondition: Condition,
        hasAction: Action
    ) -> None:
        self.hasDependentParameter = hasDependentParameter
        self.isDependingOn = isDependingOn
        self.hasCondition = hasCondition
        self.hasAction = hasAction

    def to_json(self) -> Dict:
        return {
            "hasDependentParameter": self.hasDependentParameter.to_json(),
            "isDependingOn": self.isDependingOn.to_json(),
            "hasCondition": self.hasCondition.to_json(),
            "hasAction": self.hasAction.to_json(),
        }
