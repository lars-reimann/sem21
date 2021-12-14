import pytest
from typing import Optional
from package_parser.commands.get_api._model import Parameter


@pytest.mark.parametrize(
    "docstring_type,expected",
    [
        ('{"frobenius", "spectral"}, default="frobenius"', {"frobenius", "spectral"}),
        (
            "{'strict', 'ignore', 'replace'}, default='strict'",
            {"strict", "ignore", "replace"},
        ),
        (
            "{'linear', 'poly',             'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='linear'",
            {"linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"},
        ),
        ("", None),
    ],
)
def test_enum_from_docstring_type(docstring_type: str, expected: Optional[set[str]]):
    result = Parameter.extract_enum(docstring_type)
    assert result == expected
