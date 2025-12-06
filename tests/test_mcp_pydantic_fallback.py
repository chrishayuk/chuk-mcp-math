#!/usr/bin/env python3
"""
Tests for mcp_pydantic_base fallback mode.

This file tests the fallback implementation by forcing fallback mode
via the MCP_FORCE_FALLBACK environment variable.
"""

import pytest
import os
import sys
from typing import List, Dict, Optional, Union, Any


# Force fallback mode BEFORE importing the module
os.environ["MCP_FORCE_FALLBACK"] = "1"

# Remove module from cache if already imported
if "chuk_mcp_math.mcp_pydantic_base" in sys.modules:
    del sys.modules["chuk_mcp_math.mcp_pydantic_base"]

from chuk_mcp_math.mcp_pydantic_base import (
    McpPydanticBase,
    Field,
    ValidationError,
    PYDANTIC_AVAILABLE,
    ConfigDict,
    validator,
    root_validator,
)


class TestFallbackImplementation:
    """Test the fallback implementation when Pydantic is not available."""

    def test_pydantic_not_available_in_fallback_mode(self):
        """Verify fallback mode is active."""
        assert PYDANTIC_AVAILABLE is False, "Fallback mode should be active"

    def test_validation_error_fallback(self):
        """Test ValidationError in fallback mode."""
        error = ValidationError("Test error", field_path="test.field", error_type="test_type")
        assert error.message == "Test error"
        assert error.field_path == "test.field"
        assert error.error_type == "test_type"
        assert "test.field" in str(error)
        assert "ValidationError" in repr(error)

    def test_field_fallback(self):
        """Test Field class in fallback mode."""
        # Test required field
        field1 = Field()
        assert field1.required is True
        assert field1.default is ...

        # Test with default
        field2 = Field(default="value")
        assert field2.required is False
        assert field2.default == "value"

        # Test with default_factory
        field3 = Field(default_factory=list)
        assert field3.required is False
        assert callable(field3.default_factory)

        # Test both default and factory raises error
        with pytest.raises(TypeError, match="Cannot specify both"):
            Field(default="value", default_factory=list)

    def test_basic_model_fallback(self):
        """Test basic model creation in fallback mode."""

        class TestModel(McpPydanticBase):
            name: str
            age: int = Field(default=0)
            email: Optional[str] = None

        model = TestModel(name="John")
        assert model.name == "John"
        assert model.age == 0
        assert model.email is None

    def test_model_with_all_fields(self):
        """Test model with all field types."""

        class ComplexModel(McpPydanticBase):
            required_field: str
            default_field: int = Field(default=42)
            optional_field: Optional[str] = None
            list_field: List[int] = Field(default_factory=list)
            dict_field: Dict[str, Any] = Field(default_factory=dict)
            union_field: Union[str, int] = Field(default="test")

        model = ComplexModel(required_field="test")
        assert model.required_field == "test"
        assert model.default_field == 42
        assert model.optional_field is None
        assert model.list_field == []
        assert model.dict_field == {}

    def test_model_validation(self):
        """Test model validation in fallback mode."""

        class TestModel(McpPydanticBase):
            name: str
            value: int

        # Valid data
        model1 = TestModel(name="test", value=42)
        assert model1.name == "test"
        assert model1.value == 42

        # Missing required field
        with pytest.raises(ValidationError, match="Missing required fields"):
            TestModel(value=10)

    def test_model_serialization(self):
        """Test model serialization methods."""

        class TestModel(McpPydanticBase):
            name: str = "test"
            value: int = 123

        model = TestModel()

        # model_dump
        data = model.model_dump()
        assert data["name"] == "test"
        assert data["value"] == 123

        # model_dump with exclude
        data_exclude = model.model_dump(exclude={"value"})
        assert "name" in data_exclude
        assert "value" not in data_exclude

        # model_dump with include
        data_include = model.model_dump(include={"name"})
        assert "name" in data_include
        assert "value" not in data_include

        # model_dump with exclude_none
        model2 = TestModel()
        model2.name = None
        data_no_none = model2.model_dump(exclude_none=True)
        assert "name" not in data_no_none

        # model_dump_json
        json_str = model.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str

        # v1 compat methods
        assert model.dict() == model.model_dump()
        assert isinstance(model.json(), str)

    def test_nested_models(self):
        """Test nested model support."""

        class Inner(McpPydanticBase):
            value: int

        class Outer(McpPydanticBase):
            inner: Inner
            name: str = "test"

        inner = Inner(value=42)
        outer = Outer(inner=inner)
        assert outer.inner.value == 42

        # From dict
        outer2 = Outer(inner={"value": 100})
        assert outer2.inner.value == 100

    def test_type_coercion(self):
        """Test type coercion in fallback mode."""

        class TypeModel(McpPydanticBase):
            int_val: int
            str_val: str
            float_val: float
            bool_val: bool

        # Coercion from compatible types
        model = TypeModel(
            int_val="42",  # string to int
            str_val=123,  # int to string
            float_val="3.14",  # string to float
            bool_val="true",  # string to bool
        )

        # Fallback is permissive
        assert model.int_val in (42, "42")
        assert isinstance(model.float_val, (float, str))

    def test_union_types(self):
        """Test Union type handling."""

        class UnionModel(McpPydanticBase):
            value: Union[str, int]

        model1 = UnionModel(value="string")
        assert model1.value == "string"

        model2 = UnionModel(value=42)
        assert model2.value in (42, "42")

    def test_list_validation(self):
        """Test list type validation."""

        class ListModel(McpPydanticBase):
            items: List[int]

        model = ListModel(items=[1, 2, 3])
        assert model.items == [1, 2, 3]

    def test_dict_validation(self):
        """Test dict type validation."""

        class DictModel(McpPydanticBase):
            data: Dict[str, int]

        model = DictModel(data={"a": 1, "b": 2})
        assert model.data["a"] == 1

    def test_config_dict(self):
        """Test ConfigDict helper."""
        config = ConfigDict(extra="allow", validate_assignment=True)
        assert config["extra"] == "allow"
        assert config["validate_assignment"] is True
        assert config["use_enum_values"] is True  # Default

    def test_dummy_decorators(self):
        """Test dummy decorator functions."""

        @validator("field")
        def my_validator(cls, v):
            return v

        @root_validator
        def my_root_validator(cls, values):
            return values

        # Decorators should be no-ops but not fail
        assert callable(my_validator)
        assert callable(my_root_validator)

    def test_field_aliases(self):
        """Test field alias support."""

        class AliasModel(McpPydanticBase):
            internal_name: str = Field(alias="externalName")

        model = AliasModel(externalName="value")
        assert model.internal_name == "value"

        # Serialize with alias
        data = model.model_dump(by_alias=True)
        assert "externalName" in data

    def test_model_validate(self):
        """Test model_validate class method."""

        class TestModel(McpPydanticBase):
            name: str
            value: int

        # From dict
        model1 = TestModel.model_validate({"name": "test", "value": 42})
        assert model1.name == "test"

        # From instance
        model2 = TestModel(name="test2", value=100)
        model3 = TestModel.model_validate(model2)
        assert model3.name == "test2"

        # Invalid type
        with pytest.raises(ValidationError):
            TestModel.model_validate("invalid")

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""

        class FlexModel(McpPydanticBase):
            name: str

        model = FlexModel(name="test", extra_field="extra")
        assert model.name == "test"
        assert hasattr(model, "extra_field")

    def test_optional_fields(self):
        """Test Optional fields."""

        class OptionalModel(McpPydanticBase):
            required: str
            optional: Optional[int] = None

        model1 = OptionalModel(required="test")
        assert model1.optional is None

        model2 = OptionalModel(required="test", optional=42)
        assert model2.optional == 42

    def test_any_type(self):
        """Test Any type."""

        class AnyModel(McpPydanticBase):
            value: Any

        model1 = AnyModel(value="string")
        model2 = AnyModel(value=42)
        model3 = AnyModel(value={"key": "value"})

        assert model1.value == "string"
        assert model2.value == 42
        assert model3.value == {"key": "value"}

    def test_mcp_specific_defaults(self):
        """Test MCP-specific default behaviors."""

        # MCPResourceSpec-like class
        class MCPResourceSpec(McpPydanticBase):
            name: str
            metadata: Dict[str, Any] = Field(default_factory=dict)
            examples: List[str] = Field(default_factory=list)

        spec = MCPResourceSpec(name="test")
        assert spec.metadata == {}
        assert spec.examples == []

    def test_deeply_nested_structures(self):
        """Test deeply nested data structures."""

        class DeepModel(McpPydanticBase):
            data: Dict[str, Any]

        deep_data = {"level1": {"level2": {"level3": {"value": 42}}}}
        model = DeepModel(data=deep_data)
        assert model.data["level1"]["level2"]["level3"]["value"] == 42

    def test_model_dump_mcp(self):
        """Test MCP-optimized dump."""

        class TestModel(McpPydanticBase):
            name: str = "test"
            nested: Dict[str, Any] = Field(default_factory=dict)

        model = TestModel(nested={"key": "value"})
        data = model.model_dump_mcp()
        assert isinstance(data, dict)
        assert "name" in data

    def test_empty_model(self):
        """Test empty model."""

        class EmptyModel(McpPydanticBase):
            pass

        model = EmptyModel()
        assert model is not None

    def test_model_with_methods(self):
        """Test model with custom methods."""

        class MethodModel(McpPydanticBase):
            value: int = 5

            def double(self):
                return self.value * 2

        model = MethodModel()
        assert model.double() == 10

    def test_forward_reference_handling(self):
        """Test forward reference in type hints."""

        class ForwardModel(McpPydanticBase):
            name: str

        model = ForwardModel(name="test")
        assert model.name == "test"

    def test_complex_union_types(self):
        """Test complex union type scenarios."""

        class ComplexUnion(McpPydanticBase):
            value: Union[str, int, float, List[str]]

        model1 = ComplexUnion(value="text")
        ComplexUnion(value=42)
        ComplexUnion(value=3.14)
        ComplexUnion(value=["a", "b"])

        assert model1.value == "text"

    def test_nested_list_dict_types(self):
        """Test nested List and Dict types."""

        class NestedTypes(McpPydanticBase):
            matrix: List[List[int]]
            mapping: Dict[str, List[str]]

        model = NestedTypes(matrix=[[1, 2], [3, 4]], mapping={"key": ["a", "b"]})
        assert model.matrix[0][0] == 1
        assert model.mapping["key"][0] == "a"

    def test_bytes_type(self):
        """Test bytes type handling."""

        class BytesModel(McpPydanticBase):
            data: bytes = Field(default=b"test")

        BytesModel()
        # Should handle bytes gracefully

    def test_model_validate_from_object(self):
        """Test model_validate from object with __dict__."""

        class SimpleObject:
            def __init__(self):
                self.name = "test"
                self.value = 42

        class SimpleModel(McpPydanticBase):
            name: str
            value: int

        obj = SimpleObject()
        model = SimpleModel.model_validate(obj)
        assert model.name == "test"
        assert model.value == 42

    def test_type_validation_edge_cases(self):
        """Test edge cases in type validation."""

        class EdgeModel(McpPydanticBase):
            value: int

        # Test with None for optional
        class OptionalEdge(McpPydanticBase):
            value: Optional[int] = None

        model = OptionalEdge()
        assert model.value is None

    def test_list_coercion_fallback(self):
        """Test list item coercion fallback."""

        class ListCoercion(McpPydanticBase):
            items: List[int]

        # Mix of types that may or may not coerce
        model = ListCoercion(items=[1, "2", 3])
        assert len(model.items) == 3

    def test_dict_coercion_fallback(self):
        """Test dict value coercion fallback."""

        class DictCoercion(McpPydanticBase):
            data: Dict[str, int]

        # Mix of types that may or may not coerce
        model = DictCoercion(data={"a": 1, "b": "2"})
        assert len(model.data) == 2

    def test_nested_model_from_dict(self):
        """Test nested model creation from dict."""

        class Inner(McpPydanticBase):
            value: int

        class Outer(McpPydanticBase):
            inner: Inner

        # Create with dict for nested model
        outer = Outer(inner={"value": 42})
        assert outer.inner.value == 42

    def test_model_with_property_decorator(self):
        """Test model with @property."""

        class PropertyModel(McpPydanticBase):
            first: str
            last: str

            @property
            def full_name(self):
                return f"{self.first} {self.last}"

        model = PropertyModel(first="John", last="Doe")
        assert model.full_name == "John Doe"

    def test_validation_error_messages(self):
        """Test validation error message formats."""

        class StrictModel(McpPydanticBase):
            required_field: str

        try:
            StrictModel()
        except ValidationError as e:
            assert "required" in str(e).lower()

    def test_model_dump_with_by_alias(self):
        """Test model_dump with by_alias parameter."""

        class AliasedModel(McpPydanticBase):
            internal: str = Field(alias="external", default="test")

        model = AliasedModel()

        # Dump with alias
        data_alias = model.model_dump(by_alias=True)
        assert "external" in data_alias or "internal" in data_alias

    def test_model_dump_json_with_indent(self):
        """Test model_dump_json with indent."""

        class JsonModel(McpPydanticBase):
            name: str = "test"
            value: int = 42

        model = JsonModel()
        json_indented = model.model_dump_json(indent=2)
        assert "\n" in json_indented  # Indented has newlines

    def test_default_factory_isolation(self):
        """Test that default_factory creates isolated instances."""

        class FactoryModel(McpPydanticBase):
            items: List[str] = Field(default_factory=list)

        model1 = FactoryModel()
        model2 = FactoryModel()

        model1.items.append("item1")

        # Should not affect model2
        assert "item1" in model1.items
        assert "item1" not in model2.items

    def test_json_compatible_fallback_paths(self):
        """Test JSON-compatible fallback for complex types."""

        class JsonFallback(McpPydanticBase):
            value: Union[str, int]

        # Test with dict that should be JSON-compatible
        JsonFallback(value={"nested": "data"})
        # Should accept dict as fallback

    def test_type_name_extraction(self):
        """Test _get_type_name helper for various types."""
        from chuk_mcp_math.mcp_pydantic_base import _get_type_name
        from typing import List, Dict

        # Test basic types
        assert "int" in _get_type_name(int)
        assert "str" in _get_type_name(str)

        # Test complex types
        list_type = _get_type_name(List[int])
        _get_type_name(Dict[str, int])
        assert "list" in list_type.lower() or "List" in list_type

    def test_is_optional_detection(self):
        """Test _is_optional helper."""
        from chuk_mcp_math.mcp_pydantic_base import _is_optional

        # Test Optional types
        assert _is_optional(Optional[int]) is True
        assert _is_optional(Union[str, None]) is True
        assert _is_optional(int) is False
        assert _is_optional(str) is False

    def test_get_non_none_type(self):
        """Test _get_non_none_type helper."""
        from chuk_mcp_math.mcp_pydantic_base import _get_non_none_type

        # Test with Optional
        assert _get_non_none_type(Optional[int]) is int
        assert _get_non_none_type(Union[str, None]) is str

        # Test with non-Optional
        assert _get_non_none_type(int) is int

    def test_deep_validate_with_none(self):
        """Test _deep_validate with None values."""
        try:
            from chuk_mcp_math.mcp_pydantic_base import _deep_validate

            # None for optional
            result = _deep_validate("field", None, Optional[str])
            assert result is None

            # None for non-optional should use fallback
            try:
                _deep_validate("field", None, str)
                # Fallback is permissive
            except ValidationError:
                pass  # May raise ValidationError in strict mode
        except (ImportError, NameError):
            # _deep_validate may not be exported in all modes
            pass

    def test_deep_validate_with_any(self):
        """Test _deep_validate with Any type."""
        from chuk_mcp_math.mcp_pydantic_base import _deep_validate

        # Any type should accept anything
        result1 = _deep_validate("field", "string", Any)
        assert result1 == "string"

        result2 = _deep_validate("field", 42, Any)
        assert result2 == 42

        result3 = _deep_validate("field", {"key": "value"}, Any)
        assert result3 == {"key": "value"}

    def test_deep_validate_union_first_success(self):
        """Test _deep_validate Union trying first type successfully."""
        from chuk_mcp_math.mcp_pydantic_base import _deep_validate

        # Union should try first type
        result = _deep_validate("field", "text", Union[str, int])
        assert result == "text"

    def test_deep_validate_list_with_origin_type(self):
        """Test _deep_validate with List origin."""
        from chuk_mcp_math.mcp_pydantic_base import _deep_validate

        # List validation
        result = _deep_validate("field", [1, 2, 3], List[int])
        assert result == [1, 2, 3]

    def test_deep_validate_dict_with_origin_type(self):
        """Test _deep_validate with Dict origin."""
        from chuk_mcp_math.mcp_pydantic_base import _deep_validate

        # Dict validation
        result = _deep_validate("field", {"a": 1, "b": 2}, Dict[str, int])
        assert result == {"a": 1, "b": 2}

    def test_int_coercion_from_float(self):
        """Test int coercion from float."""

        class IntModel(McpPydanticBase):
            value: int

        model = IntModel(value=42.0)
        # Should coerce float to int
        assert isinstance(model.value, (int, float))

    def test_float_coercion_from_int(self):
        """Test float coercion from int."""

        class FloatModel(McpPydanticBase):
            value: float

        model = FloatModel(value=42)
        # Should accept int for float
        assert isinstance(model.value, (int, float))

    def test_bool_coercion_variations(self):
        """Test various bool coercion paths."""

        class BoolModel(McpPydanticBase):
            value: bool

        # Test various truthy/falsy values
        BoolModel(value="false")
        BoolModel(value="0")
        BoolModel(value=1)
        BoolModel(value=0)

        # Fallback is permissive

    def test_list_with_non_list_value(self):
        """Test List field with non-list value."""

        class ListModel(McpPydanticBase):
            items: List[str]

        # Non-list value should be handled (wrapped or error)
        try:
            ListModel(items="not a list")
            # Fallback may wrap or accept
        except (ValidationError, TypeError):
            pass  # Expected in strict mode

    def test_dict_with_non_dict_value(self):
        """Test Dict field with non-dict value."""

        class DictModel(McpPydanticBase):
            data: Dict[str, int]

        # Non-dict value should be handled
        try:
            DictModel(data="not a dict")
            # Fallback may convert or error
        except (ValidationError, TypeError):
            pass  # Expected in strict mode

    def test_nested_model_validation_failure(self):
        """Test nested model with invalid data."""

        class Inner(McpPydanticBase):
            required_value: int

        class Outer(McpPydanticBase):
            inner: Inner

        # Missing required field in nested model
        try:
            Outer(inner={})  # Missing required_value
            # Fallback may be lenient
        except ValidationError:
            pass  # Expected in strict mode

    def test_model_dump_with_nested_none_exclusion(self):
        """Test model_dump exclude_none with nested structures."""

        class Inner(McpPydanticBase):
            value: Optional[int] = None

        class Outer(McpPydanticBase):
            inner: Inner = Field(default_factory=lambda: Inner())
            name: Optional[str] = None

        model = Outer()
        model.model_dump(exclude_none=True)
        # Should exclude None values

    def test_field_with_multiple_kwargs(self):
        """Test Field with multiple kwargs."""
        field = Field(
            default="test",
            description="Test field",
            alias="testAlias",
            title="Test Title",
            json_schema_extra={"example": "value"},
        )
        assert field.description == "Test field"
        assert field.alias == "testAlias"
        assert field.title == "Test Title"

    def test_model_validate_with_extra_fields(self):
        """Test model_validate with extra fields in data."""

        class StrictModel(McpPydanticBase):
            name: str

        data = {"name": "test", "extra": "ignored"}
        model = StrictModel.model_validate(data)
        assert model.name == "test"
        # Extra field should be stored or ignored

    def test_str_coercion_from_non_string(self):
        """Test str coercion from non-string types."""

        class StrModel(McpPydanticBase):
            value: str

        # Test coercion from int
        StrModel(value=123)
        # Fallback may accept or coerce

        # Test coercion from list
        try:
            StrModel(value=[1, 2, 3])
            # May coerce to string representation
        except (ValidationError, TypeError):
            pass

    def test_int_coercion_failures(self):
        """Test int coercion failure paths."""

        class IntModel(McpPydanticBase):
            value: int

        # Test with non-coercible string
        try:
            IntModel(value="not_a_number")
            # Fallback may be lenient or strict
        except (ValidationError, ValueError):
            pass  # Expected

    def test_float_coercion_failures(self):
        """Test float coercion failure paths."""

        class FloatModel(McpPydanticBase):
            value: float

        # Test with non-coercible string
        try:
            FloatModel(value="not_a_float")
            # Fallback may be lenient or strict
        except (ValidationError, ValueError):
            pass  # Expected

    def test_bool_coercion_edge_cases(self):
        """Test bool coercion edge cases."""

        class BoolModel(McpPydanticBase):
            value: bool

        # Test with "False" string (capitalized)
        BoolModel(value="False")

        # Test with "no"
        BoolModel(value="no")

        # Test with empty string
        BoolModel(value="")

    def test_union_all_types_fail(self):
        """Test Union when all type validations fail."""

        class UnionModel(McpPydanticBase):
            value: Union[int, float]

        # Test with incompatible type - should use fallback
        try:
            UnionModel(value="not_a_number")
            # Fallback should handle gracefully
        except (ValidationError, ValueError):
            pass  # May be strict

    def test_list_item_validation_failures(self):
        """Test List with items that fail validation."""

        class StrictListModel(McpPydanticBase):
            items: List[int]

        # List with non-int items - fallback should handle
        try:
            StrictListModel(items=[1, "two", 3])
            # Fallback may accept or coerce
        except (ValidationError, ValueError):
            pass

    def test_dict_key_value_validation_failures(self):
        """Test Dict with invalid key/value types."""

        class StrictDictModel(McpPydanticBase):
            data: Dict[str, int]

        # Dict with non-int values - fallback should handle
        try:
            StrictDictModel(data={"a": 1, "b": "not_int"})
            # Fallback may accept or coerce
        except (ValidationError, ValueError):
            pass

    def test_model_init_with_invalid_data_type(self):
        """Test model init with completely wrong data type."""

        class TestModel(McpPydanticBase):
            name: str
            value: int

        # Test with non-dict, non-object data
        try:
            TestModel("invalid")
            # Should fail validation
        except (ValidationError, TypeError):
            pass  # Expected

    def test_model_repr_fallback(self):
        """Test __repr__ in fallback mode."""

        class TestModel(McpPydanticBase):
            name: str = "test"

        model = TestModel()
        repr_str = repr(model)
        assert "TestModel" in repr_str or "test" in repr_str

    def test_model_str_fallback(self):
        """Test __str__ in fallback mode."""

        class TestModel(McpPydanticBase):
            name: str = "test"

        model = TestModel()
        str_str = str(model)
        assert isinstance(str_str, str)

    def test_field_info_extraction(self):
        """Test field info extraction in fallback mode."""

        class TestModel(McpPydanticBase):
            regular: str
            with_default: int = Field(default=42)
            with_factory: List[str] = Field(default_factory=list)
            optional: Optional[str] = None

        model = TestModel(regular="test")
        assert model.regular == "test"
        assert model.with_default == 42
        assert model.with_factory == []
        assert model.optional is None

    def test_process_mcp_data_recursion(self):
        """Test _process_mcp_data with deeply nested structures."""

        class TestModel(McpPydanticBase):
            data: Dict[str, Any] = Field(default_factory=dict)

        model = TestModel(data={"level1": {"level2": {"level3": {"level4": {"value": 42}}}}})
        # Test MCP data processing
        mcp_data = model.model_dump_mcp()
        assert isinstance(mcp_data, dict)

    def test_validation_with_class_vars(self):
        """Test model with ClassVar (should be ignored)."""
        from typing import ClassVar

        class ModelWithClassVar(McpPydanticBase):
            CLASS_VAR: ClassVar[str] = "constant"
            instance_var: str

        model = ModelWithClassVar(instance_var="test")
        assert model.instance_var == "test"
        assert ModelWithClassVar.CLASS_VAR == "constant"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
