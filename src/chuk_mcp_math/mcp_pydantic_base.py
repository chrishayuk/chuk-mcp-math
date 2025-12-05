# chuk_mcp_math/mcp_pydantic_base.py
import os
import json
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    Callable,
)

"""Enhanced minimal-footprint drop-in replacement for Pydantic optimized for MCP.

Key improvements for MCP usage:
1. Better type validation including nested models
2. More robust Field handling with MCP-specific defaults
3. Better error messages with field path tracking
4. Support for model_post_init and validation decorators
5. Enhanced serialization with custom encoders for MCP resources
6. Proper handling of Union types including Union[str, int]
7. More permissive validation for JSON-like data structures
8. MCP-specific optimizations and defaults
"""

FORCE_FALLBACK = os.environ.get("MCP_FORCE_FALLBACK") == "1"

try:
    if not FORCE_FALLBACK:
        from pydantic import (
            BaseModel as PydanticBase,
            Field as PydanticField,
            ConfigDict as PydanticConfigDict,
            ValidationError,
            validator,
            root_validator,
        )

        # Check if we have Pydantic v2
        try:
            from pydantic import __version__ as pydantic_version

            PYDANTIC_V2 = pydantic_version.startswith("2.")
        except Exception:
            PYDANTIC_V2 = False

        PYDANTIC_AVAILABLE = True
    else:
        PYDANTIC_AVAILABLE = False
except ImportError:
    PYDANTIC_AVAILABLE = False

# Re-exports when Pydantic is available
if PYDANTIC_AVAILABLE:

    class McpPydanticBase(PydanticBase):
        """Enhanced Pydantic base class with MCP-specific features."""

        if PYDANTIC_V2:
            # Pydantic v2 configuration
            model_config = {"extra": "allow", "validate_assignment": True}
        else:
            # Pydantic v1 configuration
            class Config:
                extra = "allow"
                validate_assignment = True
                json_encoders = {
                    # Custom encoders for MCP types
                    bytes: lambda v: v.decode("utf-8")
                    if isinstance(v, bytes)
                    else str(v),
                }

        def model_dump_mcp(self, **kwargs) -> Dict[str, Any]:
            """MCP-optimized model dump with additional processing."""
            if PYDANTIC_V2:
                data = self.model_dump(**kwargs)
            else:
                data = self.dict(**kwargs)

            # Post-process for MCP compatibility
            return self._process_mcp_data(data)

        def _process_mcp_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Process data for MCP compatibility."""
            processed: Dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    processed[key] = self._process_mcp_data(value)
                elif isinstance(value, list):
                    processed[key] = [
                        self._process_mcp_data(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    processed[key] = value
            return processed

    Field = PydanticField
    ConfigDict = PydanticConfigDict
else:
    # Enhanced fallback implementation optimized for MCP

    class ValidationError(Exception):  # type: ignore[no-redef]
        """Enhanced validation error with field path tracking for MCP."""

        def __init__(
            self,
            message: str,
            field_path: str = "",
            error_type: str = "validation_error",
        ):
            self.field_path = field_path
            self.error_type = error_type
            self.message = message
            super().__init__(f"{field_path}: {message}" if field_path else message)

        def __repr__(self):
            return f"ValidationError(message='{self.message}', field_path='{self.field_path}')"

    def _get_type_name(t: Any) -> str:
        """Get a readable name for a type."""
        if hasattr(t, "__name__"):
            return t.__name__
        return str(t)

    def _is_optional(t: Any) -> bool:
        """Check if a type is Optional (Union with None)."""
        origin, args = get_origin(t), get_args(t)
        return origin is Union and type(None) in args

    def _get_non_none_type(t: Any) -> Any:
        """Extract the non-None type from Optional[T]."""
        if _is_optional(t):
            args = get_args(t)
            return next(arg for arg in args if arg is not type(None))
        return t

    def _deep_validate(name: str, value: Any, expected: Any, path: str = "") -> Any:
        """Enhanced recursive validation optimized for MCP resource patterns."""
        current_path = f"{path}.{name}" if path else name

        if value is None:
            if _is_optional(expected):
                return None
            raise ValidationError("field required", current_path, "missing")

        # Handle typing.Any first to avoid isinstance() errors
        if expected is Any:
            return value

        # Handle Optional types
        if _is_optional(expected):
            expected = _get_non_none_type(expected)
            if expected is Any:
                return value

        origin = get_origin(expected)

        # Handle Union types FIRST (including Union[str, int])
        if origin is Union:
            args = get_args(expected)
            non_none_args = [arg for arg in args if arg is not type(None)]

            # Try each type in the union
            for union_type in non_none_args:
                try:
                    return _deep_validate(name, value, union_type, path)
                except (ValidationError, TypeError):
                    continue

            # MCP-specific fallback: Be permissive for JSON-compatible types
            if isinstance(value, (str, int, float, bool, list, dict)):
                return value

            type_names = [_get_type_name(t) for t in non_none_args]
            raise ValidationError(
                f"value does not match any type in Union[{', '.join(type_names)}]",
                current_path,
                "union_mismatch",
            )

        # Simple type validation - optimized for MCP patterns
        if origin is None:
            if expected is Any:
                return value

            if inspect.isclass(expected):
                # If it's already the right type, accept it
                if isinstance(value, expected):
                    return value

                # MCP-optimized type coercion for common patterns
                if expected in (str, int, float, bool):
                    if expected is str:
                        return str(value)
                    elif expected is int:
                        if isinstance(value, (int, float)):
                            return int(value)
                        elif isinstance(value, str):
                            try:
                                return int(value)
                            except ValueError:
                                # MCP: Accept string IDs, UUIDs, etc.
                                return value
                        else:
                            try:
                                return int(value)
                            except (ValueError, TypeError):
                                return value
                    elif expected is float:
                        if isinstance(value, (int, float)):
                            return float(value)
                        elif isinstance(value, str):
                            try:
                                return float(value)
                            except ValueError:
                                return value
                        else:
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return value
                    elif expected is bool:
                        if isinstance(value, bool):
                            return value
                        elif isinstance(value, str):
                            return value.lower() in ("true", "1", "yes", "on")
                        else:
                            return bool(value)

                # Check if it's a McpPydanticBase subclass
                if hasattr(expected, "__bases__") and any(
                    issubclass(base, McpPydanticBase) for base in expected.__mro__[1:]
                ):
                    if isinstance(value, dict):
                        return expected(**value)
                    elif isinstance(value, expected):
                        return value

                # MCP fallback: Accept compatible types
                return value

            return value

        # List validation - MCP-optimized
        if origin in (list, List):
            if not isinstance(value, list):
                raise ValidationError(
                    "value is not a valid list", current_path, "type_error"
                )

            item_type = get_args(expected)[0] if get_args(expected) else Any
            validated_items = []
            for i, item in enumerate(value):
                if item_type is Any:
                    validated_items.append(item)
                else:
                    try:
                        validated_item = _deep_validate(
                            f"[{i}]", item, item_type, current_path
                        )
                        validated_items.append(validated_item)
                    except ValidationError:
                        # MCP fallback: Include item as-is rather than failing
                        validated_items.append(item)
            return validated_items

        # Dict validation - MCP-optimized
        if origin in (dict, Dict):
            if not isinstance(value, dict):
                raise ValidationError(
                    "value is not a valid dict", current_path, "type_error"
                )

            args = get_args(expected)
            key_type = args[0] if args else Any
            val_type = args[1] if len(args) > 1 else Any

            validated_dict = {}
            for k, v in value.items():
                try:
                    validated_key = (
                        k
                        if key_type is Any
                        else _deep_validate("key", k, key_type, current_path)
                    )
                    validated_value = (
                        v
                        if val_type is Any
                        else _deep_validate(f"[{k}]", v, val_type, current_path)
                    )
                    validated_dict[validated_key] = validated_value
                except ValidationError:
                    # MCP fallback: Allow original values
                    validated_dict[k] = v
            return validated_dict

        # Default: return as-is for unknown complex types
        return value

    class Field:  # type: ignore[no-redef]
        """Enhanced Field class optimized for MCP resource patterns."""

        __slots__ = (
            "default",
            "default_factory",
            "alias",
            "description",
            "title",
            "required",
            "kwargs",
            "json_schema_extra",
        )

        def __init__(
            self,
            default: Any = ...,  # Use ... as sentinel for "not provided"
            default_factory: Optional[Callable[[], Any]] = None,
            alias: Optional[str] = None,
            title: Optional[str] = None,
            description: Optional[str] = None,
            json_schema_extra: Optional[Dict[str, Any]] = None,
            **kwargs,
        ):
            if default is not ... and default_factory is not None:
                raise TypeError("Cannot specify both 'default' and 'default_factory'")

            self.default = default
            self.default_factory = default_factory
            self.alias = alias
            self.title = title
            self.description = description
            self.json_schema_extra = json_schema_extra or {}
            self.kwargs = kwargs

            # Determine if field is required
            self.required = default is ... and default_factory is None

    @dataclass
    class McpPydanticBase:  # type: ignore[no-redef]
        """Enhanced fallback base class optimized for MCP resource management."""

        # Class-level metadata
        __model_fields__: ClassVar[Dict[str, Any]] = {}  # type: ignore[misc]
        __model_required__: ClassVar[Set[str]] = set()
        __field_aliases__: ClassVar[Dict[str, str]] = {}
        __validators__: ClassVar[Dict[str, List[Callable]]] = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

            cls.__model_fields__ = {}
            cls.__model_required__ = set()
            cls.__field_aliases__ = {}
            cls.__validators__ = {}

            # Analyze type hints and class attributes
            try:
                hints = get_type_hints(cls, include_extras=True)
            except (NameError, AttributeError, TypeError):
                # Fall back to raw annotations for forward references
                hints = getattr(cls, "__annotations__", {})

            for name, hint in hints.items():
                if name.startswith("__") and name.endswith("__"):
                    continue

                # Get field definition
                if hasattr(cls, name):
                    attr_val = getattr(cls, name)
                    if isinstance(attr_val, Field):
                        field = attr_val
                    else:
                        field = Field(default=attr_val)
                else:
                    field = Field()

                # Handle alias
                if field.alias:
                    cls.__field_aliases__[name] = field.alias

                # Handle requirements
                if isinstance(hint, str):
                    # Forward reference - be conservative
                    if field.required:
                        cls.__model_required__.add(name)
                else:
                    # Normal type hint processing
                    if field.required and not _is_optional(hint):
                        cls.__model_required__.add(name)

                cls.__model_fields__[name] = field

            # Apply MCP-specific defaults
            cls._apply_mcp_defaults()

        @classmethod
        def _apply_mcp_defaults(cls):
            """Apply MCP-specific field defaults and configurations."""

            # MCP Resource-specific defaults
            if cls.__name__ == "MCPResourceSpec":
                # Make certain fields have better defaults
                if "metadata" not in cls.__model_fields__:
                    cls.__model_fields__["metadata"] = Field(default_factory=dict)
                if "examples" not in cls.__model_fields__:
                    cls.__model_fields__["examples"] = Field(default_factory=list)
                if "dependencies" not in cls.__model_fields__:
                    cls.__model_fields__["dependencies"] = Field(default_factory=list)

                # Remove these from required fields
                cls.__model_required__.discard("metadata")
                cls.__model_required__.discard("examples")
                cls.__model_required__.discard("dependencies")

            # JSON-RPC specific defaults
            if cls.__name__ == "JSONRPCMessage":
                if "jsonrpc" not in cls.__model_fields__:
                    cls.__model_fields__["jsonrpc"] = Field(default="2.0")

                # Make id field more permissive
                if "id" in cls.__model_fields__:
                    cls.__model_fields__["id"] = Field(default=None)

                cls.__model_required__.discard("jsonrpc")
                cls.__model_required__.discard("id")

            # Parameter/Return spec defaults
            if cls.__name__ in ("ParameterSpec", "ReturnSpec"):
                if "examples" not in cls.__model_fields__:
                    cls.__model_fields__["examples"] = Field(default_factory=list)
                cls.__model_required__.discard("examples")

        def __init__(self, **data: Any):
            # Process aliases
            processed_data = self._process_aliases(data)

            # Build field values
            values = self._build_field_values(processed_data)

            # Validate required fields
            self._validate_required_fields(values)

            # Validate types
            self._validate_types(values)

            # Set attributes
            object.__setattr__(self, "__dict__", values)

            # Call post-init hooks
            self._call_post_init_hooks()

        def _process_aliases(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Convert aliased keys to field names."""
            processed = {}
            alias_to_field = {v: k for k, v in self.__class__.__field_aliases__.items()}

            for key, value in data.items():
                field_name = alias_to_field.get(key, key)
                processed[field_name] = value

            return processed

        def _build_field_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Build dictionary of field values with defaults."""
            values = {}

            # Process defined fields
            for name, field in self.__class__.__model_fields__.items():
                if name in data:
                    values[name] = data.pop(name)
                elif field.default_factory is not None:  # type: ignore[attr-defined]
                    values[name] = field.default_factory()  # type: ignore[attr-defined]
                elif field.default is not ...:  # type: ignore[attr-defined]
                    values[name] = field.default  # type: ignore[attr-defined]
                else:
                    values[name] = None

            # Add extra fields (MCP allows extra fields by default)
            values.update(data)

            return values

        def _validate_required_fields(self, values: Dict[str, Any]):
            """Validate that all required fields are present."""
            missing = []
            for name in self.__class__.__model_required__:
                if values.get(name) is None:
                    missing.append(name)

            if missing:
                raise ValidationError(  # type: ignore[call-arg]
                    f"Missing required fields: {', '.join(missing)}",
                    error_type="missing_fields",
                )

        def _validate_types(self, values: Dict[str, Any]):
            """Validate field types with MCP-optimized error handling."""
            try:
                hints = get_type_hints(self.__class__, include_extras=True)
            except (NameError, AttributeError, TypeError):
                # Skip validation for forward references during construction
                return

            for name, expected_type in hints.items():
                if name.startswith("__") and name.endswith("__"):
                    continue

                if name in values:
                    try:
                        validated_value = _deep_validate(
                            name, values[name], expected_type
                        )
                        values[name] = validated_value
                    except ValidationError as e:
                        # For critical errors, still fail
                        if e.error_type in ("missing_fields", "missing"):  # type: ignore[attr-defined]
                            raise
                        # For type coercion issues in MCP fallback mode, be lenient
                        pass

        def _call_post_init_hooks(self):
            """Call post-initialization hooks."""
            # Call __post_init__ if it exists (dataclass style)
            post_init = getattr(self, "__post_init__", None)
            if callable(post_init):
                post_init()

            # Call model_post_init if it exists (Pydantic style)
            model_post_init = getattr(self, "model_post_init", None)
            if callable(model_post_init):
                model_post_init(None)

        def model_dump(
            self,
            *,
            exclude: Optional[Union[Set[str], Dict[str, Any]]] = None,
            exclude_none: bool = False,
            by_alias: bool = False,
            include: Optional[Union[Set[str], Dict[str, Any]]] = None,
            **kwargs,
        ) -> Dict[str, Any]:
            """Enhanced model serialization optimized for MCP."""
            result = {}

            for key, value in self.__dict__.items():
                # Skip private attributes
                if key.startswith("__"):
                    continue

                # Handle include/exclude
                if include and key not in include:
                    continue
                if exclude and self._should_exclude(key, exclude):
                    continue
                if exclude_none and value is None:
                    continue

                # Handle aliases
                output_key = key
                if by_alias and key in self.__class__.__field_aliases__:
                    output_key = self.__class__.__field_aliases__[key]

                # Serialize value with MCP optimizations
                result[output_key] = self._serialize_value(
                    value, exclude, exclude_none, by_alias, include, **kwargs
                )

            return result

        def _serialize_value(
            self, value, exclude, exclude_none, by_alias, include, **kwargs
        ):
            """Serialize a value with MCP-specific optimizations."""
            if hasattr(value, "model_dump"):
                return value.model_dump(
                    exclude=exclude,
                    exclude_none=exclude_none,
                    by_alias=by_alias,
                    include=include,
                    **kwargs,
                )
            elif isinstance(value, list):
                return [
                    self._serialize_value(
                        item, exclude, exclude_none, by_alias, include, **kwargs
                    )
                    for item in value
                ]
            elif isinstance(value, dict):
                return {
                    k: self._serialize_value(
                        v, exclude, exclude_none, by_alias, include, **kwargs
                    )
                    for k, v in value.items()
                }
            else:
                return value

        def _should_exclude(
            self, key: str, exclude: Union[Set[str], Dict[str, Any]]
        ) -> bool:
            """Check if a key should be excluded."""
            if isinstance(exclude, set):
                return key in exclude
            elif isinstance(exclude, dict):
                return key in exclude
            return False

        def model_dump_json(
            self,
            *,
            exclude: Optional[Union[Set[str], Dict[str, Any]]] = None,
            exclude_none: bool = False,
            by_alias: bool = False,
            include: Optional[Union[Set[str], Dict[str, Any]]] = None,
            indent: Optional[int] = None,
            separators: Optional[tuple] = None,
            **kwargs,
        ) -> str:
            """Enhanced JSON serialization with MCP optimizations."""
            data = self.model_dump(
                exclude=exclude,
                exclude_none=exclude_none,
                by_alias=by_alias,
                include=include,
                **kwargs,
            )

            if separators is None:
                separators = (",", ":")

            return json.dumps(data, indent=indent, separators=separators, default=str)

        def model_dump_mcp(self, **kwargs) -> Dict[str, Any]:
            """MCP-optimized model dump with additional processing."""
            data = self.model_dump(**kwargs)
            return self._process_mcp_data(data)

        def _process_mcp_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Process data for MCP compatibility."""
            processed: Dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    processed[key] = self._process_mcp_data(value)
                elif isinstance(value, list):
                    processed[key] = [
                        self._process_mcp_data(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    processed[key] = value
            return processed

        @classmethod
        def model_validate(cls, data: Union[Dict[str, Any], Any]):
            """Enhanced model validation from various input types."""
            if isinstance(data, dict):
                return cls(**data)
            elif isinstance(data, cls):
                return data
            elif hasattr(data, "__dict__"):
                return cls(**data.__dict__)
            elif hasattr(data, "model_dump"):
                return cls(**data.model_dump())
            else:
                raise ValidationError(  # type: ignore[call-arg]
                    f"Cannot validate {type(data)} as {cls.__name__}",
                    error_type="invalid_input",
                )

        # Pydantic v1 compatibility methods
        def json(self, **kwargs) -> str:
            return self.model_dump_json(**kwargs)

        def dict(self, **kwargs) -> Dict[str, Any]:
            return self.model_dump(**kwargs)

    def ConfigDict(**kwargs) -> Dict[str, Any]:  # type: ignore[no-redef]
        """Enhanced configuration dictionary for MCP."""
        defaults = {
            "extra": "allow",  # MCP allows extra fields
            "validate_assignment": True,
            "use_enum_values": True,
            "arbitrary_types_allowed": True,  # Allow custom types in MCP
        }
        defaults.update(kwargs)
        return defaults

    # Dummy decorators for compatibility
    def validator(*args, **kwargs):
        """Dummy validator decorator for fallback mode."""

        def decorator(func):
            return func

        return decorator

    def root_validator(*args, **kwargs):  # type: ignore[no-redef]
        """Dummy root validator decorator for fallback mode."""

        def decorator(func):
            return func

        return decorator
