# Design Principles

## Core Philosophy

CHUK MCP Math follows these fundamental design principles to ensure consistency, performance, and usability across all mathematical domains.

## 1. Async-Native Architecture

**Every function is asynchronous by design**
- All functions use `async def` syntax
- Returns awaitable results for optimal concurrency
- Strategic yielding with `await asyncio.sleep(0)` for long operations
- Enables parallel execution of independent calculations

```python
# Good - Async-native
async def calculate(x: float) -> float:
    if x > 1000:
        await asyncio.sleep(0)  # Yield control for long operations
    return complex_calculation(x)

# Bad - Synchronous wrapper
def calculate_sync(x: float) -> float:
    return asyncio.run(calculate(x))  # Avoid this pattern
```

## 2. MCP Protocol Integration

**Standardized function decoration for AI model compatibility**
- Every mathematical function uses `@mcp_function()` decorator
- Provides metadata for AI models to understand function capabilities
- Enables smart caching, performance tracking, and error handling
- Supports streaming results for large computations

## 3. Type Safety First

**Complete type annotations with runtime validation**
- Full type hints on all function signatures
- Pydantic models for complex input/output structures
- Runtime validation of inputs to prevent errors
- Clear error messages for type mismatches

```python
from typing import Union, List, Optional

async def function(
    required: float,
    optional: Optional[int] = None,
    numbers: List[Union[int, float]] = None
) -> Dict[str, Any]:
```

## 4. Mathematical Accuracy

**Correctness over speed when trade-offs exist**
- Use proven algorithms from academic literature
- Include numerical stability considerations
- Document accuracy limitations and edge cases
- Provide references to mathematical sources

## 5. Educational Transparency

**Functions should teach as well as compute**
- Clear docstrings with mathematical formulas
- Step-by-step explanations available
- Rich examples demonstrating usage
- Visual representations where applicable

## 6. Performance Optimization

**Smart optimization without sacrificing clarity**
- Caching strategies appropriate to function characteristics
- Concurrency control to prevent resource exhaustion
- Batch processing for vectorizable operations
- Memory-efficient algorithms for large-scale computations

## 7. Error Handling Excellence

**Fail gracefully with helpful feedback**
- Validate inputs before computation
- Provide specific error messages
- Suggest corrections when possible
- Never silently fail or return incorrect results

```python
if not 0 <= probability <= 1:
    raise ValueError(
        f"Probability must be between 0 and 1, got {probability}. "
        "Did you mean to divide by 100?"
    )
```

## 8. Modular Organization

**Logical grouping by mathematical domain**
- Related functions grouped in focused modules
- Clear namespace hierarchy
- Minimal inter-module dependencies
- Easy to understand and navigate

## 9. Consistency Across Domains

**Uniform patterns regardless of mathematical area**
- Same decorator usage patterns
- Consistent parameter naming conventions
- Standardized return value structures
- Uniform error handling approaches

## 10. Progressive Complexity

**Simple cases should be simple**
- Basic usage requires minimal parameters
- Advanced features available through optional parameters
- Sensible defaults for common use cases
- Complexity revealed gradually

```python
# Simple case
result = await calculate(10)

# Advanced usage
result = await calculate(
    10,
    method="advanced",
    precision=1e-10,
    max_iterations=1000,
    return_diagnostics=True
)
```

## 11. Testability by Design

**Every function must be easily testable**
- Pure functions without side effects
- Deterministic results (except for random functions with seeds)
- Property-based testing support
- Comprehensive test coverage expected

**Testing Resources:**
- [Testing Guide](./docs/testing/TESTING.md) - Complete testing documentation
- [Math Testing Patterns](./docs/testing/MATH_PATTERNS.md) - Numerical precision patterns
- [Test Templates](./docs/testing/templates/) - Ready-to-use templates

## 12. Documentation as Code

**Documentation is part of the implementation**
- Docstrings are required, not optional
- Examples in docstrings must be executable
- Mathematical formulas in LaTeX notation
- References to academic sources

## 13. Backwards Compatibility

**Respect existing interfaces**
- New parameters should be optional with defaults
- Deprecation warnings before removing features
- Version migration guides when breaking changes necessary
- Semantic versioning strictly followed

## 14. Security Consciousness

**Safe for untrusted input**
- Resource limits on expensive operations
- Protection against DOS through input validation
- No arbitrary code execution
- Sandboxed execution environment

## 15. Accessibility

**Usable by developers of all skill levels**
- Intuitive function names
- Common mathematical notation in parameters
- Helpful error messages for beginners
- Advanced features don't complicate basic usage

## Application of Principles

When implementing new functions or modules:

1. **Start with the mathematical definition** - Understand the math before coding
2. **Design the interface first** - How will users interact with this function?
3. **Consider the common case** - Optimize for typical usage patterns
4. **Document while implementing** - Not after
5. **Test edge cases explicitly** - Don't assume correctness
6. **Review for consistency** - Does it feel like part of the library?
7. **Optimize if necessary** - But not prematurely

## Principle Conflicts

When principles conflict, prioritize in this order:

1. **Correctness** - Never compromise on mathematical accuracy
2. **Safety** - Protect against misuse and errors
3. **Usability** - Make it easy to use correctly
4. **Performance** - Optimize without sacrificing above
5. **Features** - Add capabilities that align with core purpose

## Evolution of Principles

These principles are living guidelines that evolve with the library:
- Regular review in major version planning
- Community feedback incorporated
- New principles added as patterns emerge
- Existing principles refined based on experience