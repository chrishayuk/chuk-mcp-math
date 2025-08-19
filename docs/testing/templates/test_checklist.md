# Test Checklist Template

Use this checklist when creating tests for new modules to ensure comprehensive coverage.

## Module: _______________

### ✅ Core Functionality
- [ ] Happy path - normal expected inputs
- [ ] All function parameters tested
- [ ] Optional parameters with defaults
- [ ] Return values match expected types
- [ ] Return values match expected ranges/constraints

### ✅ Edge Cases
- [ ] Empty inputs ([], "", 0, None as appropriate)
- [ ] Single element inputs
- [ ] Maximum reasonable size inputs
- [ ] Minimum valid inputs
- [ ] Boundary values (0, 1, -1, limits)

### ✅ Error Handling
- [ ] Invalid input types raise TypeError
- [ ] Invalid input values raise ValueError
- [ ] Error messages are descriptive
- [ ] All error paths tested
- [ ] Graceful degradation where appropriate

### ✅ Type Handling
- [ ] Integer inputs
- [ ] Float inputs
- [ ] Mixed numeric types
- [ ] Return type consistency
- [ ] Input immutability (inputs not modified)

### ✅ Performance Considerations
- [ ] Large input handling (mark with @pytest.mark.slow if needed)
- [ ] Async yielding for long operations (>1000 elements)
- [ ] Memory efficiency for large datasets
- [ ] Concurrent operation safety

### ✅ Mathematical Properties (if applicable)
- [ ] Commutative property (if relevant)
- [ ] Associative property (if relevant)
- [ ] Identity elements
- [ ] Inverse operations
- [ ] Known mathematical relationships

### ✅ Numerical Stability (if applicable)
- [ ] Very large values (near overflow)
- [ ] Very small values (near underflow)
- [ ] Mixed magnitude inputs
- [ ] Precision preservation
- [ ] Special float values (inf, -inf, nan)

### ✅ Integration
- [ ] Works with related functions
- [ ] Maintains consistency across module
- [ ] MCP decorator behavior (caching, metrics)

### ✅ Test Quality
- [ ] Clear test names describe what's being tested
- [ ] Tests are independent (no shared state)
- [ ] Appropriate assertions with tolerances
- [ ] Good test data organization
- [ ] Comments explain complex test logic

## Notes
_Add any module-specific testing considerations here_

## Coverage Target
- [ ] Line coverage > 90%
- [ ] Branch coverage > 85%
- [ ] All public functions tested
- [ ] All error conditions tested