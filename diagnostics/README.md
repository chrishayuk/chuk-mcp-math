# CHUK MCP Math - Diagnostics

This directory contains diagnostic scripts for troubleshooting and verifying the chuk_mcp_math package structure.

## Files

### diagnose_reorganized_structure.py

**Purpose**: Verifies the reorganized package structure and import paths

**What it does**:
- Checks the directory structure of the package
- Tests imports from various submodules
- Verifies that the reorganized structure is working correctly
- Reports on available modules and functions

**When to use**:
- After making structural changes to the package
- When troubleshooting import errors
- To verify the package is properly installed

**How to run**:
```bash
python diagnostics/diagnose_reorganized_structure.py
```

**Expected output**:
- Shows the project directory structure
- Lists available attributes in each module
- Reports successful imports and function calls
- Indicates which parts of the reorganized structure are working

### simple_diagnostic.py

**Purpose**: Basic diagnostic for project structure and Python path

**What it does**:
- Displays the current directory structure
- Shows contents of key directories (src/, src/chuk_mcp_math/)
- Tests basic imports
- Provides recommendations for common issues

**When to use**:
- Initial troubleshooting of import problems
- Verifying the project structure is correct
- Checking if the package is properly installed
- Debugging Python path issues

**How to run**:
```bash
python diagnostics/simple_diagnostic.py
```

**Expected output**:
- Directory structure tree
- Contents of src/ and src/chuk_mcp_math/
- Python path information
- Import test results
- Recommendations if issues are found

## Common Use Cases

### 1. Import Errors
If you're experiencing import errors, run both diagnostics in order:
```bash
python diagnostics/simple_diagnostic.py
python diagnostics/diagnose_reorganized_structure.py
```

### 2. After Package Reorganization
After restructuring the package, verify everything works:
```bash
python diagnostics/diagnose_reorganized_structure.py
```

### 3. New Development Environment
When setting up a new development environment:
```bash
python diagnostics/simple_diagnostic.py
```

## Interpreting Results

### Success Indicators
- All checkmarks (checkmark symbol) indicate working functionality
- "Successfully imported" messages
- Function calls returning expected results

### Failure Indicators
- X marks (cross symbol) indicate problems
- "ImportError" or "ModuleNotFoundError" messages
- "Could not find" or "No module" messages

## Troubleshooting Tips

If diagnostics fail:

1. **Check your working directory**: Run from the project root
2. **Verify package installation**: Run `pip install -e .` or `uv sync`
3. **Check Python path**: The src/ directory should be in sys.path
4. **Verify file structure**: Ensure all __init__.py files exist
5. **Check for syntax errors**: Run `python -m py_compile` on modified files

## Removed Files

The following diagnostic files were removed as they were obsolete:

- `async_arithmetic_diagnostics.py` - Failed due to outdated imports
- `debug_trig.py` - Failed due to incorrect directory structure assumptions
- `inspect_function_types.py` - Failed due to outdated import paths
- `reorganization_success_demo.py` - Failed due to incorrect module structure assumptions

These files were testing features that no longer exist or import patterns that have changed.
