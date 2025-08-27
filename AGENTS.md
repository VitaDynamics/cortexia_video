# AI Code Agent Instructions

## Build/Lint/Test Commands
- **Run single test**: `pytest tests/test_file.py::TestClass::test_method -v`
- **Run all tests**: `pytest tests/ -v`
- **Run with coverage**: `pytest --cov=cortexia`
- **Build package**: `python -m build`
- **Install in development**: `pip install -e .`

## Code Style Guidelines
- **Python**: 3.10+ required, use type hints extensively
- **Imports**: Use absolute imports, group (stdlib, third-party, local)
- **Types**: Use type hints consistently, prefer `Optional[T]` over `Union[T, None]`
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use custom exceptions from `cortexia.api.exceptions`, validate with Pydantic
- **Formatting**: Follow PEP 8, 4-space indentation, 88-100 char line length
- **Testing**: Use pytest with Mock classes, test both happy path and error cases
- **Documentation**: Docstrings for all public methods/classes, include Args/Returns

## Core Coding Philosophy 
1. **Good Taste**: Redesign to eliminate special cases rather than patching with conditions.
2. **Never Break Userspace**: Any change that disrupts existing behavior is a bug.
3. **Pragmatism with Rigor**: Solve only real, demonstrated problems with data-driven decisions.
4. **Simplicity Obsession**: Functions must be small, focused, and clear.
5. **Minimal Change Discipline**: Only change what's necessary, preserve existing structure.