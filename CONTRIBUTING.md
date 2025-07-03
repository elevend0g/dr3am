# Contributing to dr3am

We love your input! We want to make contributing to dr3am as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/dr3am.git
cd dr3am

# Set up development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Start development server
uvicorn dr3am.main:app --reload
```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **bandit** for security scanning

Run all checks:
```bash
# Format code
black dr3am/ tests/
isort dr3am/ tests/

# Lint
flake8 dr3am/ tests/
mypy dr3am/

# Security scan
bandit -r dr3am/
```

## Testing

- Write tests for any new functionality
- Aim for >90% test coverage
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dr3am --cov-report=html

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m api
```

## Documentation

- Update docstrings for any new functions/classes
- Use Google-style docstrings
- Update README.md for significant changes
- Update OpenAPI documentation for API changes

## Commit Messages

We follow conventional commits:

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding/updating tests
- `chore:` maintenance tasks

Example: `feat: add autonomous research trigger endpoint`

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by opening a new issue.

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We welcome feature requests! Please:

1. Check existing issues first
2. Provide clear use case
3. Explain why this feature would benefit users
4. Consider implementation complexity

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Project maintainers are responsible for clarifying standards and are expected to take appropriate action in response to any instances of unacceptable behavior.

## Questions?

Feel free to ask questions by:
- Opening a GitHub issue
- Starting a discussion
- Reaching out to maintainers

Thank you for contributing! 🚀