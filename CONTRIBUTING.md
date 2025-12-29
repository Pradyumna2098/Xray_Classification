# Contributing to X-Ray Classification

Thank you for your interest in contributing to the X-Ray Classification project! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include relevant details**:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Screenshots or logs if applicable

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Xray_Classification.git
   cd Xray_Classification
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Make your changes**
   - Write clean, modular code
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Keep functions focused and small

5. **Write tests**
   - Add unit tests for new functionality
   - Ensure existing tests pass
   - Aim for >80% code coverage
   ```bash
   pytest --cov=src --cov=api
   ```

6. **Run code quality checks**
   ```bash
   make lint  # or manually run flake8
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

8. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Provide a clear description of your changes
   - Reference any related issues

## 📝 Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and returns
- Maximum line length: 100 characters
- Use meaningful variable and function names

Example:
```python
def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = (150, 150),
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess an X-ray image for model inference.
    
    Args:
        image: Input PIL Image
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed numpy array
    """
    # Implementation
    pass
```

### Documentation

- All functions should have docstrings
- Use Google-style docstrings
- Document parameters, return values, and exceptions
- Add inline comments for complex logic

### Testing

- Write tests for all new features
- Use pytest for testing
- Mock external dependencies
- Test edge cases and error conditions

Example test structure:
```python
class TestFeatureName:
    """Tests for feature_name functionality."""
    
    def test_basic_functionality(self):
        """Test basic feature behavior."""
        # Arrange
        input_data = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

## 🏗️ Project Structure

When adding new features, follow the existing structure:

```
Xray_Classification/
├── api/                    # FastAPI application code
├── src/                    # Core source code
│   ├── models/            # Model-related code
│   ├── utils/             # Utility functions
│   └── evaluation/        # Evaluation metrics
├── tests/                 # Test suite
├── k8s/                   # Kubernetes manifests
├── monitoring/            # Prometheus/Grafana configs
└── examples/              # Example scripts
```

## 🧪 Testing Requirements

Before submitting a PR:

1. **All tests must pass**
   ```bash
   pytest -v
   ```

2. **Maintain code coverage above 80%**
   ```bash
   pytest --cov=src --cov=api --cov-report=term-missing
   ```

3. **Test in Docker** (if changing API or deployment)
   ```bash
   docker-compose up --build
   # Test endpoints manually or with automated tests
   docker-compose down
   ```

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass locally
- [ ] New features have corresponding tests
- [ ] Code coverage is maintained (>80%)
- [ ] Documentation is updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes
- [ ] Related issues are referenced in PR

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- Additional model architectures (ResNet, EfficientNet, etc.)
- Performance optimizations for inference
- More comprehensive evaluation metrics
- Additional explainability techniques (SHAP, LIME)

### Medium Priority
- Frontend UI for the API
- Additional deployment examples (AWS, GCP, Azure)
- Performance benchmarking suite
- Data augmentation improvements

### Documentation
- More usage examples
- Video tutorials
- Architecture diagrams
- API integration examples in different languages

### Testing
- Load testing scripts
- Integration tests for deployment
- Security testing

## 🐛 Bug Reports

Good bug reports should include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal steps to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Python version, Docker version, etc.
6. **Logs**: Relevant error messages or logs

## 💡 Feature Requests

Good feature requests should include:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you considered
4. **Additional context**: Any relevant examples or references

## 📞 Getting Help

If you need help:

- **Documentation**: Check [README.md](README.md) and [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Thank you for contributing to making this project better! Your time and effort are greatly appreciated.

---

**Questions?** Open an issue or start a discussion on GitHub.
