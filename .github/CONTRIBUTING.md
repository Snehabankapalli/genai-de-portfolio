# Contributing Guidelines

## Development Setup

```bash
git clone https://github.com/Snehabankapalli/genai-de-portfolio
cd genai-de-portfolio/project-1-rag-pipeline

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install flake8 black pytest pytest-cov pytest-mock

cp .env.example .env
# Add your API keys — never hardcode
```

## Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Write implementation + tests
4. Run quality checks
5. Commit: `git commit -m 'feat: add your feature'`
6. Open a Pull Request

## Quality Standards

```bash
black .
flake8 . --max-line-length=100 --exclude=.venv,venv
pytest tests/ -v --cov=src --cov-fail-under=80
```

## Commit Format

`<type>: <description>` — types: `feat`, `fix`, `refactor`, `docs`, `test`

Questions? Open an issue or contact [@Snehabankapalli](https://github.com/Snehabankapalli)
