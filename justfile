lint:
    torchfix src/
    ruff check --fix src/
    ruff format src/
    pyrefly check src/
