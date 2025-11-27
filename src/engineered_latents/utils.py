from pathlib import Path


def get_project_root():
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "pyproject.toml").exists():
            return path
        path = path.parent
    raise FileNotFoundError("No pyproject.toml found")


OUTPUT_DIR = get_project_root() / "out"
