from pathlib import Path

def version():
    v = Path(__file__).parents[1] / "VERSION"
    return v.read_text().strip()