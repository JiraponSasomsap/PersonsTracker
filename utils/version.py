from pathlib import Path

def version():
    v = Path(__file__).parents[1] / "VERSION"
    print(v)
    return v.read_text().strip()