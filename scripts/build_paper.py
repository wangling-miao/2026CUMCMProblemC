"""Compile the repository paper with XeLaTeX and Biber."""
from pathlib import Path
import shutil
import subprocess

def main():
    paper = Path(__file__).resolve().parents[1] / "paper"
    commands = [
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["biber", "main"],
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    for command in commands:
        if not shutil.which(command[0]):
            raise SystemExit(f"Required executable missing from PATH: {command[0]}")
        subprocess.run(command, cwd=paper, check=True)

if __name__ == "__main__":
    main()
