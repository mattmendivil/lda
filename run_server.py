import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    server_py = repo_root / "model_server/server.py"
    os.execv(sys.executable, [sys.executable, str(server_py)])


if __name__ == "__main__":
    main()
