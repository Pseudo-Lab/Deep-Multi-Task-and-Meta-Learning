from pathlib import Path
import subprocess
import shutil
import sys


def main(all: bool = False):
    __dirname = Path(__file__).parent
    if (__dirname / "docs").exists() and (__dirname / "build").exists():
        shutil.move(__dirname / "docs", __dirname / "_build")
    else:
        shutil.rmtree(__dirname / "docs", ignore_errors=True)
        shutil.rmtree(__dirname / "_build", ignore_errors=True)
    subprocess.run("jb build .", shell=True)
    shutil.move(__dirname / "_build" / "html", __dirname / "docs")


if __name__ == "__main__":
    all = "--all" in sys.argv
    main(all)
