"""Build the Pages/offline viewer from the same frontend used by FastAPI."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='build/site/demo')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ('index.html', 'style.css', 'app.js'):
        shutil.copyfile(repository/'reta/demo/static'/filename, destination/filename)
    shutil.copyfile(repository/'demo/data.js', destination/'data.js')
    page = destination/'index.html'
    page.write_text(page.read_text(encoding='utf-8').replace(
        '<script src="./app.js" defer></script>',
        '<script src="./data.js"></script>\n  <script src="./app.js" defer></script>'), encoding='utf-8')
    (destination.parent/'.nojekyll').touch()
    print(destination)


if __name__ == '__main__':
    main()
