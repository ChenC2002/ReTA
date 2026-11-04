"""Build the Pages/offline viewer from the same frontend used by FastAPI."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='build/site/demo')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    names = {}
    for filename, source in {
        'style.css': repository/'reta/demo/static/style.css',
        'app.js': repository/'reta/demo/static/app.js',
        'data.js': repository/'demo/data.js',
    }.items():
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()[:12]
        names[filename] = f'{source.stem}.{digest}{source.suffix}'
        (destination/names[filename]).write_bytes(content)
    # Content-addressed assets keep a new page from using an older cached script.
    page = (repository/'reta/demo/static/index.html').read_text(encoding='utf-8')
    page = page.replace('./style.css', './'+names['style.css']).replace(
        '<script src="./app.js" defer></script>',
        f'<script src="./{names["data.js"]}"></script>\n  '
        f'<script src="./{names["app.js"]}" defer></script>')
    (destination/'index.html').write_text(page, encoding='utf-8')
    (destination.parent/'.nojekyll').touch()
    print(destination)


if __name__ == '__main__':
    main()
