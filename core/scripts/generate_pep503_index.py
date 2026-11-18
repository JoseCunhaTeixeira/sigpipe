"""Generate or update the static PEP 503 "simple" index of sigpipe-core on GitHub Pages, as
santiludo's script does for its own.

Run from CI after a release's files are uploaded as GitHub Release assets: merges the release's
wheels and source distribution into the index already on the gh-pages branch (if any), so that
the index holds every release, then writes the two static index.html files uv and pip read.

Usage: python generate_pep503_index.py <repo> <tag> <dist_dir> <existing_index_dir> <out_dir>
"""

import hashlib
import html
import re
import sys
from pathlib import Path

PROJECT = "sigpipe-core"


def main() -> None:
    repo, tag, dist_dir, existing_index_dir, out_dir = sys.argv[1:6]

    links: dict[str, str] = {}
    existing_index = Path(existing_index_dir) / "simple" / PROJECT / "index.html"
    if existing_index.exists():
        for match in re.finditer(
            r'href="([^"]+)"[^>]*>([^<]+)</a>', existing_index.read_text(encoding="utf-8")
        ):
            href, name = match.groups()
            links[name] = href

    files = [*Path(dist_dir).glob("*.whl"), *Path(dist_dir).glob("*.tar.gz")]
    for path in sorted(files):
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        url = f"https://github.com/{repo}/releases/download/{tag}/{path.name}"
        links[path.name] = f"{url}#sha256={sha256}"

    project_dir = Path(out_dir) / "simple" / PROJECT
    project_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f'    <a href="{html.escape(href)}">{html.escape(name)}</a><br/>'
        for name, href in sorted(links.items())
    )
    (project_dir / "index.html").write_text(
        f"<!DOCTYPE html>\n<html>\n<body>\n{rows}\n</body>\n</html>\n", encoding="utf-8"
    )
    (Path(out_dir) / "simple" / "index.html").write_text(
        f'<!DOCTYPE html>\n<html>\n<body>\n    <a href="{PROJECT}/">{PROJECT}</a><br/>\n'
        "</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"Wrote the index of {len(links)} file(s) to {out_dir}/simple/")


if __name__ == "__main__":
    main()
