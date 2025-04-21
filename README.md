# release.py

Create GitHub releases based on a `.version` file:
```
VERSION_MAJOR=0
VERSION_MINOR=3
VERSION_PATCH=0
VERSION_PRERELEASE=
```

## Example workflow
```yaml
name: Release
on:
  push:
    ignore-tags:
      - "releases/**"

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          fetch-tags: true

      - name: Install release.py
        run: pipx install .

      - run: release release.py
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```
