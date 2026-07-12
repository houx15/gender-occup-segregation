# Bundled fonts

## NotoSansCJKsc-Regular.otf

Simplified-Chinese Noto Sans CJK (Regular weight), bundled so figures render
Chinese labels identically on the Slurm cluster, in CI, and on macOS — without
depending on system-installed fonts. `scripts/visualize.py:_configure_fonts`
registers this file first for any `language: zh` profile.

- **Source:** https://github.com/notofonts/noto-cjk (`Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf`)
- **License:** SIL Open Font License, Version 1.1 (https://openfontlicense.org)
- **Copyright:** © 2014–2021 Adobe (https://github.com/adobe-fonts), Google.

The OFL permits bundling and redistribution with software; the font is not sold
on its own and retains its reserved font name.
