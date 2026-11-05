# micropython-vt100

VT100 terminal emulator for MicroPython.

## Installation

```sh
mpremote mip install github:gbenson/micropython-vt100/fonts/noto-sans-mono/15
```

## Development

```sh
git clone https://github.com/gbenson/micropython-vt100.git
cd micropython-vt100
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Font harvesting

micropython-vt100 uses bitmapped fonts in its own special format.
You can use the `makefont` script to generate these from any font
supported by [Pillow](https://pillow.readthedocs.io/), e.g. the
command `makefont fonts/noto-sans-mono/15` will:

 - Read `fonts/noto-sans-mono/15/manifest.json`
 - Load the font the manifest specified
 - Render the glyphs the manifest specified
 - Write `fonts/noto-sans-mono/15/noto-sans-mono-15.ff`

So, to add a new font:

 - Make the directory
 - Copy the license (make sure its appropriate!)
 - Create the manifest (use an existing one as a template)
 - Run `makefont`
