from __future__ import annotations

import struct

from argparse import ArgumentParser
from collections import namedtuple
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated, IO, Literal, Optional, Self

from PIL import Image, ImageDraw, ImageFont
from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveFloat,
    StringConstraints,
    conint,
)


# File format:
#  - 1 FontHeader: magic, W, H, N, M
#  - N glyphs each of H uint32 rows
#  - M range groups
class FontHeader(BaseModel):
    magic_number: Literal[0xb2dc3eb9] = 0xb2dc3eb9  # uint32
    glyph_width: conint(gt=0, le=10)                # uint8
    glyph_height: conint(gt=0, lt=256)              # uint8
    num_glyphs: conint(gt=0, lt=65536)              # uint16
    num_range_groups: conint(gt=0, lt=65536)        # uint16

    def write(self, fp: IO[bytes]) -> None:
        fp.write(struct.pack(
            "<IBBHH",
            *(getattr(self, field) for field in self.model_fields),
        ))


REPLACEMENT_CHARACTER = 0xFFFD


class Manifest(BaseModel):
    name: Optional[Annotated[str, StringConstraints(min_length=1)]] = None
    size: Optional[PositiveFloat] = None
    path: Path
    adjust_min_x: int = 0
    adjust_min_y: int = 0
    adjust_max_x: int = 0
    adjust_max_y: int = 0
    replacement_character: int = REPLACEMENT_CHARACTER
    blocks: list[UnicodeBlock]

    @property
    def codepoints(self) -> Iterable[int]:
        for _, block in sorted((b.start, b) for b in self.blocks):
            yield from block.codepoints


class CodepointRange(BaseModel):
    start: NonNegativeInt
    limit: NonNegativeInt

    @property
    def codepoints(self) -> Iterable[int]:
        return range(self.start, self.limit)


class UnicodeBlock(CodepointRange):
    name: Annotated[str, StringConstraints(min_length=1)]
    include_only: Optional[Codepoints] = None

    @property
    def codepoints(self) -> Iterable[int]:
        return (
            self.include_only
            if self.include_only is not None
            else super()).codepoints


class Codepoints(BaseModel):
    range: Optional[CodepointRange] = None
    glyphs: Optional[list[NonNegativeInt]] = None

    @property
    def codepoints(self) -> Iterable[int]:
        if self.range is not None:
            assert self.glyphs is None
            return self.range.codepoints
        else:
            assert self.glyphs is not None
            return sorted(self.glyphs)


Mapping = namedtuple("Mapping", ("codepoint", "glyph_index"))


class _MappedRange:
    def __str__(self) -> str:
        return (
            f" U+{self.start.codepoint:04X}"
            f"        "
            f" => "
            f" {self.start.glyph_index}"

        ) if self.length == 1 else (

            f"[U+{self.start.codepoint:04X},"
            f"U+{self.limit.codepoint - 1:04X}]"
            f" => "
            f"[{self.start.glyph_index},"
            f"{self.limit.glyph_index - 1}]"
            f" ({self.length} mappings)"
        )


@dataclass(kw_only=True)
class MappedRange(_MappedRange):
    start: Mapping
    length: int = 1

    @property
    def limit(self) -> Mapping:
        return Mapping(*(v + self.length for v in self.start))


@dataclass(kw_only=True)
class RangeGroup(_MappedRange):
    ranges: list[MappedRange] = field(default_factory=list)

    @property
    def start(self) -> MappedRange:
        return self.ranges[0].start

    @property
    def limit(self) -> MappedRange:
        return self.ranges[-1].limit

    @property
    def length(self) -> int:
        return self.limit.codepoint - self.start.codepoint

    def append(self, r: MappedRange) -> None:
        self.ranges.append(r)

    def write(self, fp: IO[bytes]) -> None:
        num_slots = self.length
        assert 1 <= num_slots < 256
        fp.write(struct.pack("<HB", self.start.codepoint, num_slots))
        fp.write(self.bytecode)

    @property
    def bytecode(self) -> bytes:
        return bytearray(self._bytecode)

    @property
    def _bytecode(self) -> Iterable[int]:
        prev = None
        glyph_index = None
        for curr in self.ranges:
            if prev is not None:
                skip = curr.start.codepoint - prev.limit.codepoint
                if skip > 0:
                    yield 256 - skip  # -128..-1 -> skip 128..1
            prev = curr

            if glyph_index != curr.start.glyph_index:
                glyph_index = curr.start.glyph_index
                yield from struct.pack("<bH", 0, glyph_index)
            yield curr.length  # 1..127 -> step 1..127
            glyph_index += curr.length


@dataclass(kw_only=True)
class FontBuilder:
    manifest_path: Path
    _glyphs: list[bytes] = field(default_factory=list)
    _glyphs_by_codepoint: dict[int, int] = field(default_factory=dict)
    _glyphs_by_pixeldata: dict[bytes, int] = field(default_factory=dict)

    @classmethod
    def build(cls, path: str | Path) -> Self:
        manifest_path = path if isinstance(path, Path) else Path(path)
        if manifest_path.is_dir():
            manifest_path /= "manifest.json"
        return cls(manifest_path=manifest_path)

    @cached_property
    def manifest(self) -> Manifest:
        manifest_json = self.manifest_path.read_text()
        return Manifest.model_validate_json(manifest_json)

    @cached_property
    def path(self) -> Path:
        return self.manifest_path.parent / self.manifest.path

    @cached_property
    def name(self) -> str:
        if (name := self.manifest.name) is not None:
            return name
        name = self.manifest_path.parent.parent.name
        return name.replace("-", " ").title()

    @cached_property
    def size(self) -> float:
        if (size := self.manifest.size) is not None:
            return size
        return float(self.manifest_path.parent.name)

    @cached_property
    def font(self) -> ImageFont.FreeTypeFont:
        font = ImageFont.truetype(font=self.path, size=self.size)
        try:
            same_file = self.path.samefile(font.path)
        except FileNotFoundError:
            same_file = False
        if not same_file:
            print("\x1B[41;1;33mWARNING:\x1B[0m supplied font",
                  font.path, "!= requested font", self.path)
        return font

    @property
    def glyph_width(self) -> int:
        return self.glyph_size[0]

    @property
    def glyph_height(self) -> int:
        return self.glyph_size[1]

    @cached_property
    def glyph_size(self) -> tuple[int, int]:
        return self._glyph_size_offset[0]

    @cached_property
    def glyph_offset(self) -> tuple[int, int]:
        return self._glyph_size_offset[1]

    @cached_property
    def _glyph_size_offset(self) -> tuple[tuple[int, int], tuple[int, int]]:
        boxes = [self.font.getbbox(chr(c)) for c in range(32, 128)]

        min_x = min(box[0] for box in boxes) + self.manifest.adjust_min_x
        min_y = min(box[1] for box in boxes) + self.manifest.adjust_min_y
        max_x = max(box[2] for box in boxes) + self.manifest.adjust_max_x
        max_y = max(box[3] for box in boxes) + self.manifest.adjust_max_y

        return ((max_x - min_x + 1, max_y - min_y + 1), (-min_x, -min_y))

    @cached_property
    def glyphs(self) -> list[bytes]:
        tofu_glyph_index = self._get_glyph_index(self._rendered_tofu)
        if tofu_glyph_index != 0:
            raise ValueError(f"glyph 0 not U+{REPLACEMENT_CHARACTER:04X}")
        for codepoint in self.manifest.codepoints:
            self._maybe_add_glyph(codepoint)
        check = self._glyphs_by_codepoint.pop(REPLACEMENT_CHARACTER, None)
        assert check is None or check == tofu_glyph_index
        return self._glyphs

    def _maybe_add_glyph(self, codepoint: int) -> None:
        if codepoint == 0x0020:  # space
            rendered = self._rendered_space
        elif codepoint == REPLACEMENT_CHARACTER:
            rendered = self._rendered_tofu
        else:
            rendered = self._render(chr(codepoint))
            if rendered is None or rendered in self._rendered_non_glyphs:
                # don't add -> render as tofu
                # N.B. whitespace except U+0020 are included here
                # (i.e. so they'll render as tofu), since I see code
                # should translate in much the same way as e[mn]-dash
                # would be transformed to '--' or whatever.  Can add
                # them later if this turns out to be a mistake.
                return

        glyph_index = self._get_glyph_index(rendered)
        self._glyphs_by_codepoint[codepoint] = glyph_index

    def _get_glyph_index(self, rendered: bytes) -> int:
        if (index := self._glyphs_by_pixeldata.get(rendered)) is not None:
            return index

        index = len(self._glyphs)
        self._glyphs.append(rendered)
        self._glyphs_by_pixeldata[rendered] = index
        return index

    @cached_property
    def _rendered_space(self) -> bytes:
        return self._render(" ")

    @cached_property
    def _rendered_non_glyph(self) -> bytes:
        return self._render("\u034f")  # combining grapheme joiner

    @cached_property
    def _rendered_tofu(self) -> bytes:
        result = self._render(
            chr(self.manifest.replacement_character),
            allow_unprintable=True,
        )
        assert result is not None
        assert any(result)
        return result

    @cached_property
    def _rendered_non_glyphs(self) -> set[bytes]:
        return set((
            self._rendered_space,
            self._rendered_non_glyph,
            self._rendered_tofu,
        ))

    def _render(
            self,
            c: str,
            *,
            allow_unprintable: bool = False,
    ) -> Optional[bytes]:
        if not c.isprintable() and not allow_unprintable:
            return None
        img = Image.new("L", self._scratch_size)
        draw = ImageDraw.Draw(img)
        draw.text(self._scratch_offset, c, fill=255, font=self.font)

        cropbox = self._scratch_crop
        if (glyphbox := img.getbbox()) is not None:
            cropbox = self._fit_cropbox(cropbox, glyphbox)
            if cropbox is None:
                return None

        img = img.crop(cropbox)
        assert img.size == self.glyph_size
        assert img.width <= (32 // 3)  # 3 bits/pixel // 32 bits/row

        pixels = bytes(img.get_flattened_data())
        pixel_iter = iter(pixels)
        rows = []
        for y in range(img.height):
            rowpix = [next(pixel_iter) >> 6 for _ in range(img.width)]
            row = 0
            for rp in reversed(rowpix):
                row <<= 2
                row |= rp
            rows.append(row)

        return b"".join(struct.pack("<I", row) for row in rows)

    @cached_property
    def _scratch_size(self) -> tuple[int, int]:
        return tuple(d * 5 for d in self.glyph_size)

    @cached_property
    def _scratch_crop(self) -> tuple[int, int, int, int]:
        gw, gh = self.glyph_size
        sw, sh = self._scratch_size
        lx = (sw - gw) // 2
        ty = (sh - gh) // 2
        return (lx, ty, lx + gw, ty + gh)

    @cached_property
    def _scratch_offset(self) -> tuple[int, int]:
        sx, sy = self._scratch_crop[:2]
        ox, oy = self.glyph_offset
        return sx + ox, sy + oy

    @staticmethod
    def _fit_cropbox(
            cropbox: tuple[int, int, int, int],
            glyphbox: tuple[int, int, int, int],
    ) -> Optional[tuple[int, int, int, int]]:
        cl, ct, cr, cb = cropbox
        gl, gt, gr, gb = glyphbox
        if gl >= cl and gt >= ct and gr <= cr and gb <= cb:
            return cropbox  # glyphbox entirely within cropbox

        # Try nudging the glyph into the cropbox.
        x_adj = max(cl - gl, 0) + min(cr - gr, 0)
        y_adj = max(ct - gt, 0) + min(cb - gb, 0)

        cl -= x_adj
        ct -= y_adj
        cr -= x_adj
        cb -= y_adj

        # Permit up to 1 pixel overflow in each direction.
        if any(overflow > 1 for overflow in (cl - gl, ct - gt, gr - cr, gb - cb)):
            return None

        if max(map(abs, (x_adj, y_adj))) > 2:
            print(f"warning: {(x_adj, y_adj)}: big nudge!")

        return (cl, ct, cr, cb)

    @property
    def codepoints(self) -> Iterable[int]:
        _ = self.glyphs
        return sorted(self._glyphs_by_codepoint)

    @cached_property
    def mappings(self) -> Iterable[Mapping]:
        _ = self.glyphs
        return [Mapping(*m) for m in sorted(self._glyphs_by_codepoint.items())]

    @property
    def mapped_ranges(self) -> Iterable[MappedRange]:
        r: Optional[MappedRange] = None
        for m in self.mappings:
            if r is not None:
                if m == r.limit:
                    r.length += 1
                    continue
                yield r
            r = MappedRange(start=m)
        if r is not None:
            yield r

    @cached_property
    def grouped_ranges(self) -> Sequence[RangeGroup]:
        return tuple(self._grouped_ranges)

    @property
    def _grouped_ranges(self) -> Iterable[RangeGroup]:
        g: Optional[RangeGroup] = None
        for r in self.mapped_ranges:
            if g is not None and (r.start.codepoint - g.limit.codepoint) > 2:
                yield g
                g = None
            if g is None:
                g = RangeGroup()
            g.append(r)
        if g is not None:
            yield g

    @cached_property
    def output_path(self) -> Path:
        outdir = self.manifest_path.parent
        filename = f"{outdir.parent.name}-{self.size}"
        a, _, b = filename.rpartition(".")
        if b and all(c == "0" for c in b):
            filename = a
        return self.manifest_path.parent / f"{filename}.ff"

    def write(self) -> None:
        with self.output_path.open("wb") as fp:
            self._write(fp)

    def _write(self, fp: IO[bytes]) -> None:
        glyphs = self.glyphs
        range_groups = self.grouped_ranges

        header = FontHeader(
            glyph_width=self.glyph_width,
            glyph_height=self.glyph_height,
            num_glyphs=len(glyphs),
            num_range_groups=len(range_groups),
        )
        header.write(fp)
        for glyph in glyphs:
            fp.write(glyph)
        for range_group in range_groups:
            range_group.write(fp)


def main() -> None:
    parser = ArgumentParser(
        prog="makefont",
        description="Make fonts for MicroPython VT100 emulator",
        epilog="From https://github.com/gbenson/micropython-vt100",
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        nargs="+",
        help="location of JSON manifest",
    )
    args = parser.parse_args()

    for path in args.paths:
        b = FontBuilder.build(path)
        b.write()

        with b.output_path.open("rb") as fp:
            magic, glyph_width, glyph_height, num_glyphs, num_range_groups \
                = struct.unpack("<IBBHH", fp.read(10))
            assert glyph_width < 10
            print(f"magic: 0x{magic:08x}")
            assert magic == 0xb2dc3eb9
            print(f"glyph size: {glyph_width}×{glyph_height}")
            print("num_glyphs:", num_glyphs)

            PRINT_GLYPHS = PRINT_RANGES = False
            PRINT_OPS = False

            pix = ["x"] * glyph_width
            PIX = " \u2591\u2592\u2593"
            for glyph_index in range(num_glyphs):
                if PRINT_GLYPHS:
                    print(f"glyphs[{glyph_index}]:")
                for _ in range(glyph_height):
                    row = struct.unpack("<I", fp.read(4))[0]
                    for x in range(glyph_width):
                        pix[x] = PIX[row & 3]
                        row >>= 2
                    if PRINT_GLYPHS:
                        print(f" {''.join(pix)}")

            def print_op(s: str) -> None:
                if PRINT_OPS:
                    print(f"  \x1B[38;5;240m; {s}\x1B[0m")

            print("num_range_groups:", num_range_groups)
            for range_group_index in range(num_range_groups):
                start_codepoint, num_slots = struct.unpack("<HB", fp.read(3))
                print(f"range group {range_group_index} "
                      f"({num_slots} slot{'' if num_slots == 1 else 's'})"
                      f"{':' if PRINT_RANGES else ''}")
                slots = []
                glyph_index = 0
                while len(slots) < num_slots:
                    op = struct.unpack("<b", fp.read(1))[0]
                    if op == 0:  # set glyph_index
                        glyph_index = struct.unpack("<H", fp.read(2))[0]
                        print_op(f"glyph_index={glyph_index}")
                    elif op > 0:  # step
                        print_op(f"step {op}")
                        s = f"U+{start_codepoint+len(slots):04x}"
                        g = str(glyph_index)
                        for _ in range(op):
                            slots.append(glyph_index)
                            glyph_index += 1
                        print_op(f"(glyph_index now {glyph_index})")
                        if PRINT_RANGES:
                            if op == 1:
                                print(f"    {s}          => {g}")
                            else:
                                s = f"[{s},U+{start_codepoint+len(slots):04x})"
                                g = f"[{g},{glyph_index})"
                                print(f"   {s} => {g}")
                    else:  # skip
                        print_op(f"skip {-op}")
                        slots.extend([0] * -op)
                assert len(slots) == num_slots
