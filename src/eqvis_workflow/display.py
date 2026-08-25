"""The geometry a figure is drawn for, and how its ink scales to suit.

A figure has a size it is meant to be seen at and a distance it is meant to be
read from, and the default -- its natural height, at arm's length -- is only
one of the answers. :class:`Display` is the other answers.
"""

from dataclasses import dataclass

from .console import console_warn

# Every size in these figures -- fonts, line widths, marker areas -- is set in
# points, and a point is a fixed physical size no matter how large the canvas
# is. Growing text *relative to the figure* therefore means shrinking the
# canvas rather than touching the hundred-odd point sizes scattered through the
# drawing code: the same figure drawn on 1.6 in instead of 9 in has text 5.5x
# larger as a fraction of its height, and the tick labels matplotlib recreates
# at draw time come out scaled too. Raising the DPI by the same factor holds
# the pixel count, so the image still prints at full resolution at its intended
# size.
#
# The factor is not a free knob -- it follows from angular size. Text is
# legible when it subtends enough of the eye's field of view, and the default
# point sizes here are drawn to be read at arm's length:
READING_DISTANCE = 0.5  # metres


CM_PER_INCH = 2.54


# Above this much enlargement, fine annotation is dropped rather than scaled:
# basin names and their like are supporting detail, and at poster scale there
# is no room for them beside the labels that carry the figure.
DETAIL_LIMIT = 2.0


@dataclass(frozen=True)
class Display:
    """The geometry a figure is drawn for, and how its ink scales to suit.

    Not everything grows by the same factor. Text has to, or it stops being
    readable at the intended distance. Rules and markers only have to stay
    *visible*, and scaling those in step with the text turns hairline basin
    outlines into thick worms that swamp the raster they sit on -- so they grow
    as the square root instead, which keeps them present without letting them
    take over. Annotation that would collide is dropped instead of scaled.
    """

    size: tuple[float, float]
    dpi: float
    scale: float = 1.0

    @classmethod
    def for_figure(
        cls,
        design: tuple[float, float],
        dpi: int,
        display_height: float | None,
        viewing_distance: float | None,
    ) -> Display:
        """Work out the canvas for showing ``design`` at ``display_height`` cm,
        read from ``viewing_distance`` m.

        Shrinking a figure hurts legibility and standing back hurts it again,
        so the two effects multiply: shown at half its natural height and read
        from twice as far, a figure needs text four times larger relative to
        the page. Each bound falls back to what the figure is already drawn for
        -- its natural height, at arm's length -- so either option is useful on
        its own, and neither leaves the figure exactly as designed.
        """
        natural = design[1] * CM_PER_INCH
        scale = (natural / (display_height or natural)) * (
            (viewing_distance or READING_DISTANCE) / READING_DISTANCE
        )
        if scale <= 1.0:  # already big enough, for that distance
            return cls(design, float(dpi))
        if scale > DETAIL_LIMIT:
            # Point sizes are fixed physical sizes, so enlarging text relative
            # to the page is the same as shrinking the page. Past a couple of
            # times, the title and tick labels take more of the canvas than the
            # axes do, and no amount of thinning wins that room back -- the
            # figure is simply being asked to be smaller than its own labels.
            distance = viewing_distance or READING_DISTANCE
            console_warn(
                f"text has to grow {scale:.1f}x to read a "
                f"{display_height or natural:g} cm figure from {distance:g} m, "
                "which leaves the labels taking more of the canvas than the "
                f"figure; {natural * distance / READING_DISTANCE:.0f} cm tall "
                "would carry this design at that distance"
            )
        return cls((design[0] / scale, design[1] / scale), dpi * scale, scale)

    def mark(self, points: float) -> float:
        """A line width or marker size, damped so it does not swamp the figure."""
        return points / self.scale**0.5

    def ticks(self, count: int) -> int:
        """How many of ``count`` tick labels still fit along an axis."""
        return max(2, round(count / self.scale))

    def keep(self, values: list, count: int) -> list:
        """Thin ``values`` to ``count`` evenly spaced entries, ends included.

        Spacing them from the ends rather than stepping from the front matters
        on a diverging scale, where dropping the middle label loses the zero
        the colours are read against.
        """
        count = max(2, count)
        if count >= len(values):
            return list(values)
        last = len(values) - 1
        return [values[round(i * last / (count - 1))] for i in range(count)]

    @property
    def detailed(self) -> bool:
        """Whether fine annotation is still worth drawing."""
        return self.scale < DETAIL_LIMIT

    def report(self, design: tuple[float, float]) -> None:
        """Say what the scaling did, since the file itself looks no different --
        the same pixels, drawn for a different physical size."""
        if self.scale == 1.0:
            return
        print(
            f"scaled text {self.scale:.1f}x for the requested display size: "
            f"canvas {design[0]:g}x{design[1]:g} -> "
            f"{self.size[0]:.2f}x{self.size[1]:.2f} in at {self.dpi:.0f} dpi "
            f"({round(self.size[1] * self.dpi)} px tall)"
            + ("" if self.detailed else "; fine annotation dropped")
        )


# What a figure drawn at its designed size scales by: nothing. The drawing
# helpers take a Display only when a command has one, and fall back to this.
NATURAL = Display(size=(0.0, 0.0), dpi=0.0)
