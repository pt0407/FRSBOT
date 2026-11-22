"""FUEL section of RBRP frames, mirrored from the game's frame codec.

Frame types: 3 key (all FUEL, absolute, plus radii), 2 full-delta (changed FUEL, absolute),
4 packed delta (changed FUEL in 4 groups: bit 1 = int8 position delta, bit 2 = nibble rotation delta).
Entry header u16: index | where << 12 | squashed << 15. Positions are int16 * 5e-4 m.
"""
import struct

import numpy as np

POS_SCALE = 5e-4


def _nib(v: int) -> int:
    return ((v & 15) ^ 8) - 8


class FuelState:
    def __init__(self, count: int):
        self.count = count
        self.pos = np.zeros((count, 3), np.int32)
        self.rot = np.zeros((count, 4), np.int32)
        self.where = np.zeros(count, np.uint8)
        self.squash = np.zeros(count, np.uint8)
        self.radius = np.zeros(count, np.float32)

    def apply(self, f: bytes, q: int, ftype: int) -> int:
        """Decode the FUEL section starting at byte q of frame f. Returns the end offset."""
        if ftype == 4:
            for group in range(4):
                n = struct.unpack_from("<H", f, q)[0]
                q += 2
                for _ in range(n):
                    h = struct.unpack_from("<H", f, q)[0]
                    q += 2
                    i = h & 0xFFF
                    if group & 1:
                        self.pos[i] += struct.unpack_from("<bbb", f, q)
                        q += 3
                    else:
                        self.pos[i] = struct.unpack_from("<hhh", f, q)
                        q += 6
                    if group & 2:
                        a, b = f[q], f[q + 1]
                        self.rot[i] += (_nib(a), _nib(a >> 4), _nib(b), _nib(b >> 4))
                        q += 2
                    else:
                        self.rot[i] = struct.unpack_from("<bbbb", f, q)
                        q += 4
                    self.where[i] = h >> 12 & 7
                    if h & 0x8000:
                        self.squash[i] = f[q]
                        q += 4
                    else:
                        self.squash[i] = 0
            return q
        n = struct.unpack_from("<H", f, q)[0]
        q += 2
        for _ in range(n):
            h = struct.unpack_from("<H", f, q)[0]
            i = h & 0xFFF
            self.pos[i] = struct.unpack_from("<hhh", f, q + 2)
            self.rot[i] = struct.unpack_from("<bbbb", f, q + 8)
            self.where[i] = h >> 12 & 7
            q += 12
            if h & 0x8000:
                self.squash[i] = f[q]
                q += 4
            else:
                self.squash[i] = 0
        if ftype == 3:
            self.radius[:] = np.frombuffer(f, "<f4", self.count, q)
            q += 4 * self.count
        return q

    def positions(self) -> np.ndarray:
        return self.pos * POS_SCALE
