"""Robot commands and the game's little-endian, 10-byte input packet."""
import math
import struct
from dataclasses import dataclass

PACKET = struct.Struct("<BBHhhh")
BUTTONS = ("intake", "score", "outtake", "field_relative", "climb", "block", "stow", "hood")


def _axis(value: float) -> int:
    value = max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0
    return math.floor(value * 32767 + 0.5)


@dataclass
class Inputs:
    x: float = 0.0
    y: float = 0.0
    turn: float = 0.0
    intake: bool = False
    score: bool = False
    outtake: bool = False
    field_relative: bool = False
    climb: bool = False
    block: bool = False
    stow: bool = False
    hood: bool = False
    seq: int = 0

    def encode(self, seq: int | None = None) -> bytes:
        return encode(self, seq)

    @classmethod
    def decode(cls, packet: bytes):
        return decode(packet)


def encode(inputs: Inputs, seq: int | None = None) -> bytes:
    buttons = sum(1 << i for i, name in enumerate(BUTTONS) if getattr(inputs, name))
    return PACKET.pack(1, buttons, (inputs.seq if seq is None else seq) & 0xffff,
                       _axis(inputs.x), _axis(inputs.y), _axis(inputs.turn))


def decode(packet: bytes) -> Inputs:
    if len(packet) != PACKET.size:
        raise ValueError("input packet must be exactly 10 bytes")
    kind, buttons, seq, x, y, turn = PACKET.unpack(packet)
    if kind != 1:
        raise ValueError("not an input packet (type must be 1)")
    return Inputs(x=x / 32767, y=y / 32767, turn=turn / 32767, seq=seq,
                  **{name: bool(buttons & (1 << i)) for i, name in enumerate(BUTTONS)})
