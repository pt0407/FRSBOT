import struct

import pytest

from frsbot.core.inputs import BUTTONS, Inputs, decode, encode


@pytest.mark.parametrize("buttons", range(256))
def test_round_trip(buttons):
    inputs = Inputs(x=0.25, y=-0.5, turn=1, seq=54321,
                    **{name: bool(buttons & (1 << i)) for i, name in enumerate(BUTTONS)})
    packet = inputs.encode()
    decoded = Inputs.decode(packet)
    assert len(packet) == 10
    assert packet[:4] == struct.pack("<BBH", 1, buttons, 54321)
    for name in ("x", "y", "turn"):
        assert getattr(decoded, name) == pytest.approx(getattr(inputs, name), abs=0.5 / 32767)
    for name in BUTTONS:
        assert getattr(decoded, name) == getattr(inputs, name)
    assert decoded.seq == inputs.seq
    assert decoded.encode() == packet


def test_neutral_and_wire_endpoints():
    assert encode(Inputs()) == b"\x01" + bytes(9)
    assert decode(encode(Inputs())) == Inputs()
    assert encode(Inputs(x=-1, y=1, turn=-0.5), seq=0x1234) == struct.pack(
        "<BBHhhh", 1, 0, 0x1234, -32767, 32767, -16383)


def test_clamps_axes_and_wraps_sequence_like_game():
    assert decode(encode(Inputs(x=-2, y=2, turn=float("nan"), seq=65536))) == Inputs(x=-1, y=1)
    assert decode(encode(Inputs(x=float("inf"), y=-float("inf")), seq=-1)) == Inputs(seq=65535)


@pytest.mark.parametrize("packet", [b"", bytes(9), bytes(11), bytes(10), b"\x02" + bytes(9)])
def test_invalid_packet(packet):
    with pytest.raises(ValueError):
        decode(packet)
