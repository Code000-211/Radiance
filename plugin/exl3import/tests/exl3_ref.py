"""exl3_ref.py -- from Dyluhn/radiance-exl3 v1.3.0 (c88ba2b, common/exl3_ref.py; Apache-2.0,
../NOTICE.radiance-exl3), unchanged below this line. The numpy twin of the C oracle: a line-for-line transcription of Dylan's exl3-moe
`exl3_moe/kernels/trellis.py` (exl3-moe 8428c31; itself read from exllamav3 @ 12414d0), torch -> numpy.

It exists so the C decoder (exl3_decode.c) is checked against something written independently of it,
and so tools/exl3_fixture.py can make trellis tensors and their dense weights without torch. It is
checked against trellis.py itself by tests/test_ref_vs_trellis_py.py when that file is given
(EXL3_TRELLIS_PY), and against the golden through the C tests.

Format (trellis.py's docstring, verbatim in substance): a weight W [k, n] (y = x @ W) is
W = diag(suh) . Hb . W_hat . Hb . diag(svh), Hb the block-diagonal natural-order Sylvester Hadamard of
size 128 scaled 1/sqrt(128). W_hat is cut into 16x16 tiles; trellis[tk, tn] holds tile (tk, tn) as
16*K int16 = 8*K uint32 words, read little-endian as one MSB-first bit stream of 256*K bits. Position
p owns bits [p*K, (p+1)*K); its state is the 16-bit window ending at bit (p+1)*K, circular. Position
p = 8*L + j is tile row (k) 2*(L%4) + (0,1,8,9)[j%4], column (n) L//4 + 8*(j//4).

Plus what trellis.py does not have: the canonical radiance codes plane (codes_plane / plane_symbols),
written here from the format statement, not from the C.
"""
import numpy as np

HAD = 128
MUL1_MULT = 0x83DCD12D
MCG_MULT = 0xCBAC1FED
INST3_MULT, INST3_ADD = 89226354, 64248484
K_INV = float(np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0])
K_BIAS = float(np.array([0xC931], dtype=np.uint16).view(np.float16)[0])
CODEBOOKS = ("3inst", "mcg", "mul1")


def tile_rows_cols():
    """(row, col) inside the 16x16 tile of each trellis position p = 0..255 (row = k axis, col = n axis)."""
    p = np.arange(256)
    lane, j = p // 8, p % 8
    row = 2 * (lane % 4) + np.array([0, 1, 8, 9])[j % 4]
    col = lane // 4 + 8 * (j // 4)
    return row, col


def _stream_bits(words):
    """[..., W] uint32 words -> [..., 32 * W] bits (int64 0/1), MSB of word 0 first."""
    shifts = np.arange(31, -1, -1, dtype=np.int64)
    w = words.astype(np.int64) & 0xFFFFFFFF
    return ((w[..., None] >> shifts) & 1).reshape(*words.shape[:-1], -1)


def _words(trellis):
    """int16 [..., 16K] -> uint32 [..., 8K], little-endian pairs (word = lo | hi << 16)."""
    t = np.ascontiguousarray(trellis).astype(np.int64) & 0xFFFF
    return (t[..., 0::2] | (t[..., 1::2] << 16)).astype(np.uint32)


def states(trellis, bits):
    """trellis [..., 16 * bits] int16 -> [..., 256] 16-bit trellis states (int64)."""
    assert trellis.dtype == np.int16 and trellis.shape[-1] == 16 * bits
    stream = _stream_bits(_words(trellis))
    total = 256 * bits
    end = (np.arange(256) + 1) * bits
    index = (end[:, None] - 16 + np.arange(16)) % total  # [256, 16], MSB first
    window = stream[..., index]
    return (window << np.arange(15, -1, -1)).sum(-1)


def symbols(trellis, bits):
    """trellis [..., 16 * bits] -> [..., 256] K-bit symbols, symbol p = stream bits [pK, (p+1)K)."""
    stream = _stream_bits(_words(trellis)).reshape(*trellis.shape[:-1], 256, bits)
    return (stream << np.arange(bits - 1, -1, -1)).sum(-1)


def pack(syms, bits):
    """[..., 256] K-bit symbols -> [..., 16 * bits] int16 trellis. Any symbols are a valid encoding."""
    syms = np.asarray(syms)
    assert syms.shape[-1] == 256
    shifts = np.arange(bits - 1, -1, -1)
    stream = ((syms.astype(np.int64)[..., None] >> shifts) & 1).reshape(*syms.shape[:-1], -1)
    stream = stream.reshape(*syms.shape[:-1], 8 * bits, 32)
    words = (stream << np.arange(31, -1, -1)).sum(-1)  # uint32 values in int64
    lo, hi = words & 0xFFFF, (words >> 16) & 0xFFFF
    out = np.stack([lo, hi], -1).reshape(*syms.shape[:-1], 16 * bits)
    return out.astype(np.uint16).view(np.int16)


def decode(state, codebook="mul1"):
    """16-bit states -> fp16 values, exllamav3's decode_3inst<cb> (one fp16 rounding)."""
    x = np.asarray(state).astype(np.int64) & 0xFFFF
    if codebook == "mul1":
        x = (x * MUL1_MULT) & 0xFFFFFFFF
        byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
        exact = (1024 + byte_sum).astype(np.float64) * K_INV + K_BIAS  # exact in float64
        return exact.astype(np.float16)
    if codebook == "mcg":
        x = (x * MCG_MULT) & 0xFFFFFFFF
    elif codebook == "3inst":
        x = (x * INST3_MULT + INST3_ADD) & 0xFFFFFFFF
    else:
        raise ValueError(f"codebook {codebook!r} not in {CODEBOOKS}")
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    low = (x & 0xFFFF).astype(np.uint16).view(np.float16)
    high = (x >> 16).astype(np.uint16).view(np.float16)
    return (low.astype(np.float64) + high.astype(np.float64)).astype(np.float16)


def unpack_tiles(trellis, bits, codebook="mul1"):
    """trellis [kt, nt, 16 * bits] int16 -> W_hat [16 * kt, 16 * nt] fp16 (rotated basis, [k, n])."""
    kt, nt, _ = trellis.shape
    values = decode(states(trellis, bits), codebook)  # [kt, nt, 256]
    row, col = tile_rows_cols()
    tiles = np.empty((kt, nt, 16, 16), dtype=np.float16)
    tiles[:, :, row, col] = values
    return tiles.transpose(0, 2, 1, 3).reshape(16 * kt, 16 * nt)


def hadamard(n=HAD):
    """Natural-order Sylvester Hadamard matrix (+-1, float32)."""
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def had_rows(x):
    """Blockwise normalized Hadamard along the last axis (a multiple of 128), float32."""
    h = (hadamard() / np.float32(np.sqrt(HAD))).astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    return (x.reshape(*x.shape[:-1], -1, HAD) @ h).reshape(x.shape)


def reconstruct(trellis, suh, svh, bits, codebook="mul1"):
    """Full weight W [k, n] float32 with y = x @ W."""
    w_hat = unpack_tiles(trellis, bits, codebook).astype(np.float32)
    w = had_rows(w_hat.T).T
    w = w * suh.astype(np.float32)[:, None]
    return had_rows(w) * svh.astype(np.float32)[None, :]


def random_tensor(k, n, bits, rng):
    """Synthetic EXL3 tensor (trellis, suh, svh) as trellis.py makes it: uniform random symbols, suh/svh
    random signs times scales near real magnitudes."""
    syms = rng.integers(0, 1 << bits, size=(k // 16, n // 16, 256))
    sign = lambda m: rng.integers(0, 2, size=m) * 2 - 1  # noqa: E731
    scale = lambda m: 0.5 + rng.random(m)  # noqa: E731
    suh = (sign(k) * scale(k) * 0.01).astype(np.float16)
    svh = (sign(n) * scale(n)).astype(np.float16)
    return pack(syms, bits), suh, svh


# ------------------------------------------------------------------ radiance's canonical codes plane

def plane_symbols(trellis, bits):
    """trellis [kt, nt, 16K] -> symbol matrix S [n, k] (radiance's [rows, cols] view): S[n, k] is the symbol of
    the position that decodes W_hat[k, n]."""
    kt, nt, _ = trellis.shape
    sym = symbols(trellis, bits)  # [kt, nt, 256]
    row, col = tile_rows_cols()
    tiles = np.empty((kt, nt, 16, 16), dtype=np.int64)  # [tk, tn, r (k), c (n)]
    tiles[:, :, row, col] = sym
    return tiles.transpose(1, 3, 0, 2).reshape(16 * nt, 16 * kt)  # [tn, c, tk, r] -> [n, k]


def pack_plane(S, bits):
    """Symbol matrix [rows, cols] -> uK plane bytes [rows, ceil(cols*K/8)], each row LSB-first (rad_encoding.h)."""
    rows, cols = S.shape
    b = ((S.astype(np.int64)[..., None] >> np.arange(bits)) & 1).astype(np.uint8).reshape(rows, cols * bits)
    return np.packbits(b, axis=1, bitorder="little")


def unpack_plane(plane, cols, bits):
    """The inverse of pack_plane."""
    b = np.unpackbits(plane, axis=1, bitorder="little")[:, : cols * bits].astype(np.int64)
    return (b.reshape(plane.shape[0], cols, bits) << np.arange(bits)).sum(-1)


def codes_plane(trellis, bits):
    """trellis [kt, nt, 16K] -> the canonical codes plane bytes [n, ceil(k*K/8)]."""
    return pack_plane(plane_symbols(trellis, bits), bits)


def trellis_from_plane(S, bits):
    """Symbol matrix [n, k] -> trellis [k/16, n/16, 16K]."""
    n, k = S.shape
    row, col = tile_rows_cols()
    tiles = S.reshape(n // 16, 16, k // 16, 16).transpose(2, 0, 3, 1)  # [tk, tn, r (k), c (n)]
    return pack(tiles[:, :, row, col], bits)


def to_bf16_bits(x):
    """float32 -> bfloat16 bits (uint16), round to nearest even (finite inputs)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)
