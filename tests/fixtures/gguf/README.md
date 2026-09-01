# GGUF crash fixtures

Binary inputs that crashed `gen2::bundle::gguf::parse_gguf_metadata` before it
was bounded. Kept as files rather than as builder code because their size is
the point — the crash needs scale, not shape.

- `deeply_nested_arrays.gguf` — 200,000 levels of `ARRAY`-of-`ARRAY` nesting,
  ~2.4 MB. Each level cost one `skip_array` stack frame, so this overflowed the
  stack and aborted the process (`fatal runtime error: stack overflow`).
  Now refused by `MAX_ARRAY_DEPTH`. Regenerate with:

  ```python
  import struct
  b = bytearray(b"GGUF") + struct.pack("<IQQ", 3, 0, 1)
  b += struct.pack("<Q", 4) + b"deep" + struct.pack("<I", 9)
  for _ in range(200_000):
      b += struct.pack("<IQ", 9, 1)
  b += struct.pack("<IQ", 0, 0)
  open("deeply_nested_arrays.gguf", "wb").write(bytes(b))
  ```

Every other fixture in the GGUF suite is built byte-by-byte inside the test
module (`GgufBuilder`) so the bytes under test stay readable and diffable.

## `corpus/`

Seed inputs for `cargo fuzz run gguf` — the shapes a mutator will not
stumble onto (a valid magic, each header version, a string KV, a full
architecture header, a string array, a nested array, a negative signed value,
and the declared-length allocation bomb). See `fuzz/README.md`.
