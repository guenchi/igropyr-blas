# igropyr-blas

`(igropyr blas)` — vector scoring at memory bandwidth: an optional CBLAS
`sgemv` binding with a pure-Scheme fallback, for [Chez Scheme][chez]. One
call fills `scores[i] = row_i · query` over a row-major float32 matrix —
the kernel of embedding search (RAG lookups, semantic dedup,
recommendations).

It is an optional module of [Igropyr][igropyr], usable on its own: the
only dependency is Chez Scheme, and it works with or without a native
BLAS installed.

## API

```scheme
(import (igropyr blas))

(blas-scores! base n dim query scores)   ; scores[i] = row_i · query
;;   base:   flat float32 bytevector, row-major [n x dim]
;;   query:  float32 bytevector [dim]
;;   scores: float32 bytevector [n], overwritten
(blas-available?)                        ; #t when a native BLAS backs the call
```

All four buffers are native-endian float32 `bytevector`s. `n` rows of
`dim` columns are scored against one `dim`-length query; each result is
written into `scores`.

## Two lanes, one contract

Loading is graceful by design. Native BLAS candidates are tried once at
library init — Accelerate on macOS, OpenBLAS/netlib on Linux and
FreeBSD — and a miss just means the pure-Scheme loop, so **correctness
never depends on the native library**. `blas-available?` reports which
lane is live; both satisfy the same contract, agreeing within f32
accumulation tolerance (~1e-4 for unit vectors).

Callers with a fused scan of their own (inline filtering, early exit)
can keep it behind `blas-available?` and use this only for the big-scan
lane.

## Safety

Bounds are checked in Scheme *before* the native call, not left to the
callee: the native path reads raw pointers, so a short buffer would be a
silent heap overrun rather than an error. `n` and `dim` are also range-
checked against `int32-max`, because they cross the FFI as C `int` and a
larger value would be silently truncated into a different shape than the
buffers were validated for. Every violation raises an
`assertion-violation`; none reach the native call.

## Scheduling

An FFI call cannot be preempted, so one call stalls the calling
scheduler for the scan's duration — roughly 0.2 ms at 5k rows × 512 dims
float32, ~5 ms at 100k (memory bandwidth). **Spread** those stalls: let
every process scan its own replica inline, and the per-process stall
duty cycle `q*s/N` stays negligible. Do **not** funnel searches into a
few dedicated processes — that adds a hop to every search and one
saturating queue for nothing. If `q*s/N` ever climbs past a few percent,
tile the call and yield between tiles, or offload it off the scheduler
(the async-DNS threadpool pattern); when replicas outgrow memory, shard
and scatter-gather. Spread out, never centralize.

## Layout and use

The library name `(igropyr blas)` maps to `igropyr/blas.sc`, so add this
repository's root to your library directories and include `.sc` in the
library extensions:

```sh
CHEZSCHEMELIBDIRS=. CHEZSCHEMELIBEXTS=.sc scheme --script your-program.ss
```

Inside Igropyr, drop `igropyr/blas.sc` alongside the other `igropyr/*.sc`
sources; it is already listed in the build.

## Test

The scoring contract is pinned against an independent double-
accumulation reference; whichever lane is active must agree within
tolerance, and every bounds violation must fail as a Scheme error:

```sh
CHEZSCHEMELIBDIRS=. CHEZSCHEMELIBEXTS=.sc scheme -q --script test/blas.sc
```

## License

MIT. See [LICENSE](LICENSE).

[chez]: https://www.scheme.com
[igropyr]: https://github.com/guenchi/Igropyr
