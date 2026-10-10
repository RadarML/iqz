//! Native backend for `iqz`: fused mean-chirp transform kernels and zstd.
//!
//! Both entry points operate on a single chunk viewed as a C-contiguous
//! `(n, M)` int16 array; all shape handling and input coercion is done by the
//! Python layer. Zstd contexts and scratch buffers are kept per thread, and
//! all work runs with the GIL released.

use std::cell::RefCell;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use zstd_safe::{CCtx, DCtx};

struct Scratch {
    cctx: CCtx<'static>,
    dctx: DCtx<'static>,
    /// Encode: mean chirp as little-endian bytes; decode: decompressed mean.
    mean: Vec<u8>,
    /// Shuffled planes (`2 * n * M` bytes): all high bytes, then all low.
    planes: Vec<u8>,
    /// Encoded output blob.
    out: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        cctx: CCtx::create(),
        dctx: DCtx::create(),
        mean: Vec::new(),
        planes: Vec::new(),
        out: Vec::new(),
    });
}

fn zstd_err(code: usize) -> String {
    format!("zstd error: {}", zstd_safe::get_error_name(code))
}

/// Round `s / n` to the nearest integer, ties to even (matches `np.round`).
#[inline]
fn round_half_even(s: i64, n: i64) -> i16 {
    let q = s.div_euclid(n);
    let r2 = 2 * s.rem_euclid(n);
    let up = r2 > n || (r2 == n && q & 1 == 1);
    (if up { q + 1 } else { q }) as i16
}

/// Rounded mean over the `n` rows of `x`, as an `M`-vector.
fn mean_chirp(x: &[i16], n: usize, m: usize) -> Vec<i16> {
    if n <= 65536 {
        // sum is in [-2^31, 2^31 - 2^16], which fits in i32.
        let mut acc = vec![0i32; m];
        for row in x.chunks_exact(m) {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += v as i32;
            }
        }
        acc.iter()
            .map(|&a| round_half_even(a as i64, n as i64))
            .collect()
    } else {
        let mut acc = vec![0i64; m];
        for row in x.chunks_exact(m) {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += v as i64;
            }
        }
        acc.iter().map(|&a| round_half_even(a, n as i64)).collect()
    }
}

fn encode_impl(
    s: &mut Scratch,
    x: &[i16],
    n: usize,
    m: usize,
    level: i32,
) -> Result<usize, String> {
    let total = n * m;
    let mean = mean_chirp(x, n, m);

    s.planes.clear();
    s.planes.resize(2 * total, 0);
    let (hi, lo) = s.planes.split_at_mut(total);
    for ((row, h), l) in x
        .chunks_exact(m)
        .zip(hi.chunks_exact_mut(m))
        .zip(lo.chunks_exact_mut(m))
    {
        for (((&v, &mk), h), l) in row.iter().zip(&mean).zip(h.iter_mut()).zip(l.iter_mut()) {
            let d = v.wrapping_sub(mk);
            let u = (d.wrapping_shl(1) ^ (d >> 15)) as u16;
            *h = (u >> 8) as u8;
            *l = u as u8;
        }
    }

    s.mean.clear();
    s.mean.extend(mean.iter().flat_map(|v| v.to_le_bytes()));

    let cap = 4 + zstd_safe::compress_bound(2 * m) + zstd_safe::compress_bound(2 * total);
    s.out.clear();
    s.out.resize(cap, 0);
    let (head, rest) = s.out.split_at_mut(4);
    let a = s
        .cctx
        .compress(&mut rest[..], &s.mean, level)
        .map_err(zstd_err)?;
    head.copy_from_slice(&(a as u32).to_le_bytes());
    let b = s
        .cctx
        .compress(&mut rest[a..], &s.planes, level)
        .map_err(zstd_err)?;
    Ok(4 + a + b)
}

/// Encode one `(n, M)` int16 chunk; returns the blob.
#[pyfunction]
fn encode_chunk<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, i16>,
    level: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let (n, m) = (x.shape()[0], x.shape()[1]);
    if n == 0 || m == 0 {
        return Err(PyValueError::new_err("chunk must be non-empty"));
    }
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("input must be C-contiguous"))?;
    let len = py
        .detach(|| SCRATCH.with(|s| encode_impl(&mut s.borrow_mut(), xs, n, m, level)))
        .map_err(PyValueError::new_err)?;
    Ok(SCRATCH.with(|s| PyBytes::new(py, &s.borrow().out[..len])))
}

fn content_size(frame: &[u8], what: &str) -> Result<usize, String> {
    match zstd_safe::get_frame_content_size(frame) {
        Ok(Some(size)) => Ok(size as usize),
        Ok(None) => Err(format!("{what} frame has no content size")),
        Err(_) => Err(format!("{what} frame is not a valid zstd frame")),
    }
}

fn decode_impl(
    s: &mut Scratch,
    blob: &[u8],
    n: usize,
    m: usize,
    out: &mut [i16],
) -> Result<(), String> {
    let total = n * m;
    if blob.len() < 4 {
        return Err("blob is truncated".into());
    }
    let k = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
    if blob.len() < 4 + k {
        return Err("blob is truncated".into());
    }
    let (fa, fb) = blob[4..].split_at(k);
    if content_size(fa, "mean")? != 2 * m || content_size(fb, "plane")? != 2 * total {
        return Err(format!("blob does not match output shape ({n}, {m})"));
    }

    s.mean.clear();
    s.mean.reserve(2 * m);
    let a = s.dctx.decompress(&mut s.mean, fa).map_err(zstd_err)?;
    s.planes.clear();
    s.planes.reserve(2 * total);
    let b = s.dctx.decompress(&mut s.planes, fb).map_err(zstd_err)?;
    if a != 2 * m || b != 2 * total {
        return Err("decompressed size mismatch".into());
    }

    let mean: Vec<i16> = s
        .mean
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| i16::from_le_bytes(c))
        .collect();
    let (hi, lo) = s.planes.split_at(total);
    for ((orow, h), l) in out
        .chunks_exact_mut(m)
        .zip(hi.chunks_exact(m))
        .zip(lo.chunks_exact(m))
    {
        for (((o, &h), &l), &mk) in orow.iter_mut().zip(h).zip(l).zip(&mean) {
            let u = ((h as u16) << 8) | l as u16;
            let r = ((u >> 1) as i16) ^ -((u & 1) as i16);
            *o = mk.wrapping_add(r);
        }
    }
    Ok(())
}

/// Decode `blob` into a preallocated `(n, M)` int16 array.
#[pyfunction]
fn decode_into<'py>(
    py: Python<'py>,
    blob: PyReadonlyArray1<'py, u8>,
    mut out: PyReadwriteArray2<'py, i16>,
) -> PyResult<()> {
    let (n, m) = (out.shape()[0], out.shape()[1]);
    let b = blob
        .as_slice()
        .map_err(|_| PyValueError::new_err("blob must be contiguous"))?;
    let o = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("output must be C-contiguous"))?;
    py.detach(|| SCRATCH.with(|s| decode_impl(&mut s.borrow_mut(), b, n, m, o)))
        .map_err(PyValueError::new_err)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(decode_into, module)?)?;
    module.add("ZSTD_VERSION", zstd_safe::version_string())?;
    Ok(())
}
