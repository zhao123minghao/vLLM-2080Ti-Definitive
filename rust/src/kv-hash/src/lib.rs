// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Rust re-implementation of the `vllm/v1/core` prefix-cache block hashing hot path.
//!
//! This is a feasibility spike for moving `vllm/v1/core` (pure-Python scheduler and
//! KV-cache bookkeeping) into the Rust frontend. The slice ported here is the
//! per-step prefix-cache hashing chain:
//!
//! - `hash_block_tokens(hash_fn, parent_hash, token_ids, extra_keys=None)`
//! - `init_none_hash(hash_fn)` / `NONE_HASH`
//!
//! from `vllm.v1.core.kv_cache_utils`. Both supported algorithms are reproduced
//! byte-exactly, including the serialization format, so that Rust and Python agree
//! on every block hash:
//!
//! | algo          | serialization                       | digest |
//! |---------------|-------------------------------------|--------|
//! | `sha256`      | CPython pickle protocol 5           | SHA-256 |
//! | `sha256_cbor` | canonical CBOR (`cbor2.dumps(..., canonical=True)`) | SHA-256 |
//!
//! Known gaps (deliberately out of scope for the spike, see the migration notes in
//! `CHANGELOG.md`):
//!
//! - `xxhash` / `xxhash_cbor` need XXH3-128. The Python side requires
//!   `xxhash.xxh3_128_digest`, and no crate providing XXH3-128 is available in this
//!   offline cargo registry (`twox-hash` only implements XXH32/XXH64).
//! - CPython splits pickle payloads into 64 KiB frames; this encoder always emits a
//!   single frame. Our payloads (one block of token ids) are far below that limit.
//! - `extra_keys` values are limited to the shapes vLLM actually produces
//!   (ints, strings, bytes and nested tuples).

use sha2::{Digest, Sha256};

/// Length of a block hash in bytes, matching `vllm.v1.core.kv_cache_utils.BlockHash`.
pub const BLOCK_HASH_LEN: usize = 32;

/// Fixed seed used by ``init_none_hash`` for cryptographic hash functions when
/// `PYTHONHASHSEED` is unset.
pub const DEFAULT_NONE_HASH_SEED: &str = "vllm-none-hash";

/// A chained prefix-cache block hash.
pub type BlockHash = [u8; BLOCK_HASH_LEN];

/// Supported serialization/digest pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashAlgo {
    /// Python default: `pickle.dumps(..., protocol=5)` + SHA-256.
    Sha256,
    /// Portable: `cbor2.dumps(..., canonical=True)` + SHA-256.
    Sha256Cbor,
}

impl HashAlgo {
    /// Parse the name used by vLLM's `--prefix-caching-hash-algo` setting.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sha256" => Some(Self::Sha256),
            "sha256_cbor" => Some(Self::Sha256Cbor),
            _ => None,
        }
    }

    /// The vLLM-facing name of this algorithm.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
            Self::Sha256Cbor => "sha256_cbor",
        }
    }

    /// Whether this algorithm is collision resistant (mirrors
    /// `_NON_CRYPTO_HASH_FUNCTIONS` in Python).
    pub const fn is_crypto(self) -> bool {
        true
    }

    fn digest(self, payload: &[u8]) -> BlockHash {
        let digest = Sha256::digest(payload);
        let mut out: BlockHash = [0; BLOCK_HASH_LEN];
        out.copy_from_slice(&digest);
        out
    }

    /// Serialize `value` the way the Python hash function for this algorithm does.
    pub fn serialize(self, value: &HashValue) -> Vec<u8> {
        match self {
            Self::Sha256 => encode_pickle(value),
            Self::Sha256Cbor => encode_cbor(value),
        }
    }

    /// Hash a value, i.e. the Rust equivalent of `hash_fn(value)`.
    pub fn hash(self, value: &HashValue) -> BlockHash {
        self.digest(&self.serialize(value))
    }
}

/// A Python value that can appear in the prefix-cache hash chain.
///
/// Python's `hash_fn` accepts any picklable object; vLLM only ever feeds it
/// `(parent_block_hash, token_ids_tuple, extra_keys)` where the individual elements
/// are ints, strings, bytes and nested tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashValue {
    /// Python `None`, used for the absent `extra_keys`.
    None,
    /// A Python `int`.
    Int(i64),
    /// A Python `bytes` object, e.g. a parent `BlockHash`.
    Bytes(Vec<u8>),
    /// A Python `str`.
    Str(String),
    /// A Python `tuple`.
    Tuple(Vec<HashValue>),
}

impl HashValue {
    /// Build the exact value Python hashes for one block, i.e.
    /// `(parent_block_hash, tuple(token_ids), extra_keys)`.
    ///
    /// `parent` follows Python's `if not parent_block_hash: parent = NONE_HASH`, so an
    /// empty slice is treated as "no parent" just like `None`. `extra_keys` must be a
    /// [`HashValue::Tuple`] to mirror Python's `tuple[Any, ...] | None`, or `None`.
    pub fn block(
        none_hash: &BlockHash,
        parent: Option<&[u8]>,
        token_ids: &[u32],
        extra_keys: Option<&HashValue>,
    ) -> Self {
        let parent = match parent {
            Some(bytes) if !bytes.is_empty() => bytes.to_vec(),
            _ => none_hash.to_vec(),
        };
        Self::Tuple(vec![
            Self::Bytes(parent),
            Self::Tuple(token_ids.iter().map(|id| Self::Int(i64::from(*id))).collect()),
            extra_keys.cloned().unwrap_or(Self::None),
        ])
    }
}

/// Derive `NONE_HASH`, i.e. the Rust equivalent of `init_none_hash(hash_fn)`.
///
/// Python resolves the seed as: `PYTHONHASHSEED` if set, else
/// [`DEFAULT_NONE_HASH_SEED`] for cryptographic algorithms, else 32 random bytes for
/// the non-cryptographic ones. Callers that honour `PYTHONHASHSEED` must pass that
/// value as `seed`; the random-seed fallback only applies to `xxhash*`, which this
/// crate does not implement.
pub fn none_hash(algo: HashAlgo, seed: &str) -> BlockHash {
    algo.hash(&HashValue::Str(seed.to_string()))
}

/// Compute the chained block hash, matching
/// `hash_block_tokens(hash_fn, parent_block_hash, curr_block_token_ids, extra_keys)`.
pub fn hash_block_tokens(
    algo: HashAlgo,
    none_hash: &BlockHash,
    parent: Option<&[u8]>,
    token_ids: &[u32],
    extra_keys: Option<&HashValue>,
) -> BlockHash {
    algo.hash(&HashValue::block(none_hash, parent, token_ids, extra_keys))
}

/// Chain the block hashes of a whole token sequence, returning every intermediate
/// hash (the analogue of the `_hash_all_blocks` loop used by
/// `benchmarks/benchmark_prefix_block_hash.py`).
pub fn hash_blocks(
    algo: HashAlgo,
    none_hash: &BlockHash,
    blocks: &[Vec<u32>],
    extra_keys: Option<&HashValue>,
) -> Vec<BlockHash> {
    let mut parent: Option<BlockHash> = None;
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        let hash = hash_block_tokens(
            algo,
            none_hash,
            parent.as_ref().map(|h| h.as_slice()),
            block,
            extra_keys,
        );
        parent = Some(hash);
        out.push(hash);
    }
    out
}

// ---------------------------------------------------------------------------
// pickle protocol 5 (the layout CPython 3.12 produces for our value shapes)
// ---------------------------------------------------------------------------

const OP_PROTO: u8 = 0x80;
const OP_FRAME: u8 = 0x95;
const OP_STOP: u8 = b'.';
const OP_MARK: u8 = b'(';
const OP_NONE: u8 = b'N';
const OP_EMPTY_TUPLE: u8 = b')';
const OP_TUPLE1: u8 = 0x85;
const OP_TUPLE2: u8 = 0x86;
const OP_TUPLE3: u8 = 0x87;
const OP_TUPLE: u8 = b't';
const OP_MEMOIZE: u8 = 0x94;
const OP_BININT: u8 = b'J';
const OP_BININT1: u8 = b'K';
const OP_BININT2: u8 = b'M';
const OP_LONG1: u8 = 0x8a;
const OP_SHORT_BINBYTES: u8 = b'C';
const OP_BINBYTES: u8 = b'B';
const OP_SHORT_BINUNICODE: u8 = 0x8c;
const OP_BINUNICODE: u8 = 0x8d;

/// Encode `value` exactly as `pickle.dumps(value, protocol=5)` does.
pub fn encode_pickle(value: &HashValue) -> Vec<u8> {
    let mut body = Vec::new();
    write_pickle(value, &mut body);
    body.push(OP_STOP);

    let mut out = Vec::with_capacity(body.len() + 11);
    out.push(OP_PROTO);
    out.push(5);
    out.push(OP_FRAME);
    // FRAME carries the length of everything that follows it.
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn write_pickle(value: &HashValue, out: &mut Vec<u8>) {
    match value {
        HashValue::None => out.push(OP_NONE),
        HashValue::Int(int) => write_pickle_int(*int, out),
        HashValue::Bytes(bytes) => {
            if bytes.len() < 256 {
                out.push(OP_SHORT_BINBYTES);
                out.push(bytes.len() as u8);
            } else {
                out.push(OP_BINBYTES);
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            }
            out.extend_from_slice(bytes);
            out.push(OP_MEMOIZE);
        }
        HashValue::Str(string) => {
            let utf8 = string.as_bytes();
            if utf8.len() < 256 {
                out.push(OP_SHORT_BINUNICODE);
                out.push(utf8.len() as u8);
            } else {
                out.push(OP_BINUNICODE);
                out.extend_from_slice(&(utf8.len() as u32).to_le_bytes());
            }
            out.extend_from_slice(utf8);
            out.push(OP_MEMOIZE);
        }
        HashValue::Tuple(items) => {
            // CPython emits dedicated opcodes for 1..=3 items and MARK+TUPLE beyond.
            match items.len() {
                0 => out.push(OP_EMPTY_TUPLE),
                1..=3 => {
                    for item in items {
                        write_pickle(item, out);
                    }
                    out.push(match items.len() {
                        1 => OP_TUPLE1,
                        2 => OP_TUPLE2,
                        _ => OP_TUPLE3,
                    });
                }
                _ => {
                    out.push(OP_MARK);
                    for item in items {
                        write_pickle(item, out);
                    }
                    out.push(OP_TUPLE);
                }
            }
            out.push(OP_MEMOIZE);
        }
    }
}

fn write_pickle_int(int: i64, out: &mut Vec<u8>) {
    if let Ok(byte) = u8::try_from(int) {
        out.push(OP_BININT1);
        out.push(byte);
    } else if let Ok(short) = u16::try_from(int) {
        out.push(OP_BININT2);
        out.extend_from_slice(&short.to_le_bytes());
    } else if let Ok(value) = i32::try_from(int) {
        out.push(OP_BININT);
        out.extend_from_slice(&value.to_le_bytes());
    } else {
        // LONG1: minimal little-endian two's-complement representation.
        let mut width = 8;
        let bytes = int.to_le_bytes();
        if int >= 0 {
            while width > 1 && bytes[width - 1] == 0 && bytes[width - 2] < 0x80 {
                width -= 1;
            }
        } else {
            while width > 1 && bytes[width - 1] == 0xff && bytes[width - 2] >= 0x80 {
                width -= 1;
            }
        }
        out.push(OP_LONG1);
        out.push(width as u8);
        out.extend_from_slice(&bytes[..width]);
    }
}

// ---------------------------------------------------------------------------
// canonical CBOR (the layout `cbor2.dumps(value, canonical=True)` produces)
// ---------------------------------------------------------------------------

const CBOR_NULL: u8 = 0xf6;

/// Encode `value` as canonical CBOR, matching `cbor2.dumps(value, canonical=True)`.
///
/// Canonical form only affects map ordering and length minimisation; the values fed
/// to the hash chain contain no maps, so minimal-length integers and definite-length
/// arrays/byte strings are sufficient.
pub fn encode_cbor(value: &HashValue) -> Vec<u8> {
    let mut out = Vec::new();
    write_cbor(value, &mut out);
    out
}

fn write_cbor(value: &HashValue, out: &mut Vec<u8>) {
    match value {
        HashValue::None => out.push(CBOR_NULL),
        HashValue::Int(int) => {
            if *int >= 0 {
                write_cbor_uint(0, *int as u64, out);
            } else {
                // Major type 1 encodes -1 - n.
                write_cbor_uint(1, (-1 - *int) as u64, out);
            }
        }
        HashValue::Bytes(bytes) => {
            write_cbor_uint(2, bytes.len() as u64, out);
            out.extend_from_slice(bytes);
        }
        HashValue::Str(string) => {
            let utf8 = string.as_bytes();
            write_cbor_uint(3, utf8.len() as u64, out);
            out.extend_from_slice(utf8);
        }
        HashValue::Tuple(items) => {
            write_cbor_uint(4, items.len() as u64, out);
            for item in items {
                write_cbor(item, out);
            }
        }
    }
}

fn write_cbor_uint(major: u8, value: u64, out: &mut Vec<u8>) {
    let head = major << 5;
    if value < 24 {
        out.push(head | value as u8);
    } else if value <= u64::from(u8::MAX) {
        out.push(head | 24);
        out.push(value as u8);
    } else if value <= u64::from(u16::MAX) {
        out.push(head | 25);
        out.extend_from_slice(&(value as u16).to_be_bytes());
    } else if value <= u64::from(u32::MAX) {
        out.push(head | 26);
        out.extend_from_slice(&(value as u32).to_be_bytes());
    } else {
        out.push(head | 27);
        out.extend_from_slice(&value.to_be_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_encode(hash: &BlockHash) -> String {
        hex::encode(hash)
    }

    #[test]
    fn pickle_layout_matches_cpython_protocol_5() {
        // Golden bytes captured from CPython 3.12:
        //   pickle.dumps((b"A"*32, (1, 2, 3, 4), None), protocol=5)
        let value = HashValue::Tuple(vec![
            HashValue::Bytes(vec![b'A'; 32]),
            HashValue::Tuple(vec![
                HashValue::Int(1),
                HashValue::Int(2),
                HashValue::Int(3),
                HashValue::Int(4),
            ]),
            HashValue::None,
        ]);
        let expected = concat!(
            "800595320000000000000043204141414141414141414141414141414141414141414141414141",
            "41414141414194284b014b024b034b0474944e87942e"
        );
        assert_eq!(hex::encode(encode_pickle(&value)), expected);
    }

    #[test]
    fn pickle_int_width_selection_matches_cpython() {
        let value = HashValue::Tuple(vec![
            HashValue::Bytes(vec![b'A'; 32]),
            HashValue::Tuple(vec![
                HashValue::Int(300),
                HashValue::Int(65536),
                HashValue::Int(200000),
            ]),
            HashValue::None,
        ]);
        let expected = concat!(
            "800595360000000000000043204141414141414141414141414141414141414141414141414141",
            "414141414141944d2c014a000001004a400d030087944e87942e"
        );
        assert_eq!(hex::encode(encode_pickle(&value)), expected);
    }

    #[test]
    fn pickle_nested_extra_keys_match_cpython() {
        let value = HashValue::Tuple(vec![
            HashValue::Bytes(vec![b'A'; 32]),
            HashValue::Tuple(vec![HashValue::Int(1), HashValue::Int(2)]),
            HashValue::Tuple(vec![HashValue::Str("mmhash".to_string())]),
        ]);
        let expected = concat!(
            "800595370000000000000043204141414141414141414141414141414141414141414141414141",
            "414141414141944b014b0286948c066d6d6861736894859487942e"
        );
        assert_eq!(hex::encode(encode_pickle(&value)), expected);
    }

    #[test]
    fn cbor_layout_is_canonical() {
        // Golden bytes captured from cbor2.dumps((b"A"*32, (1, 2, 3, 4), None),
        // canonical=True).
        let value = HashValue::Tuple(vec![
            HashValue::Bytes(vec![b'A'; 32]),
            HashValue::Tuple(vec![
                HashValue::Int(1),
                HashValue::Int(2),
                HashValue::Int(3),
                HashValue::Int(4),
            ]),
            HashValue::None,
        ]);
        let expected =
            "83582041414141414141414141414141414141414141414141414141414141414141418401020304f6";
        assert_eq!(hex::encode(encode_cbor(&value)), expected);
    }

    #[test]
    fn empty_parent_falls_back_to_none_hash() {
        let algo = HashAlgo::Sha256Cbor;
        let none = none_hash(algo, DEFAULT_NONE_HASH_SEED);
        let tokens = [7u32, 8, 9];
        assert_eq!(
            HashValue::block(&none, None, &tokens, None),
            HashValue::block(&none, Some(&[]), &tokens, None)
        );
        assert_eq!(
            HashValue::block(&none, None, &tokens, None),
            HashValue::block(&none, Some(&none), &tokens, None)
        );
    }

    #[test]
    fn chained_hashes_are_order_dependent() {
        let algo = HashAlgo::Sha256Cbor;
        let none = none_hash(algo, DEFAULT_NONE_HASH_SEED);
        let blocks = vec![vec![1u32, 2, 3], vec![4u32, 5, 6]];
        let hashes = hash_blocks(algo, &none, &blocks, None);
        assert_eq!(hashes.len(), 2);
        assert_ne!(hashes[0], hashes[1]);
        assert_eq!(
            hashes[0],
            hash_block_tokens(algo, &none, None, &blocks[0], None)
        );
        assert_eq!(
            hashes[1],
            hash_block_tokens(algo, &none, Some(&hashes[0]), &blocks[1], None)
        );
        assert_eq!(hex_encode(&hashes[0]).len(), 64);
    }

    #[test]
    fn python_default_algorithm_is_pickle_sha256() {
        // vllm/config/cache.py: prefix_caching_hash_algo = "sha256"
        assert_eq!(HashAlgo::from_name("sha256"), Some(HashAlgo::Sha256));
        assert_eq!(HashAlgo::from_name("nope"), None);
        assert_eq!(HashAlgo::Sha256.name(), "sha256");
        assert!(HashAlgo::Sha256Cbor.is_crypto());
    }
}
