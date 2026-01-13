# arrow-wasm

Apache Arrow as a WebAssembly Component.

This project provides a complete implementation of Apache Arrow as a WebAssembly Component Model component, enabling language-agnostic access to Arrow's data processing capabilities through standard WIT (WebAssembly Interface Types) interfaces.

## Features

### Data Types
- Full support for Arrow primitive types (integers, floats, boolean, null)
- String and binary data (UTF-8, large UTF-8, binary, large binary)
- Temporal types (Date32, Date64, Timestamp, Time32, Time64, Duration, Interval)
- Complex types (List, LargeList, Struct, Map, Union, Dictionary)
- Fixed-size types (FixedSizeBinary, FixedSizeList, Decimal128, Decimal256)

### I/O Formats
- **IPC**: Arrow IPC stream and file formats for efficient serialization
- **Parquet**: Full Parquet read/write with compression support
- **CSV**: CSV parsing and writing with schema inference
- **JSON**: Line-delimited JSON read/write with schema inference
- **Avro**: Avro format reading with schema inference

### Compression Codecs
- **Built-in**: Uncompressed, Snappy, LZ4
- **Via compression-multiplexer**: ZSTD, GZIP, BZIP2, LZMA (requires component composition)

### Compute Kernels

#### Arithmetic Operations
- add, subtract, multiply, divide, modulo, negate
- Scalar operations (add-scalar-i64, add-scalar-f64, multiply-scalar-i64, multiply-scalar-f64)
- Wrapping arithmetic (add-wrapping, subtract-wrapping, multiply-wrapping, negate-wrapping)

#### Comparison Operations
- Compare arrays (eq, not-eq, lt, lt-eq, gt, gt-eq)
- Compare with scalars (i64, f64, string)
- NULL-safe comparison (distinct, not-distinct)

#### Boolean Operations
- and, or, not, is-null, is-not-null
- and-not (left AND NOT right)
- Three-valued logic (and-kleene, or-kleene) for SQL NULL semantics

#### Bitwise Operations
- bitwise-and, bitwise-or, bitwise-xor, bitwise-not
- bitwise-and-not, bitwise-shift-left, bitwise-shift-right
- Scalar operations (bitwise-and-scalar, bitwise-or-scalar, bitwise-xor-scalar)

#### Aggregations
- sum, min, max, count, mean
- variance, stddev (sample and population)
- median, percentile
- bool-any, bool-all
- first-value, last-value
- min-string, max-string, min-binary, max-binary

#### Selection & Filtering
- filter, take
- sort, sort-indices, lexsort
- limit, skip, shift
- unique, value-counts
- List membership (in-list-i64, in-list-string)

#### String Operations
- string-length, bit-length, string-lower, string-upper, string-trim
- string-contains, string-starts-with, string-ends-with
- string-concat, concat-elements, substring
- SQL LIKE: string-like, string-ilike, string-nlike, string-nilike

#### Regex Operations
- regex-match, regex-extract, regex-extract-group
- regex-replace, regex-replace-all
- regex-count, regex-split

#### Base64 Operations
- b64-encode (encode binary to base64 string)
- b64-decode (decode base64 string to binary)

#### Date/Time Operations
- date-year, date-month, date-day, date-day-of-week, date-day-of-year
- date-week, date-quarter
- time-hour, time-minute, time-second
- time-millisecond, time-microsecond, time-nanosecond
- date-add-days, date-add-months, date-add-years
- date-diff-days, timestamp-truncate

#### Window Functions
- row-number, rank, dense-rank, percent-rank, cume-dist, ntile
- lead, lag
- first-value, last-value, nth-value
- Running aggregates: sum, avg, min, max, count

#### Type Casting
- cast arrays between compatible types

#### Conditional Operations
- nullif (set values to null where condition is true)
- if-else (select values based on boolean condition)

#### Array Operations
- concat (concatenate multiple arrays)
- concat-batches (concatenate multiple record batches)
- interleave (merge arrays by index selection)

#### Partitioning Operations
- partition (group consecutive equal values)
- rank (compute rank of values)

### Flight Data Exchange
- Encode/decode record batches to/from Flight format
- Create and manage Flight descriptors and endpoints
- Serialize/deserialize Flight metadata

## Building

### Prerequisites
- Rust (1.70+)
- cargo-component (`cargo install cargo-component`)

### Build

```bash
# Build release component
cargo component build --release

# The component will be at:
# target/wasm32-wasip1/release/arrow_wasm.wasm
```

### Test

```bash
./test.sh
```

## Project Structure

```
arrow-wasm/
|-- Cargo.toml              # Project configuration
|-- src/
|   |-- lib.rs              # Implementation
|-- wit/
|   |-- world.wit           # Component world definition
|   |-- types.wit           # Arrow type definitions
|   |-- arrays.wit          # Array operations
|   |-- record-batch.wit    # RecordBatch operations
|   |-- compute.wit         # Compute kernels
|   |-- io.wit              # I/O operations
|   |-- flight.wit          # Flight data exchange
|   |-- deps/
|       |-- compression-multiplexer/  # Compression codec dependency
|-- test.sh                 # Test script
```

## WIT Interfaces

### types
Core Arrow type definitions including data types, schema, field, and error types.

### arrays
Array resource with operations for creating, accessing, and manipulating arrays.

### record-batch
RecordBatch resource for columnar data with schema.

### compute
Comprehensive compute kernels for data processing.

### io
I/O operations for reading/writing Arrow data in various formats.

### flight
Flight-like data exchange for distributed Arrow data transfer.

## Usage with Component Model

This component is designed to be composed with other WebAssembly components. To use compression codecs beyond the built-in ones (Snappy, LZ4), compose with the compression-multiplexer component:

```bash
# Example composition (requires wasm-tools)
wasm-tools compose arrow_wasm.wasm \
  -d compression-multiplexer.wasm \
  -o arrow_wasm_full.wasm
```

## Dependencies

- arrow-rs 57.2 - Apache Arrow Rust implementation
- arrow-avro 57.2 - Avro format support
- parquet 57.2 - Parquet format support
- wit-bindgen 0.51 - WebAssembly Interface Types code generation

## License

Apache-2.0

## Implementation Notes

### Supported Operations
Most compute kernels are fully implemented using the arrow-rs compute modules. Some advanced operations (window functions, certain regex operations) return `NotImplemented` errors as placeholders for future implementation.

### Compression
- **Snappy** and **LZ4** compression are built-in and work directly.
- **ZSTD**, **GZIP**, **BZIP2**, and **LZMA** require the compression-multiplexer component to be composed at runtime.

### Memory Management
Arrow data structures use reference counting through the Component Model's resource system, ensuring efficient memory usage when sharing data between operations.
