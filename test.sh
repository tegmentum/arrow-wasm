#!/bin/bash
# Test script for arrow-wasm component

set -e

COMPONENT_PATH="target/wasm32-wasip1/release/arrow_wasm.wasm"

echo "=== Arrow-WASM Component Tests ==="
echo

# Build if needed
if [ ! -f "$COMPONENT_PATH" ]; then
    echo "Building component..."
    cargo component build --release
fi

# Test 1: Component exists
echo -n "Test 1: Component file exists... "
if [ -f "$COMPONENT_PATH" ]; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 2: Component is valid WASM
echo -n "Test 2: Component is valid WASM... "
MAGIC=$(xxd -l 4 -p "$COMPONENT_PATH")
if [ "$MAGIC" = "0061736d" ]; then
    echo "PASS"
else
    echo "FAIL (magic: $MAGIC)"
    exit 1
fi

# Test 3: Component size is reasonable
echo -n "Test 3: Component size is reasonable... "
SIZE=$(stat -f%z "$COMPONENT_PATH" 2>/dev/null || stat -c%s "$COMPONENT_PATH")
SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc)
if (( $(echo "$SIZE_MB >= 1.0" | bc -l) )) && (( $(echo "$SIZE_MB <= 20.0" | bc -l) )); then
    echo "PASS (${SIZE_MB} MB)"
else
    echo "FAIL (${SIZE_MB} MB)"
    exit 1
fi

# Test 4: All WIT files exist
echo -n "Test 4: All WIT files exist... "
WIT_FILES=(
    "wit/world.wit"
    "wit/types.wit"
    "wit/arrays.wit"
    "wit/record-batch.wit"
    "wit/compute.wit"
    "wit/io.wit"
    "wit/flight.wit"
)
ALL_EXIST=true
for wit in "${WIT_FILES[@]}"; do
    if [ ! -f "$wit" ]; then
        ALL_EXIST=false
        break
    fi
done
if $ALL_EXIST; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 5: Compression multiplexer dependency exists
echo -n "Test 5: Compression multiplexer dependency exists... "
if [ -f "wit/deps/compression-multiplexer/compression-multiplexer.wit" ] && \
   [ -f "wit/deps/compression-multiplexer/world.wit" ]; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 6: World exports all interfaces
echo -n "Test 6: World exports required interfaces... "
EXPORTS_OK=true
for export in "types" "arrays" "record-batch" "compute" "io" "flight"; do
    if ! grep -q "export $export" wit/world.wit; then
        EXPORTS_OK=false
        break
    fi
done
if $EXPORTS_OK; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 7: Compute interface has core functions
echo -n "Test 7: Compute interface has core functions... "
COMPUTE_OK=true
for func in "add:" "subtract:" "filter:" "sort:" "sum-i64:" "date-year:" "regex-match:" "window-row-number:"; do
    if ! grep -q "$func" wit/compute.wit; then
        COMPUTE_OK=false
        break
    fi
done
if $COMPUTE_OK; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 8: IO interface has all formats
echo -n "Test 8: IO interface has all formats... "
IO_OK=true
for format in "ipc-read" "parquet-read" "csv-read" "json-read" "snappy" "zstd" "gzip"; do
    if ! grep -q "$format" wit/io.wit; then
        IO_OK=false
        break
    fi
done
if $IO_OK; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 9: Flight interface is complete
echo -n "Test 9: Flight interface is complete... "
FLIGHT_OK=true
for item in "flight-descriptor" "flight-data" "encode-batch" "decode-batch" "serialize-flight-info"; do
    if ! grep -q "$item" wit/flight.wit; then
        FLIGHT_OK=false
        break
    fi
done
if $FLIGHT_OK; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

# Test 10: Types interface has Arrow types
echo -n "Test 10: Types interface has Arrow types... "
TYPES_OK=true
for type in "int32" "int64" "float64" "utf8" "timestamp" "schema" "arrow-error"; do
    if ! grep -q "$type" wit/types.wit; then
        TYPES_OK=false
        break
    fi
done
if $TYPES_OK; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi

echo
echo "=== All tests passed! ==="
echo "Component: $COMPONENT_PATH"
echo "Size: ${SIZE_MB} MB"
