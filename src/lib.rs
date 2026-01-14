//! Apache Arrow WebAssembly Component
//!
//! This crate provides Apache Arrow functionality as a WebAssembly Component,
//! enabling high-performance columnar data processing in Wasm runtimes.

#[allow(warnings)]
mod bindings;

use bindings::exports::arrow::arrow_wasm::{arrays, compute, io, record_batch, types};
use bytes::Bytes;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Re-export for internal use
use arrow_array::{ArrayRef, RecordBatch as ArrowRecordBatch};

// Global storage for fields (needed for recursive type definitions)
thread_local! {
    static FIELDS: RefCell<HashMap<u32, Arc<arrow_schema::Field>>> = RefCell::new(HashMap::new());
    static FIELD_COUNTER: RefCell<u32> = RefCell::new(0);
}

fn store_field(field: Arc<arrow_schema::Field>) -> u32 {
    FIELD_COUNTER.with(|c| {
        let id = *c.borrow();
        *c.borrow_mut() += 1;
        FIELDS.with(|f| f.borrow_mut().insert(id, field));
        id
    })
}

fn get_field(id: u32) -> Option<Arc<arrow_schema::Field>> {
    FIELDS.with(|f| f.borrow().get(&id).cloned())
}

// Type conversions
mod convert {
    use super::*;
    use types::{DataType, IntervalUnit, TimeUnit};

    pub fn to_arrow_time_unit(unit: TimeUnit) -> arrow_schema::TimeUnit {
        match unit {
            TimeUnit::Second => arrow_schema::TimeUnit::Second,
            TimeUnit::Millisecond => arrow_schema::TimeUnit::Millisecond,
            TimeUnit::Microsecond => arrow_schema::TimeUnit::Microsecond,
            TimeUnit::Nanosecond => arrow_schema::TimeUnit::Nanosecond,
        }
    }

    pub fn from_arrow_time_unit(unit: arrow_schema::TimeUnit) -> TimeUnit {
        match unit {
            arrow_schema::TimeUnit::Second => TimeUnit::Second,
            arrow_schema::TimeUnit::Millisecond => TimeUnit::Millisecond,
            arrow_schema::TimeUnit::Microsecond => TimeUnit::Microsecond,
            arrow_schema::TimeUnit::Nanosecond => TimeUnit::Nanosecond,
        }
    }

    pub fn to_arrow_interval_unit(unit: IntervalUnit) -> arrow_schema::IntervalUnit {
        match unit {
            IntervalUnit::YearMonth => arrow_schema::IntervalUnit::YearMonth,
            IntervalUnit::DayTime => arrow_schema::IntervalUnit::DayTime,
            IntervalUnit::MonthDayNano => arrow_schema::IntervalUnit::MonthDayNano,
        }
    }

    pub fn from_arrow_interval_unit(unit: arrow_schema::IntervalUnit) -> IntervalUnit {
        match unit {
            arrow_schema::IntervalUnit::YearMonth => IntervalUnit::YearMonth,
            arrow_schema::IntervalUnit::DayTime => IntervalUnit::DayTime,
            arrow_schema::IntervalUnit::MonthDayNano => IntervalUnit::MonthDayNano,
        }
    }

    pub fn to_arrow_data_type(dt: &DataType) -> arrow_schema::DataType {
        match dt {
            DataType::Null => arrow_schema::DataType::Null,
            DataType::Boolean => arrow_schema::DataType::Boolean,
            DataType::Int8 => arrow_schema::DataType::Int8,
            DataType::Int16 => arrow_schema::DataType::Int16,
            DataType::Int32 => arrow_schema::DataType::Int32,
            DataType::Int64 => arrow_schema::DataType::Int64,
            DataType::Uint8 => arrow_schema::DataType::UInt8,
            DataType::Uint16 => arrow_schema::DataType::UInt16,
            DataType::Uint32 => arrow_schema::DataType::UInt32,
            DataType::Uint64 => arrow_schema::DataType::UInt64,
            DataType::Float16 => arrow_schema::DataType::Float16,
            DataType::Float32 => arrow_schema::DataType::Float32,
            DataType::Float64 => arrow_schema::DataType::Float64,
            DataType::Utf8 => arrow_schema::DataType::Utf8,
            DataType::LargeUtf8 => arrow_schema::DataType::LargeUtf8,
            DataType::Binary => arrow_schema::DataType::Binary,
            DataType::LargeBinary => arrow_schema::DataType::LargeBinary,
            DataType::FixedSizeBinary(size) => arrow_schema::DataType::FixedSizeBinary(*size),
            DataType::Date32 => arrow_schema::DataType::Date32,
            DataType::Date64 => arrow_schema::DataType::Date64,
            DataType::Time32(unit) => arrow_schema::DataType::Time32(to_arrow_time_unit(*unit)),
            DataType::Time64(unit) => arrow_schema::DataType::Time64(to_arrow_time_unit(*unit)),
            DataType::Timestamp((unit, tz)) => {
                arrow_schema::DataType::Timestamp(to_arrow_time_unit(*unit), tz.clone().map(Arc::from))
            }
            DataType::Duration(unit) => arrow_schema::DataType::Duration(to_arrow_time_unit(*unit)),
            DataType::Interval(unit) => arrow_schema::DataType::Interval(to_arrow_interval_unit(*unit)),
            DataType::Decimal128((precision, scale)) => {
                arrow_schema::DataType::Decimal128(*precision, *scale)
            }
            DataType::Decimal256((precision, scale)) => {
                arrow_schema::DataType::Decimal256(*precision, *scale)
            }
            DataType::List(field_handle) => {
                let field = get_field(*field_handle)
                    .unwrap_or_else(|| Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Null, true)));
                arrow_schema::DataType::List(field)
            }
            DataType::LargeList(field_handle) => {
                let field = get_field(*field_handle)
                    .unwrap_or_else(|| Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Null, true)));
                arrow_schema::DataType::LargeList(field)
            }
            DataType::FixedSizeList((size, field_handle)) => {
                let field = get_field(*field_handle)
                    .unwrap_or_else(|| Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Null, true)));
                arrow_schema::DataType::FixedSizeList(field, *size)
            }
            DataType::Struct(field_handles) => {
                let fields: Vec<_> = field_handles
                    .iter()
                    .filter_map(|h| get_field(*h))
                    .collect();
                arrow_schema::DataType::Struct(fields.into())
            }
            DataType::Union(_) => arrow_schema::DataType::Null,
            DataType::Dictionary(_) => arrow_schema::DataType::Null,
            DataType::Map((field_handle, sorted)) => {
                let field = get_field(*field_handle)
                    .unwrap_or_else(|| Arc::new(arrow_schema::Field::new("entries", arrow_schema::DataType::Null, true)));
                arrow_schema::DataType::Map(field, *sorted)
            }
            DataType::RunEndEncoded(_) => arrow_schema::DataType::Null,
            DataType::BinaryView => arrow_schema::DataType::BinaryView,
            DataType::Utf8View => arrow_schema::DataType::Utf8View,
            DataType::ListView(field_handle) => {
                let field = get_field(*field_handle)
                    .unwrap_or_else(|| Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Null, true)));
                arrow_schema::DataType::ListView(field)
            }
            DataType::LargeListView(field_handle) => {
                let field = get_field(*field_handle)
                    .unwrap_or_else(|| Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Null, true)));
                arrow_schema::DataType::LargeListView(field)
            }
        }
    }

    pub fn from_arrow_data_type(dt: &arrow_schema::DataType) -> DataType {
        match dt {
            arrow_schema::DataType::Null => DataType::Null,
            arrow_schema::DataType::Boolean => DataType::Boolean,
            arrow_schema::DataType::Int8 => DataType::Int8,
            arrow_schema::DataType::Int16 => DataType::Int16,
            arrow_schema::DataType::Int32 => DataType::Int32,
            arrow_schema::DataType::Int64 => DataType::Int64,
            arrow_schema::DataType::UInt8 => DataType::Uint8,
            arrow_schema::DataType::UInt16 => DataType::Uint16,
            arrow_schema::DataType::UInt32 => DataType::Uint32,
            arrow_schema::DataType::UInt64 => DataType::Uint64,
            arrow_schema::DataType::Float16 => DataType::Float16,
            arrow_schema::DataType::Float32 => DataType::Float32,
            arrow_schema::DataType::Float64 => DataType::Float64,
            arrow_schema::DataType::Utf8 => DataType::Utf8,
            arrow_schema::DataType::LargeUtf8 => DataType::LargeUtf8,
            arrow_schema::DataType::Binary => DataType::Binary,
            arrow_schema::DataType::LargeBinary => DataType::LargeBinary,
            arrow_schema::DataType::FixedSizeBinary(size) => DataType::FixedSizeBinary(*size),
            arrow_schema::DataType::Date32 => DataType::Date32,
            arrow_schema::DataType::Date64 => DataType::Date64,
            arrow_schema::DataType::Time32(unit) => DataType::Time32(from_arrow_time_unit(*unit)),
            arrow_schema::DataType::Time64(unit) => DataType::Time64(from_arrow_time_unit(*unit)),
            arrow_schema::DataType::Timestamp(unit, tz) => {
                DataType::Timestamp((from_arrow_time_unit(*unit), tz.as_ref().map(|s| s.to_string())))
            }
            arrow_schema::DataType::Duration(unit) => DataType::Duration(from_arrow_time_unit(*unit)),
            arrow_schema::DataType::Interval(unit) => DataType::Interval(from_arrow_interval_unit(*unit)),
            arrow_schema::DataType::Decimal128(precision, scale) => {
                DataType::Decimal128((*precision, *scale))
            }
            arrow_schema::DataType::Decimal256(precision, scale) => {
                DataType::Decimal256((*precision, *scale))
            }
            arrow_schema::DataType::List(field) => {
                let handle = store_field(field.clone());
                DataType::List(handle)
            }
            arrow_schema::DataType::LargeList(field) => {
                let handle = store_field(field.clone());
                DataType::LargeList(handle)
            }
            arrow_schema::DataType::FixedSizeList(field, size) => {
                let handle = store_field(field.clone());
                DataType::FixedSizeList((*size, handle))
            }
            arrow_schema::DataType::Struct(fields) => {
                let handles: Vec<_> = fields
                    .iter()
                    .map(|f| store_field(f.clone()))
                    .collect();
                DataType::Struct(handles)
            }
            arrow_schema::DataType::Map(field, sorted) => {
                let handle = store_field(field.clone());
                DataType::Map((handle, *sorted))
            }
            arrow_schema::DataType::BinaryView => DataType::BinaryView,
            arrow_schema::DataType::Utf8View => DataType::Utf8View,
            arrow_schema::DataType::ListView(field) => {
                let handle = store_field(field.clone());
                DataType::ListView(handle)
            }
            arrow_schema::DataType::LargeListView(field) => {
                let handle = store_field(field.clone());
                DataType::LargeListView(handle)
            }
            _ => DataType::Null,
        }
    }
}

// Helper function to collect f64 values from a numeric array
fn collect_f64_values(arr: &dyn arrow_array::Array) -> Result<Vec<f64>, compute::ArrowError> {
    // Try Float64
    if let Some(f64_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        return Ok(f64_arr.iter().filter_map(|v| v).collect());
    }
    // Try Float32
    if let Some(f32_arr) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        return Ok(f32_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    // Try Int64
    if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Ok(i64_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    // Try Int32
    if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        return Ok(i32_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    // Try Int16
    if let Some(i16_arr) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Ok(i16_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    // Try Int8
    if let Some(i8_arr) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Ok(i8_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    // Try UInt64
    if let Some(u64_arr) = arr.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        return Ok(u64_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    // Try UInt32
    if let Some(u32_arr) = arr.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        return Ok(u32_arr.iter().filter_map(|v| v.map(|x| x as f64)).collect());
    }
    Err(compute::ArrowError::InvalidArgument("Expected numeric array for statistical computation".to_string()))
}

// Helper functions for date arithmetic

/// Convert total days since year 0 to year, month (1-12), day (1-31)
fn days_to_ymd(total_days: i32) -> (i32, i32, u32) {
    // Algorithm based on Howard Hinnant's date algorithms
    // http://howardhinnant.github.io/date_algorithms.html
    let z = total_days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32; // day of era
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era
    let y = (yoe as i32) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as i32, d)
}

/// Convert year, month (1-12), day (1-31) to total days since year 0
fn ymd_to_days(year: i32, month: u32, day: u32) -> i32 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // year of era
    let m = month;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1; // day of year
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // day of era
    era * 146097 + (doe as i32) - 719468
}

/// Return number of days in a given month
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            // Leap year check
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30, // fallback
    }
}

// Helper functions for window operations

/// Compute partition boundaries and sort indices for window functions
/// Returns (partitions as (start, end) pairs, sort_indices mapping sorted position to original index)
fn compute_window_partitions_and_order(
    partition_by: &[arrays::Array],
    order_by: &[arrays::Array],
    order_options: &[compute::SortOptions],
) -> Result<(Vec<(usize, usize)>, Vec<usize>), compute::ArrowError> {
    // Get length from first available array
    let len = if !partition_by.is_empty() {
        partition_by[0].get::<ArrayImpl>().inner.len()
    } else if !order_by.is_empty() {
        order_by[0].get::<ArrayImpl>().inner.len()
    } else {
        return Err(compute::ArrowError::InvalidArgument("Need at least one array".to_string()));
    };

    if len == 0 {
        return Ok((vec![], vec![]));
    }

    // Build combined sort columns: partition_by columns first (asc), then order_by columns
    let mut sort_columns: Vec<arrow_ord::sort::SortColumn> = Vec::new();

    // Add partition_by columns with ascending order
    for arr in partition_by {
        let arr_impl = arr.get::<ArrayImpl>();
        sort_columns.push(arrow_ord::sort::SortColumn {
            values: arr_impl.inner.clone(),
            options: Some(arrow_ord::sort::SortOptions {
                descending: false,
                nulls_first: true,
            }),
        });
    }

    // Add order_by columns with specified options
    for (arr, opts) in order_by.iter().zip(order_options.iter()) {
        let arr_impl = arr.get::<ArrayImpl>();
        sort_columns.push(arrow_ord::sort::SortColumn {
            values: arr_impl.inner.clone(),
            options: Some(arrow_ord::sort::SortOptions {
                descending: opts.descending,
                nulls_first: opts.nulls_first,
            }),
        });
    }

    // If no sort columns, use original order
    let sort_indices: Vec<usize> = if sort_columns.is_empty() {
        (0..len).collect()
    } else {
        let indices = arrow_ord::sort::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        (0..len).map(|i| indices.value(i) as usize).collect()
    };

    // Compute partition boundaries
    let partitions = if partition_by.is_empty() {
        // Single partition containing all rows
        vec![(0, len)]
    } else {
        let partition_arrays: Vec<_> = partition_by.iter()
            .map(|a| a.get::<ArrayImpl>().inner.clone())
            .collect();

        let mut boundaries = vec![(0usize, 0usize)];

        for i in 1..len {
            let curr_idx = sort_indices[i];
            let prev_idx = sort_indices[i - 1];

            // Check if partition key changed
            let changed = partition_arrays.iter().any(|arr| {
                !arrays_equal_at_index(arr, curr_idx, prev_idx)
            });

            if changed {
                boundaries.last_mut().unwrap().1 = i;
                boundaries.push((i, 0));
            }
        }
        boundaries.last_mut().unwrap().1 = len;
        boundaries
    };

    Ok((partitions, sort_indices))
}

/// Check if two rows have equal values in the given arrays (for partition/ordering comparison)
fn arrays_equal_at_index(arr: &Arc<dyn arrow_array::Array>, idx1: usize, idx2: usize) -> bool {
    // Handle nulls
    let null1 = arr.is_null(idx1);
    let null2 = arr.is_null(idx2);
    if null1 && null2 {
        return true;
    }
    if null1 || null2 {
        return false;
    }

    // Compare based on data type
    use arrow_array::*;

    if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        return a.value(idx1) == a.value(idx2);
    }
    if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
        return a.value(idx1) == a.value(idx2);
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        return a.value(idx1) == a.value(idx2);
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        return a.value(idx1) == a.value(idx2);
    }
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return a.value(idx1) == a.value(idx2);
    }
    if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
        return a.value(idx1) == a.value(idx2);
    }

    // Default: assume not equal for unsupported types
    false
}

/// Compute frame bounds for a given row position within a partition
/// Returns (frame_start, frame_end) as indices within the partition (relative to partition start)
/// The frame is [frame_start, frame_end) - exclusive end
fn compute_frame_bounds(
    frame: &Option<compute::WindowFrame>,
    current_pos: usize,       // Position within partition (0-indexed)
    partition_start: usize,   // Start of partition in sorted order
    partition_end: usize,     // End of partition in sorted order (exclusive)
) -> (usize, usize) {
    let partition_len = partition_end - partition_start;

    match frame {
        None => {
            // Default: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            (partition_start, partition_start + current_pos + 1)
        }
        Some(f) => {
            // Compute start bound
            let frame_start = if f.unbounded_start {
                partition_start
            } else {
                let offset = f.start_offset;
                if offset <= 0 {
                    // PRECEDING: go back by |offset| rows
                    let preceding = (-offset) as usize;
                    if preceding > current_pos {
                        partition_start
                    } else {
                        partition_start + current_pos - preceding
                    }
                } else {
                    // FOLLOWING: go forward by offset rows
                    let following = offset as usize;
                    let pos = partition_start + current_pos + following;
                    pos.min(partition_end)
                }
            };

            // Compute end bound (exclusive)
            let frame_end = if f.unbounded_end {
                partition_end
            } else {
                let offset = f.end_offset;
                if offset < 0 {
                    // PRECEDING: end before current row
                    let preceding = (-offset) as usize;
                    if preceding > current_pos {
                        partition_start // Empty frame
                    } else {
                        partition_start + current_pos - preceding + 1
                    }
                } else {
                    // CURRENT ROW or FOLLOWING
                    let following = offset as usize;
                    let pos = partition_start + current_pos + following + 1;
                    pos.min(partition_end)
                }
            };

            // Ensure frame_start <= frame_end
            if frame_start >= frame_end {
                (partition_start, partition_start) // Empty frame
            } else {
                (frame_start, frame_end)
            }
        }
    }
}

/// Check if two rows are equal across all ordering columns
fn rows_equal_for_ordering(order_arrays: &[Arc<dyn arrow_array::Array>], idx1: usize, idx2: usize) -> bool {
    if order_arrays.is_empty() {
        return true; // No ordering means all rows are "equal" for ranking purposes
    }
    order_arrays.iter().all(|arr| arrays_equal_at_index(arr, idx1, idx2))
}

/// Helper to get Option<i64> from Int64Array at index, handling nulls
fn get_i64_opt(arr: &arrow_array::Int64Array, idx: usize) -> Option<i64> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<f64> from Float64Array at index, handling nulls
fn get_f64_opt(arr: &arrow_array::Float64Array, idx: usize) -> Option<f64> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<i32> from Int32Array at index, handling nulls
fn get_i32_opt(arr: &arrow_array::Int32Array, idx: usize) -> Option<i32> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<i16> from Int16Array at index, handling nulls
fn get_i16_opt(arr: &arrow_array::Int16Array, idx: usize) -> Option<i16> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<i8> from Int8Array at index, handling nulls
fn get_i8_opt(arr: &arrow_array::Int8Array, idx: usize) -> Option<i8> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<u64> from UInt64Array at index, handling nulls
fn get_u64_opt(arr: &arrow_array::UInt64Array, idx: usize) -> Option<u64> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<u32> from UInt32Array at index, handling nulls
fn get_u32_opt(arr: &arrow_array::UInt32Array, idx: usize) -> Option<u32> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<u16> from UInt16Array at index, handling nulls
fn get_u16_opt(arr: &arrow_array::UInt16Array, idx: usize) -> Option<u16> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<u8> from UInt8Array at index, handling nulls
fn get_u8_opt(arr: &arrow_array::UInt8Array, idx: usize) -> Option<u8> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<f32> from Float32Array at index, handling nulls
fn get_f32_opt(arr: &arrow_array::Float32Array, idx: usize) -> Option<f32> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<bool> from BooleanArray at index, handling nulls
fn get_bool_opt(arr: &arrow_array::BooleanArray, idx: usize) -> Option<bool> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx)) }
}

/// Helper to get Option<String> from StringArray at index, handling nulls
fn get_string_opt(arr: &arrow_array::StringArray, idx: usize) -> Option<String> {
    use arrow_array::Array;
    if arr.is_null(idx) { None } else { Some(arr.value(idx).to_string()) }
}

// Main component struct
struct Component;

bindings::export!(Component with_types_in bindings);

// ============================================================================
// Types implementation
// ============================================================================

impl types::Guest for Component {
    type Field = FieldImpl;
    type Schema = SchemaImpl;
    type SchemaBuilder = SchemaBuilderImpl;

    fn schema_merge(schemas: Vec<types::Schema>) -> Result<types::Schema, types::ArrowError> {
        if schemas.is_empty() {
            return Err(types::ArrowError::InvalidArgument("No schemas to merge".to_string()));
        }

        let mut merged_fields: Vec<Arc<arrow_schema::Field>> = Vec::new();
        let mut merged_metadata: HashMap<String, String> = HashMap::new();

        for schema_res in schemas {
            let schema_impl = schema_res.get::<SchemaImpl>();
            for field in schema_impl.inner.fields() {
                // Check if field with same name already exists
                if !merged_fields.iter().any(|f| f.name() == field.name()) {
                    merged_fields.push(field.clone());
                }
            }
            // Merge metadata (later schemas override earlier ones)
            for (k, v) in schema_impl.inner.metadata() {
                merged_metadata.insert(k.clone(), v.clone());
            }
        }

        let merged_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            arrow_schema::Fields::from(merged_fields),
            merged_metadata,
        ));

        Ok(types::Schema::new(SchemaImpl { inner: merged_schema }))
    }
}

struct FieldImpl {
    inner: Arc<arrow_schema::Field>,
}

impl types::GuestField for FieldImpl {
    fn new(name: String, data_type: types::DataType, nullable: bool) -> Self {
        let arrow_dt = convert::to_arrow_data_type(&data_type);
        Self {
            inner: Arc::new(arrow_schema::Field::new(name, arrow_dt, nullable)),
        }
    }

    fn with_metadata(
        name: String,
        data_type: types::DataType,
        nullable: bool,
        metadata: types::Metadata,
    ) -> types::Field {
        let arrow_dt = convert::to_arrow_data_type(&data_type);
        let meta: HashMap<String, String> = metadata.into_iter().collect();
        types::Field::new(FieldImpl {
            inner: Arc::new(arrow_schema::Field::new(name, arrow_dt, nullable).with_metadata(meta)),
        })
    }

    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn data_type(&self) -> types::DataType {
        convert::from_arrow_data_type(self.inner.data_type())
    }

    fn is_nullable(&self) -> bool {
        self.inner.is_nullable()
    }

    fn metadata(&self) -> types::Metadata {
        self.inner
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

struct SchemaImpl {
    inner: Arc<arrow_schema::Schema>,
}

impl types::GuestSchema for SchemaImpl {
    fn new(fields: Vec<types::Field>) -> Self {
        let arrow_fields: Vec<Arc<arrow_schema::Field>> = fields
            .into_iter()
            .map(|f| {
                let impl_ref = f.get::<FieldImpl>();
                impl_ref.inner.clone()
            })
            .collect();
        Self {
            inner: Arc::new(arrow_schema::Schema::new(arrow_fields)),
        }
    }

    fn with_metadata(fields: Vec<types::Field>, metadata: types::Metadata) -> types::Schema {
        let arrow_fields: Vec<Arc<arrow_schema::Field>> = fields
            .into_iter()
            .map(|f| {
                let impl_ref = f.get::<FieldImpl>();
                impl_ref.inner.clone()
            })
            .collect();
        let meta: HashMap<String, String> = metadata.into_iter().collect();
        types::Schema::new(SchemaImpl {
            inner: Arc::new(arrow_schema::Schema::new(arrow_fields).with_metadata(meta)),
        })
    }

    fn fields(&self) -> Vec<types::Field> {
        self.inner
            .fields()
            .iter()
            .map(|f| types::Field::new(FieldImpl { inner: f.clone() }))
            .collect()
    }

    fn field(&self, index: u32) -> Option<types::Field> {
        self.inner
            .fields()
            .get(index as usize)
            .map(|f| types::Field::new(FieldImpl { inner: f.clone() }))
    }

    fn field_by_name(&self, name: String) -> Option<types::Field> {
        self.inner
            .field_with_name(&name)
            .ok()
            .map(|f| types::Field::new(FieldImpl { inner: Arc::new(f.clone()) }))
    }

    fn num_fields(&self) -> u32 {
        self.inner.fields().len() as u32
    }

    fn metadata(&self) -> types::Metadata {
        self.inner
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn index_of(&self, name: String) -> Option<u32> {
        self.inner.index_of(&name).ok().map(|i| i as u32)
    }

    fn contains_field(&self, name: String) -> bool {
        self.inner.index_of(&name).is_ok()
    }
}

struct SchemaBuilderImpl {
    fields: std::cell::RefCell<Vec<Arc<arrow_schema::Field>>>,
    metadata: std::cell::RefCell<HashMap<String, String>>,
}

impl types::GuestSchemaBuilder for SchemaBuilderImpl {
    fn new() -> Self {
        Self {
            fields: std::cell::RefCell::new(Vec::new()),
            metadata: std::cell::RefCell::new(HashMap::new()),
        }
    }

    fn add_field(&self, name: String, data_type: types::DataType, nullable: bool) {
        let arrow_dt = convert::to_arrow_data_type(&data_type);
        let field = Arc::new(arrow_schema::Field::new(name, arrow_dt, nullable));
        self.fields.borrow_mut().push(field);
    }

    fn add_field_with_metadata(&self, name: String, data_type: types::DataType, nullable: bool, metadata: types::Metadata) {
        let arrow_dt = convert::to_arrow_data_type(&data_type);
        let meta: HashMap<String, String> = metadata.into_iter().collect();
        let field = Arc::new(arrow_schema::Field::new(name, arrow_dt, nullable).with_metadata(meta));
        self.fields.borrow_mut().push(field);
    }

    fn set_metadata(&self, key: String, value: String) {
        self.metadata.borrow_mut().insert(key, value);
    }

    fn build(&self) -> types::Schema {
        let fields = self.fields.borrow().clone();
        let metadata = self.metadata.borrow().clone();
        let schema = if metadata.is_empty() {
            arrow_schema::Schema::new(fields)
        } else {
            arrow_schema::Schema::new(fields).with_metadata(metadata)
        };
        types::Schema::new(SchemaImpl { inner: Arc::new(schema) })
    }

    fn num_fields(&self) -> u32 {
        self.fields.borrow().len() as u32
    }
}

// ============================================================================
// Arrays implementation
// ============================================================================

use arrow_array::{
    builder::{ArrayBuilder, BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int16Builder, Int32Builder, Int64Builder, Int8Builder, StringBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder},
    BooleanArray, Float64Array, Int32Array, Int64Array, StringArray,
    cast::AsArray,
};

impl arrays::Guest for Component {
    type Array = ArrayImpl;
    type BooleanArrayBuilder = BooleanArrayBuilderImpl;
    type Int8ArrayBuilder = Int8ArrayBuilderImpl;
    type Int16ArrayBuilder = Int16ArrayBuilderImpl;
    type Int32ArrayBuilder = Int32ArrayBuilderImpl;
    type Int64ArrayBuilder = Int64ArrayBuilderImpl;
    type Uint8ArrayBuilder = Uint8ArrayBuilderImpl;
    type Uint16ArrayBuilder = Uint16ArrayBuilderImpl;
    type Uint32ArrayBuilder = Uint32ArrayBuilderImpl;
    type Uint64ArrayBuilder = Uint64ArrayBuilderImpl;
    type Float32ArrayBuilder = Float32ArrayBuilderImpl;
    type Float64ArrayBuilder = Float64ArrayBuilderImpl;
    type StringArrayBuilder = StringArrayBuilderImpl;
    type BinaryArrayBuilder = BinaryArrayBuilderImpl;
    type LargeStringArrayBuilder = LargeStringArrayBuilderImpl;
    type LargeBinaryArrayBuilder = LargeBinaryArrayBuilderImpl;
    type FixedSizeBinaryArrayBuilder = FixedSizeBinaryArrayBuilderImpl;
    type ListArrayBuilder = ListArrayBuilderImpl;
    type LargeListArrayBuilder = LargeListArrayBuilderImpl;
    type StructArrayBuilder = StructArrayBuilderImpl;
    type MapArrayBuilder = MapArrayBuilderImpl;
    type Date32ArrayBuilder = Date32ArrayBuilderImpl;
    type Date64ArrayBuilder = Date64ArrayBuilderImpl;
    type TimestampArrayBuilder = TimestampArrayBuilderImpl;
    type DurationArrayBuilder = DurationArrayBuilderImpl;
    type Time32ArrayBuilder = Time32ArrayBuilderImpl;
    type Time64ArrayBuilder = Time64ArrayBuilderImpl;
    type Decimal128ArrayBuilder = Decimal128ArrayBuilderImpl;
    type Decimal256ArrayBuilder = Decimal256ArrayBuilderImpl;

    fn boolean_array_from(values: Vec<bool>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(BooleanArray::from(values)),
        })
    }

    fn int32_array_from(values: Vec<i32>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(Int32Array::from(values)),
        })
    }

    fn int64_array_from(values: Vec<i64>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(Int64Array::from(values)),
        })
    }

    fn float64_array_from(values: Vec<f64>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(Float64Array::from(values)),
        })
    }

    fn string_array_from(values: Vec<String>) -> arrays::Array {
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(StringArray::from(refs)),
        })
    }

    fn int32_array_from_nullable(values: Vec<Option<i32>>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(Int32Array::from(values)),
        })
    }

    fn int64_array_from_nullable(values: Vec<Option<i64>>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(Int64Array::from(values)),
        })
    }

    fn float64_array_from_nullable(values: Vec<Option<f64>>) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(Float64Array::from(values)),
        })
    }

    fn string_array_from_nullable(values: Vec<Option<String>>) -> arrays::Array {
        let refs: Vec<Option<&str>> = values.iter().map(|o| o.as_deref()).collect();
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(StringArray::from(refs)),
        })
    }

    fn get_boolean(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<bool>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let bool_arr = arr_impl.inner.as_boolean_opt()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a boolean array".to_string()))?;
        Ok(Some(bool_arr.value(index as usize)))
    }

    fn get_int8(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i8>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int8Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not an Int8 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_int16(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i16>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int16Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not an Int16 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_int32(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i32>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int32Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not an Int32 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_int64(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int64Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not an Int64 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_uint8(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<u8>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::UInt8Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a UInt8 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_uint16(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<u16>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::UInt16Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a UInt16 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_uint32(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<u32>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::UInt32Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a UInt32 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_uint64(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<u64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::UInt64Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a UInt64 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_float32(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<f32>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Float32Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Float32 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_float64(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<f64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Float64Type>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Float64 array".to_string()))?;
        Ok(Some(prim_arr.value(index as usize)))
    }

    fn get_string(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<String>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let str_arr = arr_impl.inner.as_string_opt::<i32>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a String array".to_string()))?;
        Ok(Some(str_arr.value(index as usize).to_string()))
    }

    fn get_binary(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<Vec<u8>>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let bin_arr = arr_impl.inner.as_binary_opt::<i32>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Binary array".to_string()))?;
        Ok(Some(bin_arr.value(index as usize).to_vec()))
    }

    // ========== Temporal Accessors ==========

    fn get_date32(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i32>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let date_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Date32Array>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Date32 array".to_string()))?;
        Ok(Some(date_arr.value(index as usize)))
    }

    fn get_date64(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let date_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Date64Array>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Date64 array".to_string()))?;
        Ok(Some(date_arr.value(index as usize)))
    }

    fn get_timestamp(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        // Try all timestamp variants
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampSecondArray>() {
            return Ok(Some(ts_arr.value(index as usize)));
        }
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMillisecondArray>() {
            return Ok(Some(ts_arr.value(index as usize)));
        }
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>() {
            return Ok(Some(ts_arr.value(index as usize)));
        }
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampNanosecondArray>() {
            return Ok(Some(ts_arr.value(index as usize)));
        }
        Err(arrays::ArrowError::InvalidArgument("Not a Timestamp array".to_string()))
    }

    fn get_duration(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        // Try all duration variants
        if let Some(dur_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DurationSecondArray>() {
            return Ok(Some(dur_arr.value(index as usize)));
        }
        if let Some(dur_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DurationMillisecondArray>() {
            return Ok(Some(dur_arr.value(index as usize)));
        }
        if let Some(dur_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DurationMicrosecondArray>() {
            return Ok(Some(dur_arr.value(index as usize)));
        }
        if let Some(dur_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DurationNanosecondArray>() {
            return Ok(Some(dur_arr.value(index as usize)));
        }
        Err(arrays::ArrowError::InvalidArgument("Not a Duration array".to_string()))
    }

    fn get_time32(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i32>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        // Try both time32 variants
        if let Some(time_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Time32SecondArray>() {
            return Ok(Some(time_arr.value(index as usize)));
        }
        if let Some(time_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Time32MillisecondArray>() {
            return Ok(Some(time_arr.value(index as usize)));
        }
        Err(arrays::ArrowError::InvalidArgument("Not a Time32 array".to_string()))
    }

    fn get_time64(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<i64>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        // Try both time64 variants
        if let Some(time_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Time64MicrosecondArray>() {
            return Ok(Some(time_arr.value(index as usize)));
        }
        if let Some(time_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Time64NanosecondArray>() {
            return Ok(Some(time_arr.value(index as usize)));
        }
        Err(arrays::ArrowError::InvalidArgument("Not a Time64 array".to_string()))
    }

    // ========== Decimal Accessors ==========

    fn get_decimal128_as_string(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<String>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let dec_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Decimal128 array".to_string()))?;
        Ok(Some(dec_arr.value_as_string(index as usize)))
    }

    fn get_decimal128_i128(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<(i64, u64)>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let dec_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Decimal128 array".to_string()))?;
        let value = dec_arr.value(index as usize);
        // Split i128 into high (signed) and low (unsigned) 64-bit parts
        let high = (value >> 64) as i64;
        let low = value as u64;
        Ok(Some((high, low)))
    }

    fn get_decimal256_bytes(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<Vec<u8>>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if index as usize >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::OutOfBounds(format!("index {} out of bounds", index)));
        }
        if arr_impl.inner.is_null(index as usize) {
            return Ok(None);
        }
        let dec_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Decimal256Array>()
            .ok_or_else(|| arrays::ArrowError::InvalidArgument("Not a Decimal256 array".to_string()))?;
        let value = dec_arr.value(index as usize);
        Ok(Some(value.to_le_bytes().to_vec()))
    }

    fn concat(arr: Vec<arrays::Array>) -> Result<arrays::Array, arrays::ArrowError> {
        let refs: Vec<&dyn arrow_array::Array> = arr
            .iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();
        let result = arrow_select::concat::concat(&refs)
            .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn dictionary_encode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // If already dictionary-encoded, return as-is
        if matches!(arr_impl.inner.data_type(), arrow_schema::DataType::Dictionary(_, _)) {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        // Try to cast to dictionary with Int32 keys (most common)
        let dict_type = arrow_schema::DataType::Dictionary(
            Box::new(arrow_schema::DataType::Int32),
            Box::new(arr_impl.inner.data_type().clone()),
        );

        let result = arrow_cast::cast(&arr_impl.inner, &dict_type)
            .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn dictionary_decode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Check if it's a dictionary array
        match arr_impl.inner.data_type() {
            arrow_schema::DataType::Dictionary(_, value_type) => {
                // Cast to the value type to decode
                let result = arrow_cast::cast(&arr_impl.inner, value_type)
                    .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                Ok(arrays::Array::new(ArrayImpl { inner: result }))
            }
            _ => Err(arrays::ArrowError::InvalidArgument(
                "dictionary_decode requires a dictionary-encoded array".to_string()
            ))
        }
    }

    fn dictionary_keys(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Try different key types
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int8Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int16Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int32Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int64Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt8Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt16Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt32Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt64Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(dict.keys().clone()) }));
        }

        Err(arrays::ArrowError::InvalidArgument(
            "dictionary_keys requires a dictionary-encoded array".to_string()
        ))
    }

    fn dictionary_values(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Try different key types
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int8Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int16Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int32Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int64Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt8Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt16Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt32Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }
        if let Some(dict) = arr_impl.inner.as_any().downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::UInt64Type>>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: dict.values().clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument(
            "dictionary_values requires a dictionary-encoded array".to_string()
        ))
    }

    // ========== List Array Operations ==========

    fn list_lengths(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            let mut builder = arrow_array::builder::Int32Builder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value((list_arr.value(i).len()) as i32);
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let mut builder = arrow_array::builder::Int64Builder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    builder.append_value(list_arr.value(i).len() as i64);
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }

        Err(arrays::ArrowError::InvalidArgument("list_lengths requires a list array".to_string()))
    }

    fn list_values(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: list_arr.values().clone() }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: list_arr.values().clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("list_values requires a list array".to_string()))
    }

    fn list_flatten(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // For flatten, we need to handle nulls - flatten removes the list structure
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            let flattened = arrow_select::concat::concat(&[list_arr.values().as_ref()])
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: flattened }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let flattened = arrow_select::concat::concat(&[list_arr.values().as_ref()])
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: flattened }));
        }

        Err(arrays::ArrowError::InvalidArgument("list_flatten requires a list array".to_string()))
    }

    fn unnest_list(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Unnest returns the flattened values array (similar to list_values/list_flatten)
        // The difference is semantic - this explicitly "explodes" the list
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: list_arr.values().clone() }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: list_arr.values().clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("unnest_list requires a list array".to_string()))
    }

    fn list_contains_i64(arr: arrays::ArrayBorrow<'_>, value: i64) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    let list_values = list_arr.value(i);
                    if let Some(int_arr) = list_values.as_any().downcast_ref::<arrow_array::Int64Array>() {
                        let contains = int_arr.iter().any(|v| v == Some(value));
                        builder.append_value(contains);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("List values must be Int64".to_string()));
                    }
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    let list_values = list_arr.value(i);
                    if let Some(int_arr) = list_values.as_any().downcast_ref::<arrow_array::Int64Array>() {
                        let contains = int_arr.iter().any(|v| v == Some(value));
                        builder.append_value(contains);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("List values must be Int64".to_string()));
                    }
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }

        Err(arrays::ArrowError::InvalidArgument("list_contains_i64 requires a list array".to_string()))
    }

    fn list_contains_f64(arr: arrays::ArrayBorrow<'_>, value: f64) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    let list_values = list_arr.value(i);
                    if let Some(float_arr) = list_values.as_any().downcast_ref::<arrow_array::Float64Array>() {
                        // Handle NaN: NaN != NaN, so use total_cmp for proper comparison
                        let contains = float_arr.iter().any(|v| {
                            if let Some(v) = v {
                                if value.is_nan() { v.is_nan() } else { v == value }
                            } else {
                                false
                            }
                        });
                        builder.append_value(contains);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("List values must be Float64".to_string()));
                    }
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    let list_values = list_arr.value(i);
                    if let Some(float_arr) = list_values.as_any().downcast_ref::<arrow_array::Float64Array>() {
                        let contains = float_arr.iter().any(|v| {
                            if let Some(v) = v {
                                if value.is_nan() { v.is_nan() } else { v == value }
                            } else {
                                false
                            }
                        });
                        builder.append_value(contains);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("List values must be Float64".to_string()));
                    }
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }

        Err(arrays::ArrowError::InvalidArgument("list_contains_f64 requires a list array".to_string()))
    }

    fn list_contains_string(arr: arrays::ArrayBorrow<'_>, value: String) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    let list_values = list_arr.value(i);
                    if let Some(str_arr) = list_values.as_any().downcast_ref::<arrow_array::StringArray>() {
                        let contains = str_arr.iter().any(|v| v == Some(value.as_str()));
                        builder.append_value(contains);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("List values must be String".to_string()));
                    }
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(list_arr.len());
            for i in 0..list_arr.len() {
                if list_arr.is_null(i) {
                    builder.append_null();
                } else {
                    let list_values = list_arr.value(i);
                    if let Some(str_arr) = list_values.as_any().downcast_ref::<arrow_array::StringArray>() {
                        let contains = str_arr.iter().any(|v| v == Some(value.as_str()));
                        builder.append_value(contains);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("List values must be String".to_string()));
                    }
                }
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
        }

        Err(arrays::ArrowError::InvalidArgument("list_contains_string requires a list array".to_string()))
    }

    fn list_element(arr: arrays::ArrayBorrow<'_>, index: i64) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Helper macro for extracting element at index from list
        macro_rules! list_element_impl {
            ($list_arr:expr, $value_type:ty, $builder_type:ty) => {{
                let mut builder = <$builder_type>::with_capacity($list_arr.len());
                for i in 0..$list_arr.len() {
                    if $list_arr.is_null(i) {
                        builder.append_null();
                    } else {
                        let list_values = $list_arr.value(i);
                        let len = list_values.len() as i64;
                        // Handle negative indices (count from end)
                        let actual_idx = if index < 0 { len + index } else { index };

                        if actual_idx < 0 || actual_idx >= len {
                            builder.append_null();
                        } else if let Some(typed_arr) = list_values.as_any().downcast_ref::<$value_type>() {
                            if typed_arr.is_null(actual_idx as usize) {
                                builder.append_null();
                            } else {
                                builder.append_value(typed_arr.value(actual_idx as usize));
                            }
                        } else {
                            builder.append_null();
                        }
                    }
                }
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
            }};
        }

        // Try ListArray
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>() {
            // Detect value type from first non-null list
            let value_type = list_arr.values().data_type();
            match value_type {
                arrow_schema::DataType::Int64 => list_element_impl!(list_arr, arrow_array::Int64Array, arrow_array::builder::Int64Builder),
                arrow_schema::DataType::Int32 => list_element_impl!(list_arr, arrow_array::Int32Array, arrow_array::builder::Int32Builder),
                arrow_schema::DataType::Int16 => list_element_impl!(list_arr, arrow_array::Int16Array, arrow_array::builder::Int16Builder),
                arrow_schema::DataType::Int8 => list_element_impl!(list_arr, arrow_array::Int8Array, arrow_array::builder::Int8Builder),
                arrow_schema::DataType::UInt64 => list_element_impl!(list_arr, arrow_array::UInt64Array, arrow_array::builder::UInt64Builder),
                arrow_schema::DataType::UInt32 => list_element_impl!(list_arr, arrow_array::UInt32Array, arrow_array::builder::UInt32Builder),
                arrow_schema::DataType::UInt16 => list_element_impl!(list_arr, arrow_array::UInt16Array, arrow_array::builder::UInt16Builder),
                arrow_schema::DataType::UInt8 => list_element_impl!(list_arr, arrow_array::UInt8Array, arrow_array::builder::UInt8Builder),
                arrow_schema::DataType::Float64 => list_element_impl!(list_arr, arrow_array::Float64Array, arrow_array::builder::Float64Builder),
                arrow_schema::DataType::Float32 => list_element_impl!(list_arr, arrow_array::Float32Array, arrow_array::builder::Float32Builder),
                arrow_schema::DataType::Boolean => list_element_impl!(list_arr, arrow_array::BooleanArray, arrow_array::builder::BooleanBuilder),
                arrow_schema::DataType::Date32 => list_element_impl!(list_arr, arrow_array::Date32Array, arrow_array::builder::Date32Builder),
                arrow_schema::DataType::Date64 => list_element_impl!(list_arr, arrow_array::Date64Array, arrow_array::builder::Date64Builder),
                arrow_schema::DataType::Utf8 => {
                    let mut builder = arrow_array::builder::StringBuilder::with_capacity(list_arr.len(), 256);
                    for i in 0..list_arr.len() {
                        if list_arr.is_null(i) {
                            builder.append_null();
                        } else {
                            let list_values = list_arr.value(i);
                            let len = list_values.len() as i64;
                            let actual_idx = if index < 0 { len + index } else { index };

                            if actual_idx < 0 || actual_idx >= len {
                                builder.append_null();
                            } else if let Some(str_arr) = list_values.as_any().downcast_ref::<arrow_array::StringArray>() {
                                if str_arr.is_null(actual_idx as usize) {
                                    builder.append_null();
                                } else {
                                    builder.append_value(str_arr.value(actual_idx as usize));
                                }
                            } else {
                                builder.append_null();
                            }
                        }
                    }
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
                }
                // Timestamp types
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => list_element_impl!(list_arr, arrow_array::TimestampSecondArray, arrow_array::builder::TimestampSecondBuilder),
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => list_element_impl!(list_arr, arrow_array::TimestampMillisecondArray, arrow_array::builder::TimestampMillisecondBuilder),
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => list_element_impl!(list_arr, arrow_array::TimestampMicrosecondArray, arrow_array::builder::TimestampMicrosecondBuilder),
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => list_element_impl!(list_arr, arrow_array::TimestampNanosecondArray, arrow_array::builder::TimestampNanosecondBuilder),
                // Duration types
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Second) => list_element_impl!(list_arr, arrow_array::DurationSecondArray, arrow_array::builder::DurationSecondBuilder),
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Millisecond) => list_element_impl!(list_arr, arrow_array::DurationMillisecondArray, arrow_array::builder::DurationMillisecondBuilder),
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Microsecond) => list_element_impl!(list_arr, arrow_array::DurationMicrosecondArray, arrow_array::builder::DurationMicrosecondBuilder),
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Nanosecond) => list_element_impl!(list_arr, arrow_array::DurationNanosecondArray, arrow_array::builder::DurationNanosecondBuilder),
                // Time types
                arrow_schema::DataType::Time32(arrow_schema::TimeUnit::Second) => list_element_impl!(list_arr, arrow_array::Time32SecondArray, arrow_array::builder::Time32SecondBuilder),
                arrow_schema::DataType::Time32(arrow_schema::TimeUnit::Millisecond) => list_element_impl!(list_arr, arrow_array::Time32MillisecondArray, arrow_array::builder::Time32MillisecondBuilder),
                arrow_schema::DataType::Time64(arrow_schema::TimeUnit::Microsecond) => list_element_impl!(list_arr, arrow_array::Time64MicrosecondArray, arrow_array::builder::Time64MicrosecondBuilder),
                arrow_schema::DataType::Time64(arrow_schema::TimeUnit::Nanosecond) => list_element_impl!(list_arr, arrow_array::Time64NanosecondArray, arrow_array::builder::Time64NanosecondBuilder),
                // Binary types
                arrow_schema::DataType::Binary => {
                    let mut builder = arrow_array::builder::BinaryBuilder::with_capacity(list_arr.len(), 256);
                    for i in 0..list_arr.len() {
                        if list_arr.is_null(i) {
                            builder.append_null();
                        } else {
                            let list_values = list_arr.value(i);
                            let len = list_values.len() as i64;
                            let actual_idx = if index < 0 { len + index } else { index };

                            if actual_idx < 0 || actual_idx >= len {
                                builder.append_null();
                            } else if let Some(bin_arr) = list_values.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                                if bin_arr.is_null(actual_idx as usize) {
                                    builder.append_null();
                                } else {
                                    builder.append_value(bin_arr.value(actual_idx as usize));
                                }
                            } else {
                                builder.append_null();
                            }
                        }
                    }
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
                }
                _ => return Err(arrays::ArrowError::NotImplemented(format!(
                    "list_element not implemented for value type {:?}",
                    value_type
                ))),
            }
        }

        // Try LargeListArray
        if let Some(list_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let value_type = list_arr.values().data_type();
            match value_type {
                arrow_schema::DataType::Int64 => list_element_impl!(list_arr, arrow_array::Int64Array, arrow_array::builder::Int64Builder),
                arrow_schema::DataType::Int32 => list_element_impl!(list_arr, arrow_array::Int32Array, arrow_array::builder::Int32Builder),
                arrow_schema::DataType::Int16 => list_element_impl!(list_arr, arrow_array::Int16Array, arrow_array::builder::Int16Builder),
                arrow_schema::DataType::Int8 => list_element_impl!(list_arr, arrow_array::Int8Array, arrow_array::builder::Int8Builder),
                arrow_schema::DataType::UInt64 => list_element_impl!(list_arr, arrow_array::UInt64Array, arrow_array::builder::UInt64Builder),
                arrow_schema::DataType::UInt32 => list_element_impl!(list_arr, arrow_array::UInt32Array, arrow_array::builder::UInt32Builder),
                arrow_schema::DataType::UInt16 => list_element_impl!(list_arr, arrow_array::UInt16Array, arrow_array::builder::UInt16Builder),
                arrow_schema::DataType::UInt8 => list_element_impl!(list_arr, arrow_array::UInt8Array, arrow_array::builder::UInt8Builder),
                arrow_schema::DataType::Float64 => list_element_impl!(list_arr, arrow_array::Float64Array, arrow_array::builder::Float64Builder),
                arrow_schema::DataType::Float32 => list_element_impl!(list_arr, arrow_array::Float32Array, arrow_array::builder::Float32Builder),
                arrow_schema::DataType::Boolean => list_element_impl!(list_arr, arrow_array::BooleanArray, arrow_array::builder::BooleanBuilder),
                arrow_schema::DataType::Date32 => list_element_impl!(list_arr, arrow_array::Date32Array, arrow_array::builder::Date32Builder),
                arrow_schema::DataType::Date64 => list_element_impl!(list_arr, arrow_array::Date64Array, arrow_array::builder::Date64Builder),
                arrow_schema::DataType::Utf8 => {
                    let mut builder = arrow_array::builder::StringBuilder::with_capacity(list_arr.len(), 256);
                    for i in 0..list_arr.len() {
                        if list_arr.is_null(i) {
                            builder.append_null();
                        } else {
                            let list_values = list_arr.value(i);
                            let len = list_values.len() as i64;
                            let actual_idx = if index < 0 { len + index } else { index };

                            if actual_idx < 0 || actual_idx >= len {
                                builder.append_null();
                            } else if let Some(str_arr) = list_values.as_any().downcast_ref::<arrow_array::StringArray>() {
                                if str_arr.is_null(actual_idx as usize) {
                                    builder.append_null();
                                } else {
                                    builder.append_value(str_arr.value(actual_idx as usize));
                                }
                            } else {
                                builder.append_null();
                            }
                        }
                    }
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
                }
                // Timestamp types
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => list_element_impl!(list_arr, arrow_array::TimestampSecondArray, arrow_array::builder::TimestampSecondBuilder),
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => list_element_impl!(list_arr, arrow_array::TimestampMillisecondArray, arrow_array::builder::TimestampMillisecondBuilder),
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => list_element_impl!(list_arr, arrow_array::TimestampMicrosecondArray, arrow_array::builder::TimestampMicrosecondBuilder),
                arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => list_element_impl!(list_arr, arrow_array::TimestampNanosecondArray, arrow_array::builder::TimestampNanosecondBuilder),
                // Duration types
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Second) => list_element_impl!(list_arr, arrow_array::DurationSecondArray, arrow_array::builder::DurationSecondBuilder),
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Millisecond) => list_element_impl!(list_arr, arrow_array::DurationMillisecondArray, arrow_array::builder::DurationMillisecondBuilder),
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Microsecond) => list_element_impl!(list_arr, arrow_array::DurationMicrosecondArray, arrow_array::builder::DurationMicrosecondBuilder),
                arrow_schema::DataType::Duration(arrow_schema::TimeUnit::Nanosecond) => list_element_impl!(list_arr, arrow_array::DurationNanosecondArray, arrow_array::builder::DurationNanosecondBuilder),
                // Time types
                arrow_schema::DataType::Time32(arrow_schema::TimeUnit::Second) => list_element_impl!(list_arr, arrow_array::Time32SecondArray, arrow_array::builder::Time32SecondBuilder),
                arrow_schema::DataType::Time32(arrow_schema::TimeUnit::Millisecond) => list_element_impl!(list_arr, arrow_array::Time32MillisecondArray, arrow_array::builder::Time32MillisecondBuilder),
                arrow_schema::DataType::Time64(arrow_schema::TimeUnit::Microsecond) => list_element_impl!(list_arr, arrow_array::Time64MicrosecondArray, arrow_array::builder::Time64MicrosecondBuilder),
                arrow_schema::DataType::Time64(arrow_schema::TimeUnit::Nanosecond) => list_element_impl!(list_arr, arrow_array::Time64NanosecondArray, arrow_array::builder::Time64NanosecondBuilder),
                // Binary types
                arrow_schema::DataType::Binary => {
                    let mut builder = arrow_array::builder::BinaryBuilder::with_capacity(list_arr.len(), 256);
                    for i in 0..list_arr.len() {
                        if list_arr.is_null(i) {
                            builder.append_null();
                        } else {
                            let list_values = list_arr.value(i);
                            let len = list_values.len() as i64;
                            let actual_idx = if index < 0 { len + index } else { index };

                            if actual_idx < 0 || actual_idx >= len {
                                builder.append_null();
                            } else if let Some(bin_arr) = list_values.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                                if bin_arr.is_null(actual_idx as usize) {
                                    builder.append_null();
                                } else {
                                    builder.append_value(bin_arr.value(actual_idx as usize));
                                }
                            } else {
                                builder.append_null();
                            }
                        }
                    }
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }));
                }
                _ => return Err(arrays::ArrowError::NotImplemented(format!(
                    "list_element not implemented for value type {:?}",
                    value_type
                ))),
            }
        }

        Err(arrays::ArrowError::InvalidArgument("list_element requires a list array".to_string()))
    }

    fn arrays_to_list(input_arrays: Vec<arrays::Array>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;

        if input_arrays.is_empty() {
            return Err(arrays::ArrowError::InvalidArgument("arrays_to_list requires at least one array".to_string()));
        }

        // Get the first array to determine type
        let first = input_arrays[0].get::<ArrayImpl>();
        let data_type = first.inner.data_type().clone();
        let num_rows = input_arrays.len();

        // Build a ListArray where each input array becomes one row's list
        match &data_type {
            arrow_schema::DataType::Int64 => {
                let mut offsets = vec![0i32];
                let mut values: Vec<i64> = Vec::new();
                let mut null_mask: Vec<bool> = Vec::new();

                for arr in &input_arrays {
                    let arr_impl = arr.get::<ArrayImpl>();
                    if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
                        for i in 0..int_arr.len() {
                            if int_arr.is_null(i) {
                                values.push(0);
                                null_mask.push(false);
                            } else {
                                values.push(int_arr.value(i));
                                null_mask.push(true);
                            }
                        }
                        offsets.push(values.len() as i32);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("All arrays must have the same type".to_string()));
                    }
                }

                let values_arr = arrow_array::Int64Array::from(values);
                let list_arr = arrow_array::ListArray::try_new(
                    Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Int64, true)),
                    arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
                    Arc::new(values_arr),
                    None,
                ).map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;

                Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_arr) }))
            }
            arrow_schema::DataType::Float64 => {
                let mut offsets = vec![0i32];
                let mut values: Vec<f64> = Vec::new();

                for arr in &input_arrays {
                    let arr_impl = arr.get::<ArrayImpl>();
                    if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
                        for i in 0..float_arr.len() {
                            values.push(if float_arr.is_null(i) { f64::NAN } else { float_arr.value(i) });
                        }
                        offsets.push(values.len() as i32);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("All arrays must have the same type".to_string()));
                    }
                }

                let values_arr = arrow_array::Float64Array::from(values);
                let list_arr = arrow_array::ListArray::try_new(
                    Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float64, true)),
                    arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
                    Arc::new(values_arr),
                    None,
                ).map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;

                Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_arr) }))
            }
            arrow_schema::DataType::Int32 => {
                let mut offsets = vec![0i32];
                let mut values: Vec<i32> = Vec::new();

                for arr in &input_arrays {
                    let arr_impl = arr.get::<ArrayImpl>();
                    if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
                        for i in 0..int_arr.len() {
                            values.push(if int_arr.is_null(i) { 0 } else { int_arr.value(i) });
                        }
                        offsets.push(values.len() as i32);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("All arrays must have the same type".to_string()));
                    }
                }

                let values_arr = arrow_array::Int32Array::from(values);
                let list_arr = arrow_array::ListArray::try_new(
                    Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Int32, true)),
                    arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
                    Arc::new(values_arr),
                    None,
                ).map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;

                Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_arr) }))
            }
            arrow_schema::DataType::Float32 => {
                let mut offsets = vec![0i32];
                let mut values: Vec<f32> = Vec::new();

                for arr in &input_arrays {
                    let arr_impl = arr.get::<ArrayImpl>();
                    if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
                        for i in 0..float_arr.len() {
                            values.push(if float_arr.is_null(i) { f32::NAN } else { float_arr.value(i) });
                        }
                        offsets.push(values.len() as i32);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("All arrays must have the same type".to_string()));
                    }
                }

                let values_arr = arrow_array::Float32Array::from(values);
                let list_arr = arrow_array::ListArray::try_new(
                    Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Float32, true)),
                    arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
                    Arc::new(values_arr),
                    None,
                ).map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;

                Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_arr) }))
            }
            arrow_schema::DataType::Boolean => {
                let mut offsets = vec![0i32];
                let mut values: Vec<bool> = Vec::new();

                for arr in &input_arrays {
                    let arr_impl = arr.get::<ArrayImpl>();
                    if let Some(bool_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>() {
                        for i in 0..bool_arr.len() {
                            values.push(if bool_arr.is_null(i) { false } else { bool_arr.value(i) });
                        }
                        offsets.push(values.len() as i32);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("All arrays must have the same type".to_string()));
                    }
                }

                let values_arr = arrow_array::BooleanArray::from(values);
                let list_arr = arrow_array::ListArray::try_new(
                    Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Boolean, true)),
                    arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
                    Arc::new(values_arr),
                    None,
                ).map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;

                Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_arr) }))
            }
            arrow_schema::DataType::Utf8 => {
                let mut offsets = vec![0i32];
                let mut values: Vec<Option<String>> = Vec::new();

                for arr in &input_arrays {
                    let arr_impl = arr.get::<ArrayImpl>();
                    if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
                        for i in 0..str_arr.len() {
                            if str_arr.is_null(i) {
                                values.push(None);
                            } else {
                                values.push(Some(str_arr.value(i).to_string()));
                            }
                        }
                        offsets.push(values.len() as i32);
                    } else {
                        return Err(arrays::ArrowError::InvalidArgument("All arrays must have the same type".to_string()));
                    }
                }

                let values_arr: arrow_array::StringArray = values.into_iter().collect();
                let list_arr = arrow_array::ListArray::try_new(
                    Arc::new(arrow_schema::Field::new("item", arrow_schema::DataType::Utf8, true)),
                    arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(offsets)),
                    Arc::new(values_arr),
                    None,
                ).map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;

                Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_arr) }))
            }
            _ => Err(arrays::ArrowError::NotImplemented(format!(
                "arrays_to_list not implemented for type {:?}",
                data_type
            ))),
        }
    }

    // ========== Struct Array Operations ==========

    fn struct_field(arr: arrays::ArrayBorrow<'_>, index: u32) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(struct_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StructArray>() {
            if (index as usize) >= struct_arr.num_columns() {
                return Err(arrays::ArrowError::OutOfBounds(format!(
                    "Field index {} out of bounds (struct has {} fields)",
                    index, struct_arr.num_columns()
                )));
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: struct_arr.column(index as usize).clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("struct_field requires a struct array".to_string()))
    }

    fn struct_field_by_name(arr: arrays::ArrayBorrow<'_>, name: String) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(struct_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StructArray>() {
            if let Some(col) = struct_arr.column_by_name(&name) {
                return Ok(arrays::Array::new(ArrayImpl { inner: col.clone() }));
            }
            return Err(arrays::ArrowError::InvalidArgument(format!(
                "Field '{}' not found in struct",
                name
            )));
        }

        Err(arrays::ArrowError::InvalidArgument("struct_field_by_name requires a struct array".to_string()))
    }

    fn struct_field_names(arr: arrays::ArrayBorrow<'_>) -> Result<Vec<String>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(struct_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StructArray>() {
            let names: Vec<String> = struct_arr.fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect();
            return Ok(names);
        }

        Err(arrays::ArrowError::InvalidArgument("struct_field_names requires a struct array".to_string()))
    }

    fn struct_num_fields(arr: arrays::ArrayBorrow<'_>) -> Result<u32, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(struct_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StructArray>() {
            return Ok(struct_arr.num_columns() as u32);
        }

        Err(arrays::ArrowError::InvalidArgument("struct_num_fields requires a struct array".to_string()))
    }

    // ========== Map Array Operations ==========

    fn map_keys(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(map_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::MapArray>() {
            let keys = map_arr.keys();
            return Ok(arrays::Array::new(ArrayImpl { inner: keys.clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("map_keys requires a map array".to_string()))
    }

    fn map_values(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(map_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::MapArray>() {
            let values = map_arr.values();
            return Ok(arrays::Array::new(ArrayImpl { inner: values.clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("map_values requires a map array".to_string()))
    }

    fn map_offsets(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(map_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::MapArray>() {
            // Get offsets as Int32Array
            let offsets = map_arr.offsets();
            let offsets_arr: arrow_array::Int32Array = offsets.iter().map(|o| Some(*o)).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(offsets_arr) }));
        }

        Err(arrays::ArrowError::InvalidArgument("map_offsets requires a map array".to_string()))
    }

    // ========== FixedSizeList Array Operations ==========

    fn fixed_list_values(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(fsl_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeListArray>() {
            let values = fsl_arr.values();
            return Ok(arrays::Array::new(ArrayImpl { inner: values.clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("fixed_list_values requires a fixed-size list array".to_string()))
    }

    fn fixed_list_size(arr: arrays::ArrayBorrow<'_>) -> Result<u32, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(fsl_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeListArray>() {
            return Ok(fsl_arr.value_length() as u32);
        }

        Err(arrays::ArrowError::InvalidArgument("fixed_list_size requires a fixed-size list array".to_string()))
    }

    // ========== Union Array Operations ==========

    fn union_type_ids(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(union_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UnionArray>() {
            let type_ids = union_arr.type_ids();
            // Convert &[i8] to Int8Array
            let type_ids_arr: arrow_array::Int8Array = type_ids.iter().map(|id| Some(*id)).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(type_ids_arr) }));
        }

        Err(arrays::ArrowError::InvalidArgument("union_type_ids requires a union array".to_string()))
    }

    fn union_child(arr: arrays::ArrayBorrow<'_>, type_id: u8) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(union_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UnionArray>() {
            let child = union_arr.child(type_id as i8);
            return Ok(arrays::Array::new(ArrayImpl { inner: child.clone() }));
        }

        Err(arrays::ArrowError::InvalidArgument("union_child requires a union array".to_string()))
    }

    fn union_children(arr: arrays::ArrayBorrow<'_>) -> Result<Vec<arrays::Array>, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(union_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UnionArray>() {
            // Get all unique type IDs and their children
            let type_ids = union_arr.type_ids();
            let mut seen_ids = std::collections::HashSet::new();
            let mut children = Vec::new();

            for &type_id in type_ids {
                if seen_ids.insert(type_id) {
                    let child = union_arr.child(type_id);
                    children.push(arrays::Array::new(ArrayImpl { inner: child.clone() }));
                }
            }

            return Ok(children);
        }

        Err(arrays::ArrowError::InvalidArgument("union_children requires a union array".to_string()))
    }

    // ========== Run-End Encoded Array Operations ==========

    fn ree_encode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Int32 arrays
        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            if typed_arr.len() == 0 {
                let run_ends = arrow_array::Int64Array::from(Vec::<i64>::new());
                let values = arrow_array::Int32Array::from(Vec::<i32>::new());
                let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends, &values)
                    .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
            }

            let mut run_ends: Vec<i64> = Vec::new();
            let mut values: Vec<Option<i32>> = Vec::new();
            let mut prev_val: Option<i32> = None;

            for (i, val) in typed_arr.iter().enumerate() {
                if i == 0 {
                    prev_val = val;
                } else if val != prev_val {
                    run_ends.push(i as i64);
                    values.push(prev_val);
                    prev_val = val;
                }
            }
            run_ends.push(typed_arr.len() as i64);
            values.push(prev_val);

            let run_ends_arr = arrow_array::Int64Array::from(run_ends);
            let values_arr: arrow_array::Int32Array = values.into_iter().collect();
            let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends_arr, &values_arr)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
        }

        // Int64 arrays
        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            if typed_arr.len() == 0 {
                let run_ends = arrow_array::Int64Array::from(Vec::<i64>::new());
                let values = arrow_array::Int64Array::from(Vec::<i64>::new());
                let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends, &values)
                    .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
            }

            let mut run_ends: Vec<i64> = Vec::new();
            let mut values: Vec<Option<i64>> = Vec::new();
            let mut prev_val: Option<i64> = None;

            for (i, val) in typed_arr.iter().enumerate() {
                if i == 0 {
                    prev_val = val;
                } else if val != prev_val {
                    run_ends.push(i as i64);
                    values.push(prev_val);
                    prev_val = val;
                }
            }
            run_ends.push(typed_arr.len() as i64);
            values.push(prev_val);

            let run_ends_arr = arrow_array::Int64Array::from(run_ends);
            let values_arr: arrow_array::Int64Array = values.into_iter().collect();
            let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends_arr, &values_arr)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
        }

        // String arrays
        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            if typed_arr.len() == 0 {
                let run_ends = arrow_array::Int64Array::from(Vec::<i64>::new());
                let values = arrow_array::StringArray::from(Vec::<Option<&str>>::new());
                let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends, &values)
                    .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
            }

            let mut run_ends: Vec<i64> = Vec::new();
            let mut values: Vec<Option<String>> = Vec::new();
            let mut prev_val: Option<&str> = None;

            for (i, val) in typed_arr.iter().enumerate() {
                if i == 0 {
                    prev_val = val;
                } else if val != prev_val {
                    run_ends.push(i as i64);
                    values.push(prev_val.map(|s| s.to_string()));
                    prev_val = val;
                }
            }
            run_ends.push(typed_arr.len() as i64);
            values.push(prev_val.map(|s| s.to_string()));

            let run_ends_arr = arrow_array::Int64Array::from(run_ends);
            let values_arr: arrow_array::StringArray = values.into_iter().collect();
            let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends_arr, &values_arr)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
        }

        // Macro for primitive types with simpler iteration
        macro_rules! ree_encode_primitive {
            ($arr_type:ty, $val_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    if typed_arr.len() == 0 {
                        let run_ends = arrow_array::Int64Array::from(Vec::<i64>::new());
                        let values: $arr_type = Vec::<Option<$val_type>>::new().into_iter().collect();
                        let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends, &values)
                            .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                        return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
                    }

                    let mut run_ends: Vec<i64> = Vec::new();
                    let mut values: Vec<Option<$val_type>> = Vec::new();
                    let mut prev_val: Option<$val_type> = None;

                    for (i, val) in typed_arr.iter().enumerate() {
                        if i == 0 {
                            prev_val = val;
                        } else if val != prev_val {
                            run_ends.push(i as i64);
                            values.push(prev_val);
                            prev_val = val;
                        }
                    }
                    run_ends.push(typed_arr.len() as i64);
                    values.push(prev_val);

                    let run_ends_arr = arrow_array::Int64Array::from(run_ends);
                    let values_arr: $arr_type = values.into_iter().collect();
                    let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends_arr, &values_arr)
                        .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
                }
            };
        }

        // Additional integer types
        ree_encode_primitive!(arrow_array::Int8Array, i8);
        ree_encode_primitive!(arrow_array::Int16Array, i16);
        ree_encode_primitive!(arrow_array::UInt8Array, u8);
        ree_encode_primitive!(arrow_array::UInt16Array, u16);
        ree_encode_primitive!(arrow_array::UInt32Array, u32);
        ree_encode_primitive!(arrow_array::UInt64Array, u64);

        // Boolean arrays
        ree_encode_primitive!(arrow_array::BooleanArray, bool);

        // Date types
        ree_encode_primitive!(arrow_array::Date32Array, i32);
        ree_encode_primitive!(arrow_array::Date64Array, i64);

        // Float types
        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            if typed_arr.len() == 0 {
                let run_ends = arrow_array::Int64Array::from(Vec::<i64>::new());
                let values = arrow_array::Float32Array::from(Vec::<f32>::new());
                let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends, &values)
                    .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
            }

            let mut run_ends: Vec<i64> = Vec::new();
            let mut values: Vec<Option<f32>> = Vec::new();
            let mut prev_val: Option<f32> = None;

            for (i, val) in typed_arr.iter().enumerate() {
                if i == 0 {
                    prev_val = val;
                } else {
                    // Float comparison: treat NaN as equal to NaN
                    let changed = match (val, prev_val) {
                        (Some(a), Some(b)) => {
                            if a.is_nan() && b.is_nan() { false }
                            else { (a - b).abs() > f32::EPSILON }
                        }
                        (None, None) => false,
                        _ => true,
                    };
                    if changed {
                        run_ends.push(i as i64);
                        values.push(prev_val);
                        prev_val = val;
                    }
                }
            }
            run_ends.push(typed_arr.len() as i64);
            values.push(prev_val);

            let run_ends_arr = arrow_array::Int64Array::from(run_ends);
            let values_arr: arrow_array::Float32Array = values.into_iter().collect();
            let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends_arr, &values_arr)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
        }

        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            if typed_arr.len() == 0 {
                let run_ends = arrow_array::Int64Array::from(Vec::<i64>::new());
                let values = arrow_array::Float64Array::from(Vec::<f64>::new());
                let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends, &values)
                    .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
            }

            let mut run_ends: Vec<i64> = Vec::new();
            let mut values: Vec<Option<f64>> = Vec::new();
            let mut prev_val: Option<f64> = None;

            for (i, val) in typed_arr.iter().enumerate() {
                if i == 0 {
                    prev_val = val;
                } else {
                    // Float comparison: treat NaN as equal to NaN
                    let changed = match (val, prev_val) {
                        (Some(a), Some(b)) => {
                            if a.is_nan() && b.is_nan() { false }
                            else { (a - b).abs() > f64::EPSILON }
                        }
                        (None, None) => false,
                        _ => true,
                    };
                    if changed {
                        run_ends.push(i as i64);
                        values.push(prev_val);
                        prev_val = val;
                    }
                }
            }
            run_ends.push(typed_arr.len() as i64);
            values.push(prev_val);

            let run_ends_arr = arrow_array::Int64Array::from(run_ends);
            let values_arr: arrow_array::Float64Array = values.into_iter().collect();
            let ree = arrow_array::RunArray::<arrow_array::types::Int64Type>::try_new(&run_ends_arr, &values_arr)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(ree) }));
        }

        Err(arrays::ArrowError::NotImplemented("ree_encode not implemented for this array type".to_string()))
    }

    fn ree_decode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Try to decode Int64 run-end encoded array
        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int64Type>>() {
            let run_ends = ree_arr.run_ends();
            let values = ree_arr.values();
            let len = ree_arr.len();

            // Build indices to expand the values
            let mut indices: Vec<u64> = Vec::with_capacity(len);
            let run_ends_values = run_ends.values();

            for (run_idx, &end) in run_ends_values.iter().enumerate() {
                let start = if run_idx == 0 { 0 } else { run_ends_values[run_idx - 1] as usize };
                for _ in start..(end as usize) {
                    indices.push(run_idx as u64);
                }
            }

            let indices_arr: arrow_array::UInt64Array = indices.into_iter().map(Some).collect();
            let decoded = arrow_select::take::take(values.as_ref(), &indices_arr, None)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: decoded }));
        }

        // Try to decode Int32 run-end encoded array
        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>() {
            let run_ends = ree_arr.run_ends();
            let values = ree_arr.values();
            let len = ree_arr.len();

            // Build indices to expand the values
            let mut indices: Vec<u64> = Vec::with_capacity(len);
            let run_ends_values = run_ends.values();

            for (run_idx, &end) in run_ends_values.iter().enumerate() {
                let start = if run_idx == 0 { 0 } else { run_ends_values[run_idx - 1] as usize };
                for _ in start..(end as usize) {
                    indices.push(run_idx as u64);
                }
            }

            let indices_arr: arrow_array::UInt64Array = indices.into_iter().map(Some).collect();
            let decoded = arrow_select::take::take(values.as_ref(), &indices_arr, None)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: decoded }));
        }

        // Try to decode Int16 run-end encoded array
        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int16Type>>() {
            let run_ends = ree_arr.run_ends();
            let values = ree_arr.values();
            let len = ree_arr.len();

            let mut indices: Vec<u64> = Vec::with_capacity(len);
            let run_ends_values = run_ends.values();

            for (run_idx, &end) in run_ends_values.iter().enumerate() {
                let start = if run_idx == 0 { 0 } else { run_ends_values[run_idx - 1] as usize };
                for _ in start..(end as usize) {
                    indices.push(run_idx as u64);
                }
            }

            let indices_arr: arrow_array::UInt64Array = indices.into_iter().map(Some).collect();
            let decoded = arrow_select::take::take(values.as_ref(), &indices_arr, None)
                .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: decoded }));
        }

        Err(arrays::ArrowError::NotImplemented("ree_decode requires a run-end encoded array".to_string()))
    }

    fn ree_run_ends(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int64Type>>() {
            let run_ends = ree_arr.run_ends();
            // Convert RunEndBuffer to Int64Array
            let values: Vec<i64> = run_ends.values().iter().copied().collect();
            let result = arrow_array::Int64Array::from(values);
            return Ok(arrays::Array::new(ArrayImpl {
                inner: Arc::new(result)
            }));
        }

        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>() {
            let run_ends = ree_arr.run_ends();
            // Convert RunEndBuffer to Int32Array
            let values: Vec<i32> = run_ends.values().iter().copied().collect();
            let result = arrow_array::Int32Array::from(values);
            return Ok(arrays::Array::new(ArrayImpl {
                inner: Arc::new(result)
            }));
        }

        Err(arrays::ArrowError::InvalidArgument("ree_run_ends requires a run-end encoded array".to_string()))
    }

    fn ree_values(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int64Type>>() {
            let values = ree_arr.values();
            return Ok(arrays::Array::new(ArrayImpl {
                inner: values.clone()
            }));
        }

        if let Some(ree_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::RunArray<arrow_array::types::Int32Type>>() {
            let values = ree_arr.values();
            return Ok(arrays::Array::new(ArrayImpl {
                inner: values.clone()
            }));
        }

        Err(arrays::ArrowError::InvalidArgument("ree_values requires a run-end encoded array".to_string()))
    }

    // ========== Binary Data Operations ==========

    fn binary_get_byte(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let idx = index as usize;

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            let result: arrow_array::Int32Array = bin_arr.iter()
                .map(|opt| opt.and_then(|bytes| {
                    bytes.get(idx).map(|&b| b as i32)
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>() {
            let result: arrow_array::Int32Array = bin_arr.iter()
                .map(|opt| opt.and_then(|bytes| {
                    bytes.get(idx).map(|&b| b as i32)
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Expected binary array".to_string()))
    }

    fn binary_slice(arr: arrays::ArrayBorrow<'_>, start: i64, length: Option<u64>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            let result: arrow_array::BinaryArray = bin_arr.iter()
                .map(|opt| opt.map(|bytes| {
                    let len = bytes.len() as i64;
                    let actual_start = if start < 0 {
                        (len + start).max(0) as usize
                    } else {
                        start.min(len) as usize
                    };
                    let end = match length {
                        Some(l) => (actual_start + l as usize).min(bytes.len()),
                        None => bytes.len(),
                    };
                    &bytes[actual_start..end]
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>() {
            let result: arrow_array::LargeBinaryArray = bin_arr.iter()
                .map(|opt| opt.map(|bytes| {
                    let len = bytes.len() as i64;
                    let actual_start = if start < 0 {
                        (len + start).max(0) as usize
                    } else {
                        start.min(len) as usize
                    };
                    let end = match length {
                        Some(l) => (actual_start + l as usize).min(bytes.len()),
                        None => bytes.len(),
                    };
                    &bytes[actual_start..end]
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Expected binary array".to_string()))
    }

    fn binary_concat(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as _;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(left_arr), Some(right_arr)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>(),
        ) {
            if left_arr.len() != right_arr.len() {
                return Err(arrays::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::BinaryArray = left_arr.iter()
                .zip(right_arr.iter())
                .map(|(l, r)| match (l, r) {
                    (Some(lb), Some(rb)) => {
                        let mut concat = lb.to_vec();
                        concat.extend_from_slice(rb);
                        Some(concat)
                    }
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let (Some(left_arr), Some(right_arr)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>(),
        ) {
            if left_arr.len() != right_arr.len() {
                return Err(arrays::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::LargeBinaryArray = left_arr.iter()
                .zip(right_arr.iter())
                .map(|(l, r)| match (l, r) {
                    (Some(lb), Some(rb)) => {
                        let mut concat = lb.to_vec();
                        concat.extend_from_slice(rb);
                        Some(concat)
                    }
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Expected binary arrays of same type".to_string()))
    }

    fn binary_length(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as _;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            let result: arrow_array::Int64Array = bin_arr.iter()
                .map(|opt| opt.map(|bytes| bytes.len() as i64))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>() {
            let result: arrow_array::Int64Array = bin_arr.iter()
                .map(|opt| opt.map(|bytes| bytes.len() as i64))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(fsb_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeBinaryArray>() {
            // All elements have the same size
            let size = fsb_arr.value_length() as i64;
            let result: arrow_array::Int64Array = (0..fsb_arr.len())
                .map(|i| if fsb_arr.is_valid(i) { Some(size) } else { None })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Expected binary array".to_string()))
    }

    fn binary_to_hex(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        use arrow_array::Array as _;
        let arr_impl = arr.get::<ArrayImpl>();

        fn bytes_to_hex(bytes: &[u8]) -> String {
            bytes.iter().map(|b| format!("{:02x}", b)).collect()
        }

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            let result: arrow_array::StringArray = bin_arr.iter()
                .map(|opt| opt.map(bytes_to_hex))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>() {
            let result: arrow_array::StringArray = bin_arr.iter()
                .map(|opt| opt.map(bytes_to_hex))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(fsb_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeBinaryArray>() {
            let result: arrow_array::StringArray = (0..fsb_arr.len())
                .map(|i| {
                    if fsb_arr.is_valid(i) {
                        Some(bytes_to_hex(fsb_arr.value(i)))
                    } else {
                        None
                    }
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Expected binary array".to_string()))
    }

    fn hex_to_binary(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
            let hex = hex.trim();
            if hex.len() % 2 != 0 {
                return None;
            }
            (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
                .collect()
        }

        if let Some(string_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let result: arrow_array::BinaryArray = string_arr.iter()
                .map(|opt| opt.and_then(hex_to_bytes))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(string_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let result: arrow_array::LargeBinaryArray = string_arr.iter()
                .map(|opt| opt.and_then(hex_to_bytes))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Expected string array".to_string()))
    }

    // ========== Array Generation Utilities ==========

    fn repeat_i64(value: i64, count: u64) -> arrays::Array {
        let values: Vec<i64> = vec![value; count as usize];
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::Int64Array::from(values)),
        })
    }

    fn repeat_f64(value: f64, count: u64) -> arrays::Array {
        let values: Vec<f64> = vec![value; count as usize];
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::Float64Array::from(values)),
        })
    }

    fn repeat_string(value: String, count: u64) -> arrays::Array {
        let values: Vec<&str> = vec![value.as_str(); count as usize];
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::StringArray::from(values)),
        })
    }

    fn repeat_bool(value: bool, count: u64) -> arrays::Array {
        let values: Vec<bool> = vec![value; count as usize];
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::BooleanArray::from(values)),
        })
    }

    fn range_i64(start: i64, stop: i64, step: i64) -> Result<arrays::Array, arrays::ArrowError> {
        if step == 0 {
            return Err(arrays::ArrowError::InvalidArgument("step cannot be zero".to_string()));
        }

        let values: Vec<i64> = if step > 0 {
            (start..stop).step_by(step as usize).collect()
        } else {
            // For negative step, we need to go from start down to stop
            let mut v = Vec::new();
            let mut current = start;
            while current > stop {
                v.push(current);
                current += step;
            }
            v
        };

        Ok(arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::Int64Array::from(values)),
        }))
    }

    fn range_f64(start: f64, stop: f64, step: f64) -> Result<arrays::Array, arrays::ArrowError> {
        if step == 0.0 {
            return Err(arrays::ArrowError::InvalidArgument("step cannot be zero".to_string()));
        }

        let mut values: Vec<f64> = Vec::new();
        let mut current = start;

        if step > 0.0 {
            while current < stop {
                values.push(current);
                current += step;
            }
        } else {
            while current > stop {
                values.push(current);
                current += step;
            }
        }

        Ok(arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::Float64Array::from(values)),
        }))
    }

    fn range_date(start: i32, stop: i32, step: i32) -> Result<arrays::Array, arrays::ArrowError> {
        if step == 0 {
            return Err(arrays::ArrowError::InvalidArgument("step cannot be zero".to_string()));
        }

        let values: Vec<i32> = if step > 0 {
            (start..stop).step_by(step as usize).collect()
        } else {
            let mut v = Vec::new();
            let mut current = start;
            while current > stop {
                v.push(current);
                current += step;
            }
            v
        };

        Ok(arrays::Array::new(ArrayImpl {
            inner: Arc::new(arrow_array::Date32Array::from(values)),
        }))
    }
}

struct ArrayImpl {
    inner: ArrayRef,
}

impl arrays::GuestArray for ArrayImpl {
    fn data_type(&self) -> types::DataType {
        convert::from_arrow_data_type(self.inner.data_type())
    }

    fn len(&self) -> u64 {
        self.inner.len() as u64
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn null_count(&self) -> u64 {
        self.inner.null_count() as u64
    }

    fn is_null(&self, index: u64) -> bool {
        self.inner.is_null(index as usize)
    }

    fn is_valid(&self, index: u64) -> bool {
        self.inner.is_valid(index as usize)
    }

    fn slice(&self, offset: u64, length: u64) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: self.inner.slice(offset as usize, length as usize),
        })
    }

    fn get_buffer_memory_size(&self) -> u64 {
        self.inner.get_buffer_memory_size() as u64
    }

    fn get_array_memory_size(&self) -> u64 {
        self.inner.get_array_memory_size() as u64
    }
}

// Builder implementations - using RefCell for interior mutability since WIT generates &self methods
macro_rules! impl_builder {
    ($name:ident, $guest:ident, $builder:ty, $type:ty, $resource:ident) => {
        struct $name {
            inner: RefCell<$builder>,
        }

        impl arrays::$guest for $name {
            fn new() -> Self {
                Self { inner: RefCell::new(<$builder>::new()) }
            }

            fn with_capacity(capacity: u64) -> arrays::$resource {
                arrays::$resource::new(Self {
                    inner: RefCell::new(<$builder>::with_capacity(capacity as usize)),
                })
            }

            fn append_value(&self, value: $type) {
                self.inner.borrow_mut().append_value(value);
            }

            fn append_null(&self) {
                self.inner.borrow_mut().append_null();
            }

            fn append_values(&self, values: Vec<$type>, is_valid: Vec<bool>) {
                let mut builder = self.inner.borrow_mut();
                for (v, valid) in values.into_iter().zip(is_valid.into_iter()) {
                    if valid {
                        builder.append_value(v);
                    } else {
                        builder.append_null();
                    }
                }
            }

            fn len(&self) -> u64 {
                self.inner.borrow().len() as u64
            }

            fn finish(&self) -> arrays::Array {
                arrays::Array::new(ArrayImpl {
                    inner: Arc::new(self.inner.borrow_mut().finish()),
                })
            }
        }
    };
}

impl_builder!(BooleanArrayBuilderImpl, GuestBooleanArrayBuilder, BooleanBuilder, bool, BooleanArrayBuilder);
impl_builder!(Int8ArrayBuilderImpl, GuestInt8ArrayBuilder, Int8Builder, i8, Int8ArrayBuilder);
impl_builder!(Int16ArrayBuilderImpl, GuestInt16ArrayBuilder, Int16Builder, i16, Int16ArrayBuilder);
impl_builder!(Int32ArrayBuilderImpl, GuestInt32ArrayBuilder, Int32Builder, i32, Int32ArrayBuilder);
impl_builder!(Int64ArrayBuilderImpl, GuestInt64ArrayBuilder, Int64Builder, i64, Int64ArrayBuilder);
impl_builder!(Uint8ArrayBuilderImpl, GuestUint8ArrayBuilder, UInt8Builder, u8, Uint8ArrayBuilder);
impl_builder!(Uint16ArrayBuilderImpl, GuestUint16ArrayBuilder, UInt16Builder, u16, Uint16ArrayBuilder);
impl_builder!(Uint32ArrayBuilderImpl, GuestUint32ArrayBuilder, UInt32Builder, u32, Uint32ArrayBuilder);
impl_builder!(Uint64ArrayBuilderImpl, GuestUint64ArrayBuilder, UInt64Builder, u64, Uint64ArrayBuilder);
impl_builder!(Float32ArrayBuilderImpl, GuestFloat32ArrayBuilder, Float32Builder, f32, Float32ArrayBuilder);
impl_builder!(Float64ArrayBuilderImpl, GuestFloat64ArrayBuilder, Float64Builder, f64, Float64ArrayBuilder);

struct StringArrayBuilderImpl {
    inner: RefCell<StringBuilder>,
}

impl arrays::GuestStringArrayBuilder for StringArrayBuilderImpl {
    fn new() -> Self {
        Self { inner: RefCell::new(StringBuilder::new()) }
    }

    fn with_capacity(capacity: u64) -> arrays::StringArrayBuilder {
        arrays::StringArrayBuilder::new(Self {
            inner: RefCell::new(StringBuilder::with_capacity(capacity as usize, 0)),
        })
    }

    fn append_value(&self, value: String) {
        self.inner.borrow_mut().append_value(&value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn append_option(&self, value: Option<String>) {
        self.inner.borrow_mut().append_option(value);
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

struct BinaryArrayBuilderImpl {
    inner: RefCell<BinaryBuilder>,
}

impl arrays::GuestBinaryArrayBuilder for BinaryArrayBuilderImpl {
    fn new() -> Self {
        Self { inner: RefCell::new(BinaryBuilder::new()) }
    }

    fn with_capacity(capacity: u64) -> arrays::BinaryArrayBuilder {
        arrays::BinaryArrayBuilder::new(Self {
            inner: RefCell::new(BinaryBuilder::with_capacity(capacity as usize, 0)),
        })
    }

    fn append_value(&self, value: Vec<u8>) {
        self.inner.borrow_mut().append_value(&value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

// ============================================================================
// Large String/Binary and FixedSize Builders
// ============================================================================

struct LargeStringArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::LargeStringBuilder>,
}

impl arrays::GuestLargeStringArrayBuilder for LargeStringArrayBuilderImpl {
    fn new() -> Self {
        Self { inner: RefCell::new(arrow_array::builder::LargeStringBuilder::new()) }
    }

    fn with_capacity(capacity: u64) -> arrays::LargeStringArrayBuilder {
        arrays::LargeStringArrayBuilder::new(Self {
            inner: RefCell::new(arrow_array::builder::LargeStringBuilder::with_capacity(capacity as usize, 0)),
        })
    }

    fn append_value(&self, value: String) {
        self.inner.borrow_mut().append_value(&value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn append_option(&self, value: Option<String>) {
        self.inner.borrow_mut().append_option(value);
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

struct LargeBinaryArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::LargeBinaryBuilder>,
}

impl arrays::GuestLargeBinaryArrayBuilder for LargeBinaryArrayBuilderImpl {
    fn new() -> Self {
        Self { inner: RefCell::new(arrow_array::builder::LargeBinaryBuilder::new()) }
    }

    fn with_capacity(capacity: u64) -> arrays::LargeBinaryArrayBuilder {
        arrays::LargeBinaryArrayBuilder::new(Self {
            inner: RefCell::new(arrow_array::builder::LargeBinaryBuilder::with_capacity(capacity as usize, 0)),
        })
    }

    fn append_value(&self, value: Vec<u8>) {
        self.inner.borrow_mut().append_value(&value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

struct FixedSizeBinaryArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::FixedSizeBinaryBuilder>,
    byte_width: u32,
}

impl arrays::GuestFixedSizeBinaryArrayBuilder for FixedSizeBinaryArrayBuilderImpl {
    fn new(byte_width: u32) -> Self {
        Self {
            inner: RefCell::new(arrow_array::builder::FixedSizeBinaryBuilder::new(byte_width as i32)),
            byte_width,
        }
    }

    fn append_value(&self, value: Vec<u8>) -> Result<(), arrays::ArrowError> {
        if value.len() != self.byte_width as usize {
            return Err(arrays::ArrowError::InvalidArgument(
                format!("Expected {} bytes, got {}", self.byte_width, value.len())
            ));
        }
        self.inner.borrow_mut().append_value(&value)
            .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

struct ListArrayBuilderImpl {
    i64_builder: Option<RefCell<arrow_array::builder::ListBuilder<arrow_array::builder::Int64Builder>>>,
    f64_builder: Option<RefCell<arrow_array::builder::ListBuilder<arrow_array::builder::Float64Builder>>>,
    string_builder: Option<RefCell<arrow_array::builder::ListBuilder<arrow_array::builder::StringBuilder>>>,
    value_type: types::DataType,
}

impl arrays::GuestListArrayBuilder for ListArrayBuilderImpl {
    fn new(value_type: types::DataType) -> Self {
        match &value_type {
            types::DataType::Int64 => Self {
                i64_builder: Some(RefCell::new(arrow_array::builder::ListBuilder::new(arrow_array::builder::Int64Builder::new()))),
                f64_builder: None,
                string_builder: None,
                value_type,
            },
            types::DataType::Float64 => Self {
                i64_builder: None,
                f64_builder: Some(RefCell::new(arrow_array::builder::ListBuilder::new(arrow_array::builder::Float64Builder::new()))),
                string_builder: None,
                value_type,
            },
            types::DataType::Utf8 => Self {
                i64_builder: None,
                f64_builder: None,
                string_builder: Some(RefCell::new(arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new()))),
                value_type,
            },
            _ => Self {
                // Default to i64 for unsupported types
                i64_builder: Some(RefCell::new(arrow_array::builder::ListBuilder::new(arrow_array::builder::Int64Builder::new()))),
                f64_builder: None,
                string_builder: None,
                value_type,
            },
        }
    }

    fn append_null(&self) {
        if let Some(builder) = &self.i64_builder {
            builder.borrow_mut().append_null();
        } else if let Some(builder) = &self.f64_builder {
            builder.borrow_mut().append_null();
        } else if let Some(builder) = &self.string_builder {
            builder.borrow_mut().append_null();
        }
    }

    fn append_values_i64(&self, values: Vec<i64>) {
        if let Some(builder) = &self.i64_builder {
            let mut b = builder.borrow_mut();
            let values_builder = b.values();
            for v in values {
                values_builder.append_value(v);
            }
            b.append(true);
        }
    }

    fn append_values_f64(&self, values: Vec<f64>) {
        if let Some(builder) = &self.f64_builder {
            let mut b = builder.borrow_mut();
            let values_builder = b.values();
            for v in values {
                values_builder.append_value(v);
            }
            b.append(true);
        }
    }

    fn append_values_string(&self, values: Vec<String>) {
        if let Some(builder) = &self.string_builder {
            let mut b = builder.borrow_mut();
            let values_builder = b.values();
            for v in values {
                values_builder.append_value(&v);
            }
            b.append(true);
        }
    }

    fn len(&self) -> u64 {
        if let Some(builder) = &self.i64_builder {
            builder.borrow().len() as u64
        } else if let Some(builder) = &self.f64_builder {
            builder.borrow().len() as u64
        } else if let Some(builder) = &self.string_builder {
            builder.borrow().len() as u64
        } else {
            0
        }
    }

    fn finish(&self) -> arrays::Array {
        if let Some(builder) = &self.i64_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else if let Some(builder) = &self.f64_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else if let Some(builder) = &self.string_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else {
            // Shouldn't happen, return empty list
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(arrow_array::builder::ListBuilder::new(arrow_array::builder::Int64Builder::new()).finish()),
            })
        }
    }
}

// ============================================================================
// LargeList Array Builder
// ============================================================================

struct LargeListArrayBuilderImpl {
    i64_builder: Option<RefCell<arrow_array::builder::LargeListBuilder<arrow_array::builder::Int64Builder>>>,
    f64_builder: Option<RefCell<arrow_array::builder::LargeListBuilder<arrow_array::builder::Float64Builder>>>,
    string_builder: Option<RefCell<arrow_array::builder::LargeListBuilder<arrow_array::builder::StringBuilder>>>,
}

impl arrays::GuestLargeListArrayBuilder for LargeListArrayBuilderImpl {
    fn new(value_type: types::DataType) -> Self {
        match &value_type {
            types::DataType::Int64 => Self {
                i64_builder: Some(RefCell::new(arrow_array::builder::LargeListBuilder::new(arrow_array::builder::Int64Builder::new()))),
                f64_builder: None,
                string_builder: None,
            },
            types::DataType::Float64 => Self {
                i64_builder: None,
                f64_builder: Some(RefCell::new(arrow_array::builder::LargeListBuilder::new(arrow_array::builder::Float64Builder::new()))),
                string_builder: None,
            },
            types::DataType::Utf8 => Self {
                i64_builder: None,
                f64_builder: None,
                string_builder: Some(RefCell::new(arrow_array::builder::LargeListBuilder::new(arrow_array::builder::StringBuilder::new()))),
            },
            _ => Self {
                i64_builder: Some(RefCell::new(arrow_array::builder::LargeListBuilder::new(arrow_array::builder::Int64Builder::new()))),
                f64_builder: None,
                string_builder: None,
            },
        }
    }

    fn append_null(&self) {
        if let Some(builder) = &self.i64_builder {
            builder.borrow_mut().append_null();
        } else if let Some(builder) = &self.f64_builder {
            builder.borrow_mut().append_null();
        } else if let Some(builder) = &self.string_builder {
            builder.borrow_mut().append_null();
        }
    }

    fn append_values_i64(&self, values: Vec<i64>) {
        if let Some(builder) = &self.i64_builder {
            let mut b = builder.borrow_mut();
            let values_builder = b.values();
            for v in values {
                values_builder.append_value(v);
            }
            b.append(true);
        }
    }

    fn append_values_f64(&self, values: Vec<f64>) {
        if let Some(builder) = &self.f64_builder {
            let mut b = builder.borrow_mut();
            let values_builder = b.values();
            for v in values {
                values_builder.append_value(v);
            }
            b.append(true);
        }
    }

    fn append_values_string(&self, values: Vec<String>) {
        if let Some(builder) = &self.string_builder {
            let mut b = builder.borrow_mut();
            let values_builder = b.values();
            for v in values {
                values_builder.append_value(&v);
            }
            b.append(true);
        }
    }

    fn len(&self) -> u64 {
        if let Some(builder) = &self.i64_builder {
            builder.borrow().len() as u64
        } else if let Some(builder) = &self.f64_builder {
            builder.borrow().len() as u64
        } else if let Some(builder) = &self.string_builder {
            builder.borrow().len() as u64
        } else {
            0
        }
    }

    fn finish(&self) -> arrays::Array {
        if let Some(builder) = &self.i64_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else if let Some(builder) = &self.f64_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else if let Some(builder) = &self.string_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(arrow_array::builder::LargeListBuilder::new(arrow_array::builder::Int64Builder::new()).finish()),
            })
        }
    }
}

// ============================================================================
// Struct Array Builder
// ============================================================================

struct StructArrayBuilderImpl {
    fields: Vec<(String, types::DataType)>,
    // Store builders for each field, keyed by index
    i64_builders: RefCell<std::collections::HashMap<u32, arrow_array::builder::Int64Builder>>,
    f64_builders: RefCell<std::collections::HashMap<u32, arrow_array::builder::Float64Builder>>,
    string_builders: RefCell<std::collections::HashMap<u32, arrow_array::builder::StringBuilder>>,
    bool_builders: RefCell<std::collections::HashMap<u32, arrow_array::builder::BooleanBuilder>>,
    null_bitmap: RefCell<Vec<bool>>,
    row_count: RefCell<usize>,
}

impl arrays::GuestStructArrayBuilder for StructArrayBuilderImpl {
    fn new(fields: Vec<(String, types::DataType)>) -> Self {
        let mut i64_builders = std::collections::HashMap::new();
        let mut f64_builders = std::collections::HashMap::new();
        let mut string_builders = std::collections::HashMap::new();
        let mut bool_builders = std::collections::HashMap::new();

        for (idx, (_, dtype)) in fields.iter().enumerate() {
            match dtype {
                types::DataType::Int64 => { i64_builders.insert(idx as u32, arrow_array::builder::Int64Builder::new()); }
                types::DataType::Float64 => { f64_builders.insert(idx as u32, arrow_array::builder::Float64Builder::new()); }
                types::DataType::Utf8 => { string_builders.insert(idx as u32, arrow_array::builder::StringBuilder::new()); }
                types::DataType::Boolean => { bool_builders.insert(idx as u32, arrow_array::builder::BooleanBuilder::new()); }
                _ => { i64_builders.insert(idx as u32, arrow_array::builder::Int64Builder::new()); }
            }
        }

        Self {
            fields,
            i64_builders: RefCell::new(i64_builders),
            f64_builders: RefCell::new(f64_builders),
            string_builders: RefCell::new(string_builders),
            bool_builders: RefCell::new(bool_builders),
            null_bitmap: RefCell::new(Vec::new()),
            row_count: RefCell::new(0),
        }
    }

    fn append_field_i64(&self, field_index: u32, value: Option<i64>) -> Result<(), arrays::ArrowError> {
        let mut builders = self.i64_builders.borrow_mut();
        if let Some(builder) = builders.get_mut(&field_index) {
            match value {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
            Ok(())
        } else {
            Err(arrays::ArrowError::InvalidArgument(format!(
                "Field {} is not an Int64 field",
                field_index
            )))
        }
    }

    fn append_field_f64(&self, field_index: u32, value: Option<f64>) -> Result<(), arrays::ArrowError> {
        let mut builders = self.f64_builders.borrow_mut();
        if let Some(builder) = builders.get_mut(&field_index) {
            match value {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
            Ok(())
        } else {
            Err(arrays::ArrowError::InvalidArgument(format!(
                "Field {} is not a Float64 field",
                field_index
            )))
        }
    }

    fn append_field_string(&self, field_index: u32, value: Option<String>) -> Result<(), arrays::ArrowError> {
        let mut builders = self.string_builders.borrow_mut();
        if let Some(builder) = builders.get_mut(&field_index) {
            match value {
                Some(v) => builder.append_value(&v),
                None => builder.append_null(),
            }
            Ok(())
        } else {
            Err(arrays::ArrowError::InvalidArgument(format!(
                "Field {} is not a String field",
                field_index
            )))
        }
    }

    fn append_field_bool(&self, field_index: u32, value: Option<bool>) -> Result<(), arrays::ArrowError> {
        let mut builders = self.bool_builders.borrow_mut();
        if let Some(builder) = builders.get_mut(&field_index) {
            match value {
                Some(v) => builder.append_value(v),
                None => builder.append_null(),
            }
            Ok(())
        } else {
            Err(arrays::ArrowError::InvalidArgument(format!(
                "Field {} is not a Boolean field",
                field_index
            )))
        }
    }

    fn append_row(&self) -> Result<(), arrays::ArrowError> {
        self.null_bitmap.borrow_mut().push(true);
        *self.row_count.borrow_mut() += 1;
        Ok(())
    }

    fn append_null(&self) {
        // Append nulls to all field builders
        for (idx, (_, dtype)) in self.fields.iter().enumerate() {
            match dtype {
                types::DataType::Int64 => {
                    if let Some(builder) = self.i64_builders.borrow_mut().get_mut(&(idx as u32)) {
                        builder.append_null();
                    }
                }
                types::DataType::Float64 => {
                    if let Some(builder) = self.f64_builders.borrow_mut().get_mut(&(idx as u32)) {
                        builder.append_null();
                    }
                }
                types::DataType::Utf8 => {
                    if let Some(builder) = self.string_builders.borrow_mut().get_mut(&(idx as u32)) {
                        builder.append_null();
                    }
                }
                types::DataType::Boolean => {
                    if let Some(builder) = self.bool_builders.borrow_mut().get_mut(&(idx as u32)) {
                        builder.append_null();
                    }
                }
                _ => {}
            }
        }
        self.null_bitmap.borrow_mut().push(false);
        *self.row_count.borrow_mut() += 1;
    }

    fn len(&self) -> u64 {
        *self.row_count.borrow() as u64
    }

    fn finish(&self) -> arrays::Array {
        let mut field_arrays: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        let mut schema_fields: Vec<arrow_schema::FieldRef> = Vec::new();

        for (idx, (name, dtype)) in self.fields.iter().enumerate() {
            let arrow_dtype = convert::to_arrow_data_type(dtype);
            schema_fields.push(Arc::new(arrow_schema::Field::new(name.clone(), arrow_dtype.clone(), true)));

            match dtype {
                types::DataType::Int64 => {
                    if let Some(builder) = self.i64_builders.borrow_mut().get_mut(&(idx as u32)) {
                        field_arrays.push(Arc::new(builder.finish()));
                    }
                }
                types::DataType::Float64 => {
                    if let Some(builder) = self.f64_builders.borrow_mut().get_mut(&(idx as u32)) {
                        field_arrays.push(Arc::new(builder.finish()));
                    }
                }
                types::DataType::Utf8 => {
                    if let Some(builder) = self.string_builders.borrow_mut().get_mut(&(idx as u32)) {
                        field_arrays.push(Arc::new(builder.finish()));
                    }
                }
                types::DataType::Boolean => {
                    if let Some(builder) = self.bool_builders.borrow_mut().get_mut(&(idx as u32)) {
                        field_arrays.push(Arc::new(builder.finish()));
                    }
                }
                _ => {
                    // Default to empty Int64 for unsupported types
                    let empty_values: Vec<i64> = vec![];
                    let empty = arrow_array::Int64Array::from(empty_values);
                    field_arrays.push(Arc::new(empty));
                }
            }
        }

        // Create null buffer from bitmap
        let null_bitmap = self.null_bitmap.borrow();
        let null_buffer = if null_bitmap.iter().any(|&v| !v) {
            Some(arrow_buffer::NullBuffer::from(null_bitmap.clone()))
        } else {
            None
        };

        let struct_arr = arrow_array::StructArray::try_new(
            schema_fields.into(),
            field_arrays,
            null_buffer,
        ).unwrap_or_else(|_| {
            // Fallback to empty struct
            arrow_array::StructArray::new_empty_fields(0, None)
        });

        arrays::Array::new(ArrayImpl { inner: Arc::new(struct_arr) })
    }
}

// ============================================================================
// Map Array Builder
// ============================================================================

struct MapArrayBuilderImpl {
    // For simplicity, support string keys with different value types
    string_i64_builder: Option<RefCell<arrow_array::builder::MapBuilder<arrow_array::builder::StringBuilder, arrow_array::builder::Int64Builder>>>,
    string_f64_builder: Option<RefCell<arrow_array::builder::MapBuilder<arrow_array::builder::StringBuilder, arrow_array::builder::Float64Builder>>>,
    string_string_builder: Option<RefCell<arrow_array::builder::MapBuilder<arrow_array::builder::StringBuilder, arrow_array::builder::StringBuilder>>>,
}

impl arrays::GuestMapArrayBuilder for MapArrayBuilderImpl {
    fn new(key_type: types::DataType, value_type: types::DataType) -> Self {
        match (&key_type, &value_type) {
            (types::DataType::Utf8, types::DataType::Int64) => Self {
                string_i64_builder: Some(RefCell::new(arrow_array::builder::MapBuilder::new(
                    None,
                    arrow_array::builder::StringBuilder::new(),
                    arrow_array::builder::Int64Builder::new(),
                ))),
                string_f64_builder: None,
                string_string_builder: None,
            },
            (types::DataType::Utf8, types::DataType::Float64) => Self {
                string_i64_builder: None,
                string_f64_builder: Some(RefCell::new(arrow_array::builder::MapBuilder::new(
                    None,
                    arrow_array::builder::StringBuilder::new(),
                    arrow_array::builder::Float64Builder::new(),
                ))),
                string_string_builder: None,
            },
            (types::DataType::Utf8, types::DataType::Utf8) => Self {
                string_i64_builder: None,
                string_f64_builder: None,
                string_string_builder: Some(RefCell::new(arrow_array::builder::MapBuilder::new(
                    None,
                    arrow_array::builder::StringBuilder::new(),
                    arrow_array::builder::StringBuilder::new(),
                ))),
            },
            _ => Self {
                // Default to string -> i64
                string_i64_builder: Some(RefCell::new(arrow_array::builder::MapBuilder::new(
                    None,
                    arrow_array::builder::StringBuilder::new(),
                    arrow_array::builder::Int64Builder::new(),
                ))),
                string_f64_builder: None,
                string_string_builder: None,
            },
        }
    }

    fn start_map(&self) {
        // Maps don't have explicit start - entries are appended
    }

    fn append_entry_string_i64(&self, key: String, value: Option<i64>) {
        if let Some(builder) = &self.string_i64_builder {
            let mut b = builder.borrow_mut();
            b.keys().append_value(&key);
            match value {
                Some(v) => b.values().append_value(v),
                None => b.values().append_null(),
            }
        }
    }

    fn append_entry_string_f64(&self, key: String, value: Option<f64>) {
        if let Some(builder) = &self.string_f64_builder {
            let mut b = builder.borrow_mut();
            b.keys().append_value(&key);
            match value {
                Some(v) => b.values().append_value(v),
                None => b.values().append_null(),
            }
        }
    }

    fn append_entry_string_string(&self, key: String, value: Option<String>) {
        if let Some(builder) = &self.string_string_builder {
            let mut b = builder.borrow_mut();
            b.keys().append_value(&key);
            match value {
                Some(v) => b.values().append_value(&v),
                None => b.values().append_null(),
            }
        }
    }

    fn end_map(&self) {
        if let Some(builder) = &self.string_i64_builder {
            let _ = builder.borrow_mut().append(true);
        } else if let Some(builder) = &self.string_f64_builder {
            let _ = builder.borrow_mut().append(true);
        } else if let Some(builder) = &self.string_string_builder {
            let _ = builder.borrow_mut().append(true);
        }
    }

    fn append_null(&self) {
        if let Some(builder) = &self.string_i64_builder {
            let _ = builder.borrow_mut().append(false);
        } else if let Some(builder) = &self.string_f64_builder {
            let _ = builder.borrow_mut().append(false);
        } else if let Some(builder) = &self.string_string_builder {
            let _ = builder.borrow_mut().append(false);
        }
    }

    fn len(&self) -> u64 {
        if let Some(builder) = &self.string_i64_builder {
            builder.borrow().len() as u64
        } else if let Some(builder) = &self.string_f64_builder {
            builder.borrow().len() as u64
        } else if let Some(builder) = &self.string_string_builder {
            builder.borrow().len() as u64
        } else {
            0
        }
    }

    fn finish(&self) -> arrays::Array {
        if let Some(builder) = &self.string_i64_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else if let Some(builder) = &self.string_f64_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else if let Some(builder) = &self.string_string_builder {
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.borrow_mut().finish()),
            })
        } else {
            // Return empty map
            let mut builder: arrow_array::builder::MapBuilder<arrow_array::builder::StringBuilder, arrow_array::builder::Int64Builder> = arrow_array::builder::MapBuilder::new(
                None,
                arrow_array::builder::StringBuilder::new(),
                arrow_array::builder::Int64Builder::new(),
            );
            arrays::Array::new(ArrayImpl {
                inner: Arc::new(builder.finish()),
            })
        }
    }
}

// ============================================================================
// Temporal Array Builders
// ============================================================================

struct Date32ArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::Date32Builder>,
}

impl arrays::GuestDate32ArrayBuilder for Date32ArrayBuilderImpl {
    fn new() -> Self {
        Self { inner: RefCell::new(arrow_array::builder::Date32Builder::new()) }
    }

    fn with_capacity(capacity: u64) -> arrays::Date32ArrayBuilder {
        arrays::Date32ArrayBuilder::new(Self {
            inner: RefCell::new(arrow_array::builder::Date32Builder::with_capacity(capacity as usize)),
        })
    }

    fn append_value(&self, value: i32) {
        self.inner.borrow_mut().append_value(value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn append_values(&self, values: Vec<i32>, is_valid: Vec<bool>) {
        let mut builder = self.inner.borrow_mut();
        for (v, valid) in values.into_iter().zip(is_valid.into_iter()) {
            if valid {
                builder.append_value(v);
            } else {
                builder.append_null();
            }
        }
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

struct Date64ArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::Date64Builder>,
}

impl arrays::GuestDate64ArrayBuilder for Date64ArrayBuilderImpl {
    fn new() -> Self {
        Self { inner: RefCell::new(arrow_array::builder::Date64Builder::new()) }
    }

    fn with_capacity(capacity: u64) -> arrays::Date64ArrayBuilder {
        arrays::Date64ArrayBuilder::new(Self {
            inner: RefCell::new(arrow_array::builder::Date64Builder::with_capacity(capacity as usize)),
        })
    }

    fn append_value(&self, value: i64) {
        self.inner.borrow_mut().append_value(value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn append_values(&self, values: Vec<i64>, is_valid: Vec<bool>) {
        let mut builder = self.inner.borrow_mut();
        for (v, valid) in values.into_iter().zip(is_valid.into_iter()) {
            if valid {
                builder.append_value(v);
            } else {
                builder.append_null();
            }
        }
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

enum TimestampBuilderVariant {
    Second(arrow_array::builder::TimestampSecondBuilder),
    Millisecond(arrow_array::builder::TimestampMillisecondBuilder),
    Microsecond(arrow_array::builder::TimestampMicrosecondBuilder),
    Nanosecond(arrow_array::builder::TimestampNanosecondBuilder),
}

struct TimestampArrayBuilderImpl {
    inner: RefCell<TimestampBuilderVariant>,
}

impl arrays::GuestTimestampArrayBuilder for TimestampArrayBuilderImpl {
    fn new(unit: types::TimeUnit, timezone: Option<String>) -> Self {
        let tz: Option<Arc<str>> = timezone.map(|s| Arc::from(s.as_str()));
        let builder = match unit {
            types::TimeUnit::Second => TimestampBuilderVariant::Second(
                arrow_array::builder::TimestampSecondBuilder::new().with_timezone_opt(tz)
            ),
            types::TimeUnit::Millisecond => TimestampBuilderVariant::Millisecond(
                arrow_array::builder::TimestampMillisecondBuilder::new().with_timezone_opt(tz)
            ),
            types::TimeUnit::Microsecond => TimestampBuilderVariant::Microsecond(
                arrow_array::builder::TimestampMicrosecondBuilder::new().with_timezone_opt(tz)
            ),
            types::TimeUnit::Nanosecond => TimestampBuilderVariant::Nanosecond(
                arrow_array::builder::TimestampNanosecondBuilder::new().with_timezone_opt(tz)
            ),
        };
        Self { inner: RefCell::new(builder) }
    }

    fn append_value(&self, value: i64) {
        match &mut *self.inner.borrow_mut() {
            TimestampBuilderVariant::Second(b) => b.append_value(value),
            TimestampBuilderVariant::Millisecond(b) => b.append_value(value),
            TimestampBuilderVariant::Microsecond(b) => b.append_value(value),
            TimestampBuilderVariant::Nanosecond(b) => b.append_value(value),
        }
    }

    fn append_null(&self) {
        match &mut *self.inner.borrow_mut() {
            TimestampBuilderVariant::Second(b) => b.append_null(),
            TimestampBuilderVariant::Millisecond(b) => b.append_null(),
            TimestampBuilderVariant::Microsecond(b) => b.append_null(),
            TimestampBuilderVariant::Nanosecond(b) => b.append_null(),
        }
    }

    fn len(&self) -> u64 {
        match &*self.inner.borrow() {
            TimestampBuilderVariant::Second(b) => b.len() as u64,
            TimestampBuilderVariant::Millisecond(b) => b.len() as u64,
            TimestampBuilderVariant::Microsecond(b) => b.len() as u64,
            TimestampBuilderVariant::Nanosecond(b) => b.len() as u64,
        }
    }

    fn finish(&self) -> arrays::Array {
        let arr: Arc<dyn arrow_array::Array> = match &mut *self.inner.borrow_mut() {
            TimestampBuilderVariant::Second(b) => Arc::new(b.finish()),
            TimestampBuilderVariant::Millisecond(b) => Arc::new(b.finish()),
            TimestampBuilderVariant::Microsecond(b) => Arc::new(b.finish()),
            TimestampBuilderVariant::Nanosecond(b) => Arc::new(b.finish()),
        };
        arrays::Array::new(ArrayImpl { inner: arr })
    }
}

enum DurationBuilderVariant {
    Second(arrow_array::builder::DurationSecondBuilder),
    Millisecond(arrow_array::builder::DurationMillisecondBuilder),
    Microsecond(arrow_array::builder::DurationMicrosecondBuilder),
    Nanosecond(arrow_array::builder::DurationNanosecondBuilder),
}

struct DurationArrayBuilderImpl {
    inner: RefCell<DurationBuilderVariant>,
}

impl arrays::GuestDurationArrayBuilder for DurationArrayBuilderImpl {
    fn new(unit: types::TimeUnit) -> Self {
        let builder = match unit {
            types::TimeUnit::Second => DurationBuilderVariant::Second(
                arrow_array::builder::DurationSecondBuilder::new()
            ),
            types::TimeUnit::Millisecond => DurationBuilderVariant::Millisecond(
                arrow_array::builder::DurationMillisecondBuilder::new()
            ),
            types::TimeUnit::Microsecond => DurationBuilderVariant::Microsecond(
                arrow_array::builder::DurationMicrosecondBuilder::new()
            ),
            types::TimeUnit::Nanosecond => DurationBuilderVariant::Nanosecond(
                arrow_array::builder::DurationNanosecondBuilder::new()
            ),
        };
        Self { inner: RefCell::new(builder) }
    }

    fn append_value(&self, value: i64) {
        match &mut *self.inner.borrow_mut() {
            DurationBuilderVariant::Second(b) => b.append_value(value),
            DurationBuilderVariant::Millisecond(b) => b.append_value(value),
            DurationBuilderVariant::Microsecond(b) => b.append_value(value),
            DurationBuilderVariant::Nanosecond(b) => b.append_value(value),
        }
    }

    fn append_null(&self) {
        match &mut *self.inner.borrow_mut() {
            DurationBuilderVariant::Second(b) => b.append_null(),
            DurationBuilderVariant::Millisecond(b) => b.append_null(),
            DurationBuilderVariant::Microsecond(b) => b.append_null(),
            DurationBuilderVariant::Nanosecond(b) => b.append_null(),
        }
    }

    fn len(&self) -> u64 {
        match &*self.inner.borrow() {
            DurationBuilderVariant::Second(b) => b.len() as u64,
            DurationBuilderVariant::Millisecond(b) => b.len() as u64,
            DurationBuilderVariant::Microsecond(b) => b.len() as u64,
            DurationBuilderVariant::Nanosecond(b) => b.len() as u64,
        }
    }

    fn finish(&self) -> arrays::Array {
        let arr: Arc<dyn arrow_array::Array> = match &mut *self.inner.borrow_mut() {
            DurationBuilderVariant::Second(b) => Arc::new(b.finish()),
            DurationBuilderVariant::Millisecond(b) => Arc::new(b.finish()),
            DurationBuilderVariant::Microsecond(b) => Arc::new(b.finish()),
            DurationBuilderVariant::Nanosecond(b) => Arc::new(b.finish()),
        };
        arrays::Array::new(ArrayImpl { inner: arr })
    }
}

enum Time32BuilderVariant {
    Second(arrow_array::builder::Time32SecondBuilder),
    Millisecond(arrow_array::builder::Time32MillisecondBuilder),
}

struct Time32ArrayBuilderImpl {
    inner: RefCell<Time32BuilderVariant>,
}

impl arrays::GuestTime32ArrayBuilder for Time32ArrayBuilderImpl {
    fn new(unit: types::TimeUnit) -> Self {
        let builder = match unit {
            types::TimeUnit::Second => Time32BuilderVariant::Second(
                arrow_array::builder::Time32SecondBuilder::new()
            ),
            types::TimeUnit::Millisecond => Time32BuilderVariant::Millisecond(
                arrow_array::builder::Time32MillisecondBuilder::new()
            ),
            // Time32 only supports second and millisecond, default to second for other units
            _ => Time32BuilderVariant::Second(
                arrow_array::builder::Time32SecondBuilder::new()
            ),
        };
        Self { inner: RefCell::new(builder) }
    }

    fn append_value(&self, value: i32) {
        match &mut *self.inner.borrow_mut() {
            Time32BuilderVariant::Second(b) => b.append_value(value),
            Time32BuilderVariant::Millisecond(b) => b.append_value(value),
        }
    }

    fn append_null(&self) {
        match &mut *self.inner.borrow_mut() {
            Time32BuilderVariant::Second(b) => b.append_null(),
            Time32BuilderVariant::Millisecond(b) => b.append_null(),
        }
    }

    fn len(&self) -> u64 {
        match &*self.inner.borrow() {
            Time32BuilderVariant::Second(b) => b.len() as u64,
            Time32BuilderVariant::Millisecond(b) => b.len() as u64,
        }
    }

    fn finish(&self) -> arrays::Array {
        let arr: Arc<dyn arrow_array::Array> = match &mut *self.inner.borrow_mut() {
            Time32BuilderVariant::Second(b) => Arc::new(b.finish()),
            Time32BuilderVariant::Millisecond(b) => Arc::new(b.finish()),
        };
        arrays::Array::new(ArrayImpl { inner: arr })
    }
}

enum Time64BuilderVariant {
    Microsecond(arrow_array::builder::Time64MicrosecondBuilder),
    Nanosecond(arrow_array::builder::Time64NanosecondBuilder),
}

struct Time64ArrayBuilderImpl {
    inner: RefCell<Time64BuilderVariant>,
}

impl arrays::GuestTime64ArrayBuilder for Time64ArrayBuilderImpl {
    fn new(unit: types::TimeUnit) -> Self {
        let builder = match unit {
            types::TimeUnit::Microsecond => Time64BuilderVariant::Microsecond(
                arrow_array::builder::Time64MicrosecondBuilder::new()
            ),
            types::TimeUnit::Nanosecond => Time64BuilderVariant::Nanosecond(
                arrow_array::builder::Time64NanosecondBuilder::new()
            ),
            // Time64 only supports microsecond and nanosecond, default to microsecond for other units
            _ => Time64BuilderVariant::Microsecond(
                arrow_array::builder::Time64MicrosecondBuilder::new()
            ),
        };
        Self { inner: RefCell::new(builder) }
    }

    fn append_value(&self, value: i64) {
        match &mut *self.inner.borrow_mut() {
            Time64BuilderVariant::Microsecond(b) => b.append_value(value),
            Time64BuilderVariant::Nanosecond(b) => b.append_value(value),
        }
    }

    fn append_null(&self) {
        match &mut *self.inner.borrow_mut() {
            Time64BuilderVariant::Microsecond(b) => b.append_null(),
            Time64BuilderVariant::Nanosecond(b) => b.append_null(),
        }
    }

    fn len(&self) -> u64 {
        match &*self.inner.borrow() {
            Time64BuilderVariant::Microsecond(b) => b.len() as u64,
            Time64BuilderVariant::Nanosecond(b) => b.len() as u64,
        }
    }

    fn finish(&self) -> arrays::Array {
        let arr: Arc<dyn arrow_array::Array> = match &mut *self.inner.borrow_mut() {
            Time64BuilderVariant::Microsecond(b) => Arc::new(b.finish()),
            Time64BuilderVariant::Nanosecond(b) => Arc::new(b.finish()),
        };
        arrays::Array::new(ArrayImpl { inner: arr })
    }
}

// ============================================================================
// Decimal Array Builders
// ============================================================================

struct Decimal128ArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::Decimal128Builder>,
    precision: u8,
    scale: i8,
}

impl arrays::GuestDecimal128ArrayBuilder for Decimal128ArrayBuilderImpl {
    fn new(precision: u8, scale: i8) -> Self {
        Self {
            inner: RefCell::new(
                arrow_array::builder::Decimal128Builder::new()
                    .with_precision_and_scale(precision, scale)
                    .unwrap_or_else(|_| arrow_array::builder::Decimal128Builder::new())
            ),
            precision,
            scale,
        }
    }

    fn append_value_string(&self, value: String) -> Result<(), arrays::ArrowError> {
        // Parse string to i128 considering scale
        let parsed = parse_decimal_string(&value, self.scale)
            .map_err(|e| arrays::ArrowError::InvalidArgument(e))?;
        self.inner.borrow_mut().append_value(parsed);
        Ok(())
    }

    fn append_value_i128(&self, high: i64, low: u64) {
        // Reconstruct i128 from high and low parts
        let value = ((high as i128) << 64) | (low as i128);
        self.inner.borrow_mut().append_value(value);
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

struct Decimal256ArrayBuilderImpl {
    inner: RefCell<arrow_array::builder::Decimal256Builder>,
}

impl arrays::GuestDecimal256ArrayBuilder for Decimal256ArrayBuilderImpl {
    fn new(precision: u8, scale: i8) -> Self {
        Self {
            inner: RefCell::new(
                arrow_array::builder::Decimal256Builder::new()
                    .with_precision_and_scale(precision, scale)
                    .unwrap_or_else(|_| arrow_array::builder::Decimal256Builder::new())
            ),
        }
    }

    fn append_value_bytes(&self, value: Vec<u8>) -> Result<(), arrays::ArrowError> {
        if value.len() != 32 {
            return Err(arrays::ArrowError::InvalidArgument(
                format!("Decimal256 requires exactly 32 bytes, got {}", value.len())
            ));
        }
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&value);
        let i256 = arrow_buffer::i256::from_le_bytes(bytes);
        self.inner.borrow_mut().append_value(i256);
        Ok(())
    }

    fn append_null(&self) {
        self.inner.borrow_mut().append_null();
    }

    fn len(&self) -> u64 {
        self.inner.borrow().len() as u64
    }

    fn finish(&self) -> arrays::Array {
        arrays::Array::new(ArrayImpl {
            inner: Arc::new(self.inner.borrow_mut().finish()),
        })
    }
}

/// Parse a decimal string like "123.45" or "-999.99" into i128 with given scale
fn parse_decimal_string(s: &str, scale: i8) -> Result<i128, String> {
    let s = s.trim();
    let negative = s.starts_with('-');
    let s = if negative { &s[1..] } else { s };

    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() > 2 {
        return Err("Invalid decimal string: multiple decimal points".to_string());
    }

    let integer_part = parts[0];
    let fractional_part = if parts.len() == 2 { parts[1] } else { "" };

    // Parse integer part
    let int_val: i128 = if integer_part.is_empty() {
        0
    } else {
        integer_part.parse().map_err(|e| format!("Invalid integer part: {}", e))?
    };

    // Parse fractional part, padded or truncated to scale
    let scale_usize = scale.max(0) as usize;
    let frac_val: i128 = if fractional_part.is_empty() {
        0
    } else {
        let padded = if fractional_part.len() < scale_usize {
            format!("{:0<width$}", fractional_part, width = scale_usize)
        } else {
            fractional_part[..scale_usize].to_string()
        };
        padded.parse().map_err(|e| format!("Invalid fractional part: {}", e))?
    };

    // Combine: integer * 10^scale + fractional
    let multiplier: i128 = 10_i128.pow(scale_usize as u32);
    let result = int_val * multiplier + frac_val;

    Ok(if negative { -result } else { result })
}

/// Helper function to aggregate values at given row indices
fn aggregate_values(arr: &dyn arrow_array::Array, rows: &[usize], func: &compute::AggFunction) -> Option<f64> {
    use arrow_array::Array as ArrowArrayTrait;
    use compute::AggFunction;

    // Extract numeric values from the array at the specified rows
    let mut values: Vec<f64> = Vec::new();

    if let Some(float_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        for &row in rows {
            if !float_arr.is_null(row) {
                values.push(float_arr.value(row));
            }
        }
    } else if let Some(float_arr) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        for &row in rows {
            if !float_arr.is_null(row) {
                values.push(float_arr.value(row) as f64);
            }
        }
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        for &row in rows {
            if !int_arr.is_null(row) {
                values.push(int_arr.value(row) as f64);
            }
        }
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        for &row in rows {
            if !int_arr.is_null(row) {
                values.push(int_arr.value(row) as f64);
            }
        }
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        for &row in rows {
            if !int_arr.is_null(row) {
                values.push(int_arr.value(row) as f64);
            }
        }
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        for &row in rows {
            if !int_arr.is_null(row) {
                values.push(int_arr.value(row) as f64);
            }
        }
    } else if let Some(uint_arr) = arr.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        for &row in rows {
            if !uint_arr.is_null(row) {
                values.push(uint_arr.value(row) as f64);
            }
        }
    } else if let Some(uint_arr) = arr.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        for &row in rows {
            if !uint_arr.is_null(row) {
                values.push(uint_arr.value(row) as f64);
            }
        }
    }

    if values.is_empty() && !matches!(func, AggFunction::Count) {
        return None;
    }

    match func {
        AggFunction::Sum => Some(values.iter().sum()),
        AggFunction::Min => values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
        AggFunction::Max => values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
        AggFunction::Count => Some(rows.len() as f64),
        AggFunction::Mean => {
            if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
        }
        AggFunction::First => values.first().cloned(),
        AggFunction::Last => values.last().cloned(),
    }
}

/// Helper function to build a join key from a row
fn build_join_key(batch: &ArrowRecordBatch, row: usize, col_indices: &[usize]) -> Vec<String> {
    col_indices.iter().map(|&idx| {
        let col = batch.column(idx);
        if col.is_null(row) {
            "NULL".to_string()
        } else if let Some(str_arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
            str_arr.value(row).to_string()
        } else if let Some(int_arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
            int_arr.value(row).to_string()
        } else if let Some(int_arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
            int_arr.value(row).to_string()
        } else if let Some(float_arr) = col.as_any().downcast_ref::<arrow_array::Float64Array>() {
            format!("{:.10}", float_arr.value(row))
        } else {
            format!("row_{}", row)
        }
    }).collect()
}

/// Helper function to build the result of a join operation
fn build_join_result(
    left: &ArrowRecordBatch,
    right: &ArrowRecordBatch,
    left_indices: &[Option<u64>],
    right_indices: &[Option<u64>],
    join_type: &compute::JoinType,
) -> Result<record_batch::RecordBatch, compute::ArrowError> {
    use compute::JoinType;

    let mut result_columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
    let mut result_fields: Vec<arrow_schema::Field> = Vec::new();

    // Add left columns
    let include_left = !matches!(join_type, JoinType::LeftSemi | JoinType::LeftAnti) || true;
    if include_left {
        for (i, field) in left.schema().fields().iter().enumerate() {
            let col = left.column(i);
            let taken = take_with_nulls(col.as_ref(), left_indices)?;
            result_columns.push(taken);
            result_fields.push(field.as_ref().clone());
        }
    }

    // Add right columns (not for semi/anti joins)
    if !matches!(join_type, JoinType::LeftSemi | JoinType::LeftAnti) {
        for (i, field) in right.schema().fields().iter().enumerate() {
            let col = right.column(i);
            let taken = take_with_nulls(col.as_ref(), right_indices)?;
            result_columns.push(taken);
            // Rename right columns to avoid collision
            let new_name = format!("{}_right", field.name());
            result_fields.push(arrow_schema::Field::new(&new_name, field.data_type().clone(), true));
        }
    }

    let schema = Arc::new(arrow_schema::Schema::new(result_fields));
    let result = ArrowRecordBatch::try_new(schema, result_columns)
        .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

    Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
}

/// Helper function to take values with nullable indices
fn take_with_nulls(arr: &dyn arrow_array::Array, indices: &[Option<u64>]) -> Result<Arc<dyn arrow_array::Array>, compute::ArrowError> {
    // Build a UInt64Array with nulls where indices are None
    let indices_arr: arrow_array::UInt64Array = indices.iter()
        .map(|&opt| opt)
        .collect();

    arrow_select::take::take(arr, &indices_arr, None)
        .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))
}

/// Helper function to extract Float64 values from various numeric array types
fn extract_float64_values(arr: &arrays::ArrayBorrow<'_>) -> Result<Vec<f64>, compute::ArrowError> {
    let arr_impl = arr.get::<ArrayImpl>();

    // Int64
    if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Ok(int_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // Int32
    if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
        return Ok(int_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // Int16
    if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Ok(int_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // Int8
    if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Ok(int_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // UInt64
    if let Some(uint_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        return Ok(uint_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // UInt32
    if let Some(uint_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        return Ok(uint_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // UInt16
    if let Some(uint_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UInt16Array>() {
        return Ok(uint_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // UInt8
    if let Some(uint_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UInt8Array>() {
        return Ok(uint_arr.iter().flatten().map(|v| v as f64).collect());
    }

    // Float64
    if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
        return Ok(float_arr.iter().flatten().collect());
    }

    // Float32
    if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
        return Ok(float_arr.iter().flatten().map(|v| v as f64).collect());
    }

    Err(compute::ArrowError::NotImplemented("extract_float64_values only supports numeric arrays".to_string()))
}

/// Helper function to interpolate between sorted values for quantile calculation
fn interpolate_quantile(sorted: &[f64], idx: f64) -> f64 {
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower == upper || upper >= sorted.len() {
        sorted[lower.min(sorted.len() - 1)]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Helper function for rolling window aggregations
fn rolling_agg<F>(arr: &arrays::ArrayBorrow<'_>, options: &compute::RollingOptions, agg_fn: F) -> Result<arrays::Array, compute::ArrowError>
where
    F: Fn(&[f64]) -> f64,
{
    let values = extract_float64_values(arr)?;
    let window_size = options.window_size as usize;
    let min_periods = options.min_periods.map(|p| p as usize).unwrap_or(window_size);
    let center = options.center;
    let n = values.len();

    if window_size == 0 {
        return Err(compute::ArrowError::InvalidArgument("Window size must be greater than 0".to_string()));
    }

    let mut result: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        // Calculate window bounds
        let (start, end) = if center {
            let half = (window_size - 1) / 2;
            let start = if i >= half { i - half } else { 0 };
            let end = (i + (window_size - half)).min(n);
            (start, end)
        } else {
            // Trailing window
            let start = if i + 1 >= window_size { i + 1 - window_size } else { 0 };
            let end = i + 1;
            (start, end)
        };

        let window: Vec<f64> = values[start..end].to_vec();

        if window.len() >= min_periods {
            result.push(Some(agg_fn(&window)));
        } else {
            result.push(None);
        }
    }

    let result_arr: arrow_array::Float64Array = result.into_iter().collect();
    Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
}

// ============================================================================
// String Distance Algorithm Helpers
// ============================================================================

/// Compute Levenshtein edit distance between two strings
fn compute_levenshtein(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    // Use two-row optimization for space efficiency
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)               // deletion
                .min(curr[j - 1] + 1)             // insertion
                .min(prev[j - 1] + cost);         // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Compute Jaro similarity score (0.0 to 1.0) between two strings
fn compute_jaro(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 { return 1.0; }
    if len1 == 0 || len2 == 0 { return 0.0; }

    // Match window
    let match_distance = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];

    let mut matches = 0;
    let mut transpositions = 0;

    // Find matches
    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(len2);

        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 { return 0.0; }

    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !s1_matches[i] { continue; }
        while !s2_matches[k] { k += 1; }
        if s1_chars[i] != s2_chars[k] { transpositions += 1; }
        k += 1;
    }

    let matches = matches as f64;
    ((matches / len1 as f64) +
     (matches / len2 as f64) +
     ((matches - transpositions as f64 / 2.0) / matches)) / 3.0
}

/// Compute Jaro-Winkler similarity score (0.0 to 1.0) between two strings
fn compute_jaro_winkler(s1: &str, s2: &str) -> f64 {
    let jaro = compute_jaro(s1, s2);

    // Calculate common prefix (up to 4 characters)
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let prefix_len = s1_chars.iter()
        .zip(s2_chars.iter())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    // Winkler modification: boost score for common prefix
    // Scaling factor is 0.1 (standard Winkler)
    jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro))
}

/// Compute Soundex phonetic code for a string
fn compute_soundex(s: &str) -> String {
    let chars: Vec<char> = s.to_uppercase().chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if chars.is_empty() {
        return "0000".to_string();
    }

    let mut result = String::with_capacity(4);
    result.push(chars[0]);

    let soundex_code = |c: char| -> Option<char> {
        match c {
            'B' | 'F' | 'P' | 'V' => Some('1'),
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
            'D' | 'T' => Some('3'),
            'L' => Some('4'),
            'M' | 'N' => Some('5'),
            'R' => Some('6'),
            _ => None, // A, E, I, O, U, H, W, Y are not coded
        }
    };

    let mut last_code = soundex_code(chars[0]);

    for &c in &chars[1..] {
        if result.len() >= 4 { break; }

        let code = soundex_code(c);
        if code.is_some() && code != last_code {
            result.push(code.unwrap());
        }
        last_code = code;
    }

    // Pad with zeros to length 4
    while result.len() < 4 {
        result.push('0');
    }

    result
}

/// Compute Damerau-Levenshtein distance (allows transpositions)
fn compute_damerau_levenshtein(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    // Full matrix for Damerau-Levenshtein (needed for transpositions)
    let mut d = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m { d[i][0] = i; }
    for j in 0..=n { d[0][j] = j; }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };

            d[i][j] = (d[i - 1][j] + 1)          // deletion
                .min(d[i][j - 1] + 1)            // insertion
                .min(d[i - 1][j - 1] + cost);    // substitution

            // Transposition
            if i > 1 && j > 1
                && s1_chars[i - 1] == s2_chars[j - 2]
                && s1_chars[i - 2] == s2_chars[j - 1]
            {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + cost);
            }
        }
    }

    d[m][n]
}

/// Compute Longest Common Subsequence length
fn compute_lcs_length(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 || n == 0 { return 0; }

    // Use two-row optimization
    let mut prev = vec![0; n + 1];
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if s1_chars[i - 1] == s2_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[n]
}

// ============================================================================
// RecordBatch implementation
// ============================================================================

impl record_batch::Guest for Component {
    type RecordBatch = RecordBatchImpl;
    type RecordBatchBuilder = RecordBatchBuilderImpl;

    fn concat_batches(
        schema: types::Schema,
        batches: Vec<record_batch::RecordBatch>,
    ) -> Result<record_batch::RecordBatch, record_batch::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();
        let arrow_batches: Vec<ArrowRecordBatch> = batches
            .iter()
            .map(|b| b.get::<RecordBatchImpl>().inner.clone())
            .collect();
        let refs: Vec<&ArrowRecordBatch> = arrow_batches.iter().collect();
        let result = arrow_select::concat::concat_batches(&schema_impl.inner, refs)
            .map_err(|e| record_batch::ArrowError::InvalidArgument(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn validate_batch(batch: record_batch::RecordBatchBorrow<'_>) -> Vec<record_batch::ValidationError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        validate_batch_internal(&batch_impl.inner, &batch_impl.inner.schema())
    }

    fn validate_batch_schema(batch: record_batch::RecordBatchBorrow<'_>, expected_schema: types::SchemaBorrow<'_>) -> Vec<record_batch::ValidationError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let schema_impl = expected_schema.get::<SchemaImpl>();
        validate_batch_internal(&batch_impl.inner, &schema_impl.inner)
    }

    fn record_batch_with_schema(batch: record_batch::RecordBatchBorrow<'_>, new_schema: types::Schema) -> Result<record_batch::RecordBatch, types::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let schema_impl = new_schema.get::<SchemaImpl>();

        // Verify column count matches
        if batch_impl.inner.num_columns() != schema_impl.inner.fields().len() {
            return Err(types::ArrowError::SchemaMismatch(format!(
                "Column count mismatch: batch has {} columns, schema has {} fields",
                batch_impl.inner.num_columns(),
                schema_impl.inner.fields().len()
            )));
        }

        // Create new batch with the new schema
        let columns: Vec<Arc<dyn arrow_array::Array>> = batch_impl.inner.columns().to_vec();

        let result = ArrowRecordBatch::try_new(schema_impl.inner.clone(), columns)
            .map_err(|e| types::ArrowError::SchemaMismatch(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn record_batch_rename_columns(batch: record_batch::RecordBatchBorrow<'_>, new_names: Vec<String>) -> Result<record_batch::RecordBatch, types::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();

        if new_names.len() != batch_impl.inner.num_columns() {
            return Err(types::ArrowError::InvalidArgument(format!(
                "Number of names ({}) must match number of columns ({})",
                new_names.len(),
                batch_impl.inner.num_columns()
            )));
        }

        // Build new schema with renamed fields
        let old_schema = batch_impl.inner.schema();
        let new_fields: Vec<Arc<arrow_schema::Field>> = old_schema.fields()
            .iter()
            .zip(new_names.iter())
            .map(|(field, new_name)| {
                Arc::new(arrow_schema::Field::new(
                    new_name.clone(),
                    field.data_type().clone(),
                    field.is_nullable(),
                ).with_metadata(field.metadata().clone()))
            })
            .collect();

        let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            arrow_schema::Fields::from(new_fields),
            old_schema.metadata().clone(),
        ));

        // Create new batch with the new schema
        let columns: Vec<Arc<dyn arrow_array::Array>> = batch_impl.inner.columns().to_vec();

        let result = ArrowRecordBatch::try_new(new_schema, columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }
}

fn validate_batch_internal(batch: &ArrowRecordBatch, expected_schema: &Arc<arrow_schema::Schema>) -> Vec<record_batch::ValidationError> {
    let mut errors = Vec::new();
    let batch_schema = batch.schema();

    // Check number of columns matches
    if batch.num_columns() != expected_schema.fields().len() {
        errors.push(record_batch::ValidationError {
            column_index: 0,
            column_name: "".to_string(),
            row_index: None,
            error_type: "column_count_mismatch".to_string(),
            message: format!(
                "Expected {} columns, got {}",
                expected_schema.fields().len(),
                batch.num_columns()
            ),
        });
        return errors;
    }

    // Check each column
    for (col_idx, expected_field) in expected_schema.fields().iter().enumerate() {
        let column = batch.column(col_idx);
        let actual_field = batch_schema.field(col_idx);

        // Check column name
        if actual_field.name() != expected_field.name() {
            errors.push(record_batch::ValidationError {
                column_index: col_idx as u32,
                column_name: expected_field.name().to_string(),
                row_index: None,
                error_type: "column_name_mismatch".to_string(),
                message: format!(
                    "Expected column name '{}', got '{}'",
                    expected_field.name(),
                    actual_field.name()
                ),
            });
        }

        // Check data type
        if actual_field.data_type() != expected_field.data_type() {
            errors.push(record_batch::ValidationError {
                column_index: col_idx as u32,
                column_name: expected_field.name().to_string(),
                row_index: None,
                error_type: "type_mismatch".to_string(),
                message: format!(
                    "Expected type {:?}, got {:?}",
                    expected_field.data_type(),
                    actual_field.data_type()
                ),
            });
        }

        // Check nullability constraint
        if !expected_field.is_nullable() && column.null_count() > 0 {
            errors.push(record_batch::ValidationError {
                column_index: col_idx as u32,
                column_name: expected_field.name().to_string(),
                row_index: None,
                error_type: "unexpected_nulls".to_string(),
                message: format!(
                    "Column '{}' is not nullable but contains {} null values",
                    expected_field.name(),
                    column.null_count()
                ),
            });
        }

        // Check array length matches batch row count
        if column.len() != batch.num_rows() {
            errors.push(record_batch::ValidationError {
                column_index: col_idx as u32,
                column_name: expected_field.name().to_string(),
                row_index: None,
                error_type: "length_mismatch".to_string(),
                message: format!(
                    "Column '{}' has {} rows, but batch has {} rows",
                    expected_field.name(),
                    column.len(),
                    batch.num_rows()
                ),
            });
        }
    }

    errors
}

struct RecordBatchImpl {
    inner: ArrowRecordBatch,
}

impl record_batch::GuestRecordBatch for RecordBatchImpl {
    fn new(schema: types::Schema, columns: Vec<arrays::Array>) -> Self {
        let schema_impl = schema.get::<SchemaImpl>();
        let arrow_columns: Vec<ArrayRef> = columns
            .into_iter()
            .map(|a| a.get::<ArrayImpl>().inner.clone())
            .collect();
        Self {
            inner: ArrowRecordBatch::try_new(schema_impl.inner.clone(), arrow_columns)
                .expect("Failed to create RecordBatch"),
        }
    }

    fn try_new(
        schema: types::Schema,
        columns: Vec<arrays::Array>,
    ) -> Result<record_batch::RecordBatch, record_batch::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();
        let arrow_columns: Vec<ArrayRef> = columns
            .into_iter()
            .map(|a| a.get::<ArrayImpl>().inner.clone())
            .collect();
        let batch = ArrowRecordBatch::try_new(schema_impl.inner.clone(), arrow_columns)
            .map_err(|e| record_batch::ArrowError::InvalidArgument(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }))
    }

    fn schema(&self) -> types::Schema {
        types::Schema::new(SchemaImpl { inner: self.inner.schema() })
    }

    fn column(&self, index: u32) -> Option<arrays::Array> {
        if (index as usize) < self.inner.num_columns() {
            Some(arrays::Array::new(ArrayImpl {
                inner: self.inner.column(index as usize).clone(),
            }))
        } else {
            None
        }
    }

    fn column_by_name(&self, name: String) -> Option<arrays::Array> {
        self.inner.column_by_name(&name).map(|c| {
            arrays::Array::new(ArrayImpl { inner: c.clone() })
        })
    }

    fn num_columns(&self) -> u32 {
        self.inner.num_columns() as u32
    }

    fn num_rows(&self) -> u64 {
        self.inner.num_rows() as u64
    }

    fn slice(&self, offset: u64, length: u64) -> record_batch::RecordBatch {
        record_batch::RecordBatch::new(RecordBatchImpl {
            inner: self.inner.slice(offset as usize, length as usize),
        })
    }

    fn project(&self, indices: Vec<u32>) -> Result<record_batch::RecordBatch, record_batch::ArrowError> {
        let usize_indices: Vec<usize> = indices.iter().map(|i| *i as usize).collect();
        let projected = self.inner.project(&usize_indices)
            .map_err(|e| record_batch::ArrowError::InvalidArgument(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: projected }))
    }

    fn columns(&self) -> Vec<arrays::Array> {
        self.inner.columns().iter()
            .map(|c| arrays::Array::new(ArrayImpl { inner: c.clone() }))
            .collect()
    }

    fn column_names(&self) -> Vec<String> {
        self.inner.schema().fields().iter()
            .map(|f| f.name().clone())
            .collect()
    }

    fn remove_column(&self, index: u32) -> Result<record_batch::RecordBatch, record_batch::ArrowError> {
        // Build a new record batch without the specified column
        let schema = self.inner.schema();
        let num_cols = self.inner.num_columns();
        if index as usize >= num_cols {
            return Err(record_batch::ArrowError::OutOfBounds(format!(
                "Column index {} out of bounds for {} columns",
                index, num_cols
            )));
        }
        let new_fields: Vec<_> = schema.fields().iter()
            .enumerate()
            .filter(|(i, _)| *i != index as usize)
            .map(|(_, f)| f.clone())
            .collect();
        let new_columns: Vec<_> = self.inner.columns().iter()
            .enumerate()
            .filter(|(i, _)| *i != index as usize)
            .map(|(_, c)| c.clone())
            .collect();
        let new_schema = Arc::new(arrow_schema::Schema::new(new_fields));
        let batch = ArrowRecordBatch::try_new(new_schema, new_columns)
            .map_err(|e| record_batch::ArrowError::InvalidArgument(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }))
    }

    fn get_array_memory_size(&self) -> u64 {
        self.inner.get_array_memory_size() as u64
    }
}

struct RecordBatchBuilderImpl {
    schema: Arc<arrow_schema::Schema>,
    columns: RefCell<Vec<ArrayRef>>,
}

impl record_batch::GuestRecordBatchBuilder for RecordBatchBuilderImpl {
    fn new(schema: types::Schema) -> Self {
        let schema_impl = schema.get::<SchemaImpl>();
        Self {
            schema: schema_impl.inner.clone(),
            columns: RefCell::new(Vec::new()),
        }
    }

    fn append_column(&self, array: arrays::Array) -> Result<(), record_batch::ArrowError> {
        self.columns.borrow_mut().push(array.get::<ArrayImpl>().inner.clone());
        Ok(())
    }

    fn finish(&self) -> Result<record_batch::RecordBatch, record_batch::ArrowError> {
        let columns = std::mem::take(&mut *self.columns.borrow_mut());
        let batch = ArrowRecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| record_batch::ArrowError::InvalidArgument(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }))
    }
}

// ============================================================================
// Compute implementation (stub - to be expanded)
// ============================================================================

impl compute::Guest for Component {
    fn add(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::add(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn subtract(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::sub(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn multiply(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::mul(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn divide(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::div(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn modulo(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::rem(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn negate(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::numeric::neg(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    // ========== Wrapping Arithmetic ==========

    fn add_wrapping(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::add_wrapping(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn subtract_wrapping(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::sub_wrapping(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn multiply_wrapping(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_arith::numeric::mul_wrapping(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn negate_wrapping(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::numeric::neg_wrapping(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn add_scalar_i64(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Int64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::add(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn add_scalar_f64(arr: arrays::ArrayBorrow<'_>, scalar: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Float64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::add(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn multiply_scalar_i64(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Int64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::mul(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn multiply_scalar_f64(arr: arrays::ArrayBorrow<'_>, scalar: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Float64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::mul(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn subtract_scalar_i64(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Int64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::sub(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn subtract_scalar_f64(arr: arrays::ArrayBorrow<'_>, scalar: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Float64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::sub(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn divide_scalar_i64(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Int64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::div(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn divide_scalar_f64(arr: arrays::ArrayBorrow<'_>, scalar: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Float64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::div(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn modulo_scalar_i64(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Int64Array::new_scalar(scalar);
        let result = arrow_arith::numeric::rem(&arr_impl.inner, &scalar_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn multiply_decimal(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>, result_precision: u8, result_scale: i8) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        // Decimal128 - use multiply_fixed_point for precise result scale
        if let (Some(left_dec), Some(right_dec)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>()
        ) {
            let result = arrow_arith::arithmetic::multiply_fixed_point(left_dec, right_dec, result_scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            // Cast to the desired precision if needed
            let result_type = arrow_schema::DataType::Decimal128(result_precision, result_scale);
            let result = arrow_cast::cast(&result, &result_type)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Decimal256 - use regular multiply and cast (multiply_fixed_point doesn't support Decimal256)
        if left_impl.inner.as_any().is::<arrow_array::Decimal256Array>() &&
           right_impl.inner.as_any().is::<arrow_array::Decimal256Array>() {
            let result = arrow_arith::numeric::mul(&left_impl.inner, &right_impl.inner)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            let result_type = arrow_schema::DataType::Decimal256(result_precision, result_scale);
            let result = arrow_cast::cast(&result, &result_type)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        Err(compute::ArrowError::InvalidArgument("multiply_decimal requires Decimal128 or Decimal256 arrays".to_string()))
    }

    fn divide_decimal(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>, result_precision: u8, result_scale: i8) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        // Use standard divide and then cast to desired precision/scale
        // Note: There's no divide_fixed_point in arrow-arith, so we use regular divide
        let result = arrow_arith::numeric::div(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Cast result to desired precision and scale
        let result_type = if left_impl.inner.as_any().is::<arrow_array::Decimal128Array>() {
            arrow_schema::DataType::Decimal128(result_precision, result_scale)
        } else if left_impl.inner.as_any().is::<arrow_array::Decimal256Array>() {
            arrow_schema::DataType::Decimal256(result_precision, result_scale)
        } else {
            return Err(compute::ArrowError::InvalidArgument("divide_decimal requires Decimal128 or Decimal256 arrays".to_string()));
        };

        let result = arrow_cast::cast(&result, &result_type)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn decimal_rescale(arr: arrays::ArrayBorrow<'_>, new_precision: u8, new_scale: i8) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Decimal128
        if arr_impl.inner.as_any().is::<arrow_array::Decimal128Array>() {
            let result_type = arrow_schema::DataType::Decimal128(new_precision, new_scale);
            let result = arrow_cast::cast(&arr_impl.inner, &result_type)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Decimal256
        if arr_impl.inner.as_any().is::<arrow_array::Decimal256Array>() {
            let result_type = arrow_schema::DataType::Decimal256(new_precision, new_scale);
            let result = arrow_cast::cast(&arr_impl.inner, &result_type)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        Err(compute::ArrowError::InvalidArgument("decimal_rescale requires Decimal128 or Decimal256 array".to_string()))
    }

    fn decimal_round(arr: arrays::ArrayBorrow<'_>, scale: i8) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Decimal128
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>() {
            let precision = dec_arr.precision();
            let current_scale = dec_arr.scale();

            // If target scale is greater or equal, just rescale
            if scale >= current_scale {
                let result_type = arrow_schema::DataType::Decimal128(precision, scale);
                let result = arrow_cast::cast(&arr_impl.inner, &result_type)
                    .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: result }));
            }

            // Round to lower scale by casting (arrow handles rounding)
            let scale_diff = (current_scale - scale) as u32;
            let divisor = 10i128.pow(scale_diff);
            let half = divisor / 2;

            let result: arrow_array::Decimal128Array = dec_arr.iter()
                .map(|opt| opt.map(|v| {
                    let sign = if v < 0 { -1 } else { 1 };
                    let abs_v = v.abs();
                    let remainder = abs_v % divisor;
                    let base = abs_v / divisor;
                    if remainder >= half {
                        sign * (base + 1)
                    } else {
                        sign * base
                    }
                }))
                .collect();

            let result = result.with_precision_and_scale(precision, scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Decimal256
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>() {
            let precision = dec_arr.precision();

            // Cast to target scale (arrow handles rounding)
            let result_type = arrow_schema::DataType::Decimal256(precision, scale);
            let result = arrow_cast::cast(&arr_impl.inner, &result_type)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        Err(compute::ArrowError::InvalidArgument("decimal_round requires Decimal128 or Decimal256 array".to_string()))
    }

    fn decimal_trunc(arr: arrays::ArrayBorrow<'_>, scale: i8) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Decimal128
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>() {
            let precision = dec_arr.precision();
            let current_scale = dec_arr.scale();

            // If target scale is greater or equal, just rescale
            if scale >= current_scale {
                let result_type = arrow_schema::DataType::Decimal128(precision, scale);
                let result = arrow_cast::cast(&arr_impl.inner, &result_type)
                    .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
                return Ok(arrays::Array::new(ArrayImpl { inner: result }));
            }

            // Truncate to lower scale
            let scale_diff = (current_scale - scale) as u32;
            let divisor = 10i128.pow(scale_diff);

            let result: arrow_array::Decimal128Array = dec_arr.iter()
                .map(|opt| opt.map(|v| {
                    let sign = if v < 0 { -1 } else { 1 };
                    let abs_v = v.abs();
                    sign * (abs_v / divisor)
                }))
                .collect();

            let result = result.with_precision_and_scale(precision, scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Decimal256 - use cast with truncation mode
        if arr_impl.inner.as_any().is::<arrow_array::Decimal256Array>() {
            let dec_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>().unwrap();
            let precision = dec_arr.precision();

            // Cast to target scale
            let result_type = arrow_schema::DataType::Decimal256(precision, scale);
            let result = arrow_cast::cast(&arr_impl.inner, &result_type)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        Err(compute::ArrowError::InvalidArgument("decimal_trunc requires Decimal128 or Decimal256 array".to_string()))
    }

    fn decimal_abs(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Decimal128
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>() {
            let precision = dec_arr.precision();
            let scale = dec_arr.scale();

            let result: arrow_array::Decimal128Array = dec_arr.iter()
                .map(|opt| opt.map(|v| v.abs()))
                .collect();

            let result = result.with_precision_and_scale(precision, scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Decimal256
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>() {
            let precision = dec_arr.precision();
            let scale = dec_arr.scale();

            let result: arrow_array::Decimal256Array = dec_arr.iter()
                .map(|opt| opt.map(|v| {
                    if v.is_negative() { -v } else { v }
                }))
                .collect();

            let result = result.with_precision_and_scale(precision, scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("decimal_abs requires Decimal128 or Decimal256 array".to_string()))
    }

    fn decimal_negate(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Decimal128
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>() {
            let precision = dec_arr.precision();
            let scale = dec_arr.scale();

            let result: arrow_array::Decimal128Array = dec_arr.iter()
                .map(|opt| opt.map(|v| -v))
                .collect();

            let result = result.with_precision_and_scale(precision, scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Decimal256
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>() {
            let precision = dec_arr.precision();
            let scale = dec_arr.scale();

            let result: arrow_array::Decimal256Array = dec_arr.iter()
                .map(|opt| opt.map(|v| -v))
                .collect();

            let result = result.with_precision_and_scale(precision, scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("decimal_negate requires Decimal128 or Decimal256 array".to_string()))
    }

    fn decimal_sum(arr: arrays::ArrayBorrow<'_>, result_precision: u8, result_scale: i8) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Decimal128
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>() {
            let current_scale = dec_arr.scale();
            let scale_diff = result_scale - current_scale;
            let scale_factor = 10i128.pow(scale_diff.unsigned_abs() as u32);

            let mut sum: i128 = 0;
            let mut has_value = false;

            for opt in dec_arr.iter() {
                if let Some(v) = opt {
                    has_value = true;
                    // Adjust scale
                    let adjusted = if scale_diff > 0 {
                        v * scale_factor
                    } else if scale_diff < 0 {
                        v / scale_factor
                    } else {
                        v
                    };
                    sum = sum.saturating_add(adjusted);
                }
            }

            let result = if has_value {
                arrow_array::Decimal128Array::from(vec![Some(sum)])
            } else {
                arrow_array::Decimal128Array::from(vec![None])
            };

            let result = result.with_precision_and_scale(result_precision, result_scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Decimal256
        if let Some(dec_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>() {
            use arrow_buffer::i256;

            let current_scale = dec_arr.scale();
            let scale_diff = result_scale - current_scale;
            let scale_factor = i256::from_i128(10i128.pow(scale_diff.unsigned_abs() as u32));

            let mut sum = i256::ZERO;
            let mut has_value = false;

            for opt in dec_arr.iter() {
                if let Some(v) = opt {
                    has_value = true;
                    // Adjust scale
                    let adjusted = if scale_diff > 0 {
                        v * scale_factor
                    } else if scale_diff < 0 {
                        v / scale_factor
                    } else {
                        v
                    };
                    sum = sum.wrapping_add(adjusted);
                }
            }

            let result = if has_value {
                arrow_array::Decimal256Array::from(vec![Some(sum)])
            } else {
                arrow_array::Decimal256Array::from(vec![None])
            };

            let result = result.with_precision_and_scale(result_precision, result_scale)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("decimal_sum requires Decimal128 or Decimal256 array".to_string()))
    }

    // ========== Mathematical Functions ==========

    fn abs(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Try different numeric types
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Int64Array = int_arr.iter()
                .map(|v| v.map(|x| x.abs()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result: arrow_array::Int32Array = int_arr.iter()
                .map(|v| v.map(|x| x.abs()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.abs()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.abs()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("abs requires a numeric array".to_string()))
    }

    fn round(arr: arrays::ArrayBorrow<'_>, decimals: i32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let multiplier = 10_f64.powi(decimals);

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| (x * multiplier).round() / multiplier))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let multiplier = 10_f32.powi(decimals);
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| (x * multiplier).round() / multiplier))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // For integer types, return as-is (already rounded)
        if arr_impl.inner.as_any().is::<arrow_array::Int64Array>() ||
           arr_impl.inner.as_any().is::<arrow_array::Int32Array>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        Err(compute::ArrowError::InvalidArgument("round requires a numeric array".to_string()))
    }

    fn ceil(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.ceil()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.ceil()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // For integer types, return as-is
        if arr_impl.inner.as_any().is::<arrow_array::Int64Array>() ||
           arr_impl.inner.as_any().is::<arrow_array::Int32Array>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        Err(compute::ArrowError::InvalidArgument("ceil requires a numeric array".to_string()))
    }

    fn floor(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.floor()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.floor()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // For integer types, return as-is
        if arr_impl.inner.as_any().is::<arrow_array::Int64Array>() ||
           arr_impl.inner.as_any().is::<arrow_array::Int32Array>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        Err(compute::ArrowError::InvalidArgument("floor requires a numeric array".to_string()))
    }

    fn trunc(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.trunc()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.trunc()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // For integer types, return as-is
        if arr_impl.inner.as_any().is::<arrow_array::Int64Array>() ||
           arr_impl.inner.as_any().is::<arrow_array::Int32Array>() {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        Err(compute::ArrowError::InvalidArgument("trunc requires a numeric array".to_string()))
    }

    fn sqrt(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.sqrt()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.sqrt()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // For integer types, cast to f64 first
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).sqrt()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("sqrt requires a numeric array".to_string()))
    }

    fn cbrt(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.cbrt()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.cbrt()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // For integer types, cast to f64 first
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).cbrt()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("cbrt requires a numeric array".to_string()))
    }

    fn pow(base: arrays::ArrayBorrow<'_>, exponent: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let base_impl = base.get::<ArrayImpl>();
        let exp_impl = exponent.get::<ArrayImpl>();

        if let (Some(base_arr), Some(exp_arr)) = (
            base_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
            exp_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
        ) {
            if base_arr.len() != exp_arr.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::Float64Array = base_arr.iter().zip(exp_arr.iter())
                .map(|(b, e)| match (b, e) {
                    (Some(b), Some(e)) => Some(b.powf(e)),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("pow requires Float64 arrays".to_string()))
    }

    fn pow_scalar(arr: arrays::ArrayBorrow<'_>, exponent: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.powf(exponent)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.powf(exponent as f32)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).powf(exponent)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("pow_scalar requires a numeric array".to_string()))
    }

    fn exp(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.exp()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.exp()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).exp()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("exp requires a numeric array".to_string()))
    }

    fn ln(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.ln()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.ln()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).ln()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("ln requires a numeric array".to_string()))
    }

    fn log2(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.log2()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.log2()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).log2()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("log2 requires a numeric array".to_string()))
    }

    fn log10(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.log10()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.log10()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).log10()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("log10 requires a numeric array".to_string()))
    }

    fn log(arr: arrays::ArrayBorrow<'_>, base: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter()
                .map(|v| v.map(|x| x.log(base)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter()
                .map(|v| v.map(|x| x.log(base as f32)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Float64Array = int_arr.iter()
                .map(|v| v.map(|x| (x as f64).log(base)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("log requires a numeric array".to_string()))
    }

    fn sign(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Int32Array = float_arr.iter()
                .map(|v| v.map(|x| {
                    if x > 0.0 { 1 }
                    else if x < 0.0 { -1 }
                    else { 0 }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Int32Array = float_arr.iter()
                .map(|v| v.map(|x| {
                    if x > 0.0 { 1 }
                    else if x < 0.0 { -1 }
                    else { 0 }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::Int32Array = int_arr.iter()
                .map(|v| v.map(|x| {
                    if x > 0 { 1 }
                    else if x < 0 { -1 }
                    else { 0 }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result: arrow_array::Int32Array = int_arr.iter()
                .map(|v| v.map(|x| {
                    if x > 0 { 1 }
                    else if x < 0 { -1 }
                    else { 0 }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("sign requires a numeric array".to_string()))
    }

    // ========== Extended Mathematical Functions ==========

    fn degrees(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.to_degrees())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.to_degrees())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("degrees requires a float array".to_string()))
    }

    fn radians(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.to_radians())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.to_radians())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("radians requires a float array".to_string()))
    }

    fn hypot(x: arrays::ArrayBorrow<'_>, y: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let x_impl = x.get::<ArrayImpl>();
        let y_impl = y.get::<ArrayImpl>();

        // Float64
        if let (Some(x_arr), Some(y_arr)) = (
            x_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
            y_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
        ) {
            if x_arr.len() != y_arr.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
            }
            let result: arrow_array::Float64Array = x_arr.iter().zip(y_arr.iter())
                .map(|(xv, yv)| match (xv, yv) {
                    (Some(x), Some(y)) => Some(x.hypot(y)),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Float32
        if let (Some(x_arr), Some(y_arr)) = (
            x_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>(),
            y_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>(),
        ) {
            if x_arr.len() != y_arr.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
            }
            let result: arrow_array::Float32Array = x_arr.iter().zip(y_arr.iter())
                .map(|(xv, yv)| match (xv, yv) {
                    (Some(x), Some(y)) => Some(x.hypot(y)),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("hypot requires float arrays of the same type".to_string()))
    }

    fn expm1(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.exp_m1())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.exp_m1())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("expm1 requires a float array".to_string()))
    }

    fn log1p(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.ln_1p())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.ln_1p())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("log1p requires a float array".to_string()))
    }

    fn copysign(magnitude: arrays::ArrayBorrow<'_>, sign: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let mag_impl = magnitude.get::<ArrayImpl>();
        let sign_impl = sign.get::<ArrayImpl>();

        // Float64
        if let (Some(mag_arr), Some(sign_arr)) = (
            mag_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
            sign_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
        ) {
            if mag_arr.len() != sign_arr.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
            }
            let result: arrow_array::Float64Array = mag_arr.iter().zip(sign_arr.iter())
                .map(|(mv, sv)| match (mv, sv) {
                    (Some(m), Some(s)) => Some(m.copysign(s)),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Float32
        if let (Some(mag_arr), Some(sign_arr)) = (
            mag_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>(),
            sign_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>(),
        ) {
            if mag_arr.len() != sign_arr.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
            }
            let result: arrow_array::Float32Array = mag_arr.iter().zip(sign_arr.iter())
                .map(|(mv, sv)| match (mv, sv) {
                    (Some(m), Some(s)) => Some(m.copysign(s)),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("copysign requires float arrays of the same type".to_string()))
    }

    fn fmax(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        macro_rules! fmax_impl {
            ($arr_type:ty) => {
                if let (Some(l_arr), Some(r_arr)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    if l_arr.len() != r_arr.len() {
                        return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
                    }
                    let result: $arr_type = l_arr.iter().zip(r_arr.iter())
                        .map(|(lv, rv)| match (lv, rv) {
                            (Some(l), Some(r)) => Some(l.max(r)),
                            (Some(l), None) => Some(l),
                            (None, Some(r)) => Some(r),
                            (None, None) => None,
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        fmax_impl!(arrow_array::Float64Array);
        fmax_impl!(arrow_array::Float32Array);
        fmax_impl!(arrow_array::Int64Array);
        fmax_impl!(arrow_array::Int32Array);
        fmax_impl!(arrow_array::Int16Array);
        fmax_impl!(arrow_array::Int8Array);
        fmax_impl!(arrow_array::UInt64Array);
        fmax_impl!(arrow_array::UInt32Array);
        fmax_impl!(arrow_array::UInt16Array);
        fmax_impl!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("fmax requires numeric arrays of the same type".to_string()))
    }

    fn fmin(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        macro_rules! fmin_impl {
            ($arr_type:ty) => {
                if let (Some(l_arr), Some(r_arr)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    if l_arr.len() != r_arr.len() {
                        return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
                    }
                    let result: $arr_type = l_arr.iter().zip(r_arr.iter())
                        .map(|(lv, rv)| match (lv, rv) {
                            (Some(l), Some(r)) => Some(l.min(r)),
                            (Some(l), None) => Some(l),
                            (None, Some(r)) => Some(r),
                            (None, None) => None,
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        fmin_impl!(arrow_array::Float64Array);
        fmin_impl!(arrow_array::Float32Array);
        fmin_impl!(arrow_array::Int64Array);
        fmin_impl!(arrow_array::Int32Array);
        fmin_impl!(arrow_array::Int16Array);
        fmin_impl!(arrow_array::Int8Array);
        fmin_impl!(arrow_array::UInt64Array);
        fmin_impl!(arrow_array::UInt32Array);
        fmin_impl!(arrow_array::UInt16Array);
        fmin_impl!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("fmin requires numeric arrays of the same type".to_string()))
    }

    fn gcd(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        fn compute_gcd<T: num_traits::Signed + Copy>(mut a: T, mut b: T) -> T {
            while !b.is_zero() {
                let t = b;
                b = a % b;
                a = t;
            }
            a.abs()
        }

        fn compute_gcd_unsigned<T: num_traits::Unsigned + Copy>(mut a: T, mut b: T) -> T {
            while !b.is_zero() {
                let t = b;
                b = a % b;
                a = t;
            }
            a
        }

        macro_rules! gcd_impl_signed {
            ($arr_type:ty, $native_type:ty) => {
                if let (Some(l_arr), Some(r_arr)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    if l_arr.len() != r_arr.len() {
                        return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
                    }
                    let result: $arr_type = l_arr.iter().zip(r_arr.iter())
                        .map(|(lv, rv)| match (lv, rv) {
                            (Some(l), Some(r)) => Some(compute_gcd::<$native_type>(l, r)),
                            _ => None,
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        macro_rules! gcd_impl_unsigned {
            ($arr_type:ty, $native_type:ty) => {
                if let (Some(l_arr), Some(r_arr)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    if l_arr.len() != r_arr.len() {
                        return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
                    }
                    let result: $arr_type = l_arr.iter().zip(r_arr.iter())
                        .map(|(lv, rv)| match (lv, rv) {
                            (Some(l), Some(r)) => Some(compute_gcd_unsigned::<$native_type>(l, r)),
                            _ => None,
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        gcd_impl_signed!(arrow_array::Int64Array, i64);
        gcd_impl_signed!(arrow_array::Int32Array, i32);
        gcd_impl_signed!(arrow_array::Int16Array, i16);
        gcd_impl_signed!(arrow_array::Int8Array, i8);
        gcd_impl_unsigned!(arrow_array::UInt64Array, u64);
        gcd_impl_unsigned!(arrow_array::UInt32Array, u32);
        gcd_impl_unsigned!(arrow_array::UInt16Array, u16);
        gcd_impl_unsigned!(arrow_array::UInt8Array, u8);

        Err(compute::ArrowError::InvalidArgument("gcd requires integer arrays of the same type".to_string()))
    }

    fn lcm(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        fn compute_gcd<T: num_traits::Signed + Copy>(mut a: T, mut b: T) -> T {
            while !b.is_zero() {
                let t = b;
                b = a % b;
                a = t;
            }
            a.abs()
        }

        fn compute_gcd_unsigned<T: num_traits::Unsigned + Copy>(mut a: T, mut b: T) -> T {
            while !b.is_zero() {
                let t = b;
                b = a % b;
                a = t;
            }
            a
        }

        fn compute_lcm<T: num_traits::Signed + Copy + std::ops::Mul<Output = T> + std::ops::Div<Output = T>>(a: T, b: T) -> T {
            if a.is_zero() || b.is_zero() {
                return T::zero();
            }
            (a / compute_gcd(a, b)) * b
        }

        fn compute_lcm_unsigned<T: num_traits::Unsigned + Copy + std::ops::Mul<Output = T> + std::ops::Div<Output = T>>(a: T, b: T) -> T {
            if a.is_zero() || b.is_zero() {
                return T::zero();
            }
            (a / compute_gcd_unsigned(a, b)) * b
        }

        macro_rules! lcm_impl_signed {
            ($arr_type:ty, $native_type:ty) => {
                if let (Some(l_arr), Some(r_arr)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    if l_arr.len() != r_arr.len() {
                        return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
                    }
                    let result: $arr_type = l_arr.iter().zip(r_arr.iter())
                        .map(|(lv, rv)| match (lv, rv) {
                            (Some(l), Some(r)) => Some(compute_lcm::<$native_type>(l.abs(), r.abs())),
                            _ => None,
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        macro_rules! lcm_impl_unsigned {
            ($arr_type:ty, $native_type:ty) => {
                if let (Some(l_arr), Some(r_arr)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    if l_arr.len() != r_arr.len() {
                        return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
                    }
                    let result: $arr_type = l_arr.iter().zip(r_arr.iter())
                        .map(|(lv, rv)| match (lv, rv) {
                            (Some(l), Some(r)) => Some(compute_lcm_unsigned::<$native_type>(l, r)),
                            _ => None,
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        lcm_impl_signed!(arrow_array::Int64Array, i64);
        lcm_impl_signed!(arrow_array::Int32Array, i32);
        lcm_impl_signed!(arrow_array::Int16Array, i16);
        lcm_impl_signed!(arrow_array::Int8Array, i8);
        lcm_impl_unsigned!(arrow_array::UInt64Array, u64);
        lcm_impl_unsigned!(arrow_array::UInt32Array, u32);
        lcm_impl_unsigned!(arrow_array::UInt16Array, u16);
        lcm_impl_unsigned!(arrow_array::UInt8Array, u8);

        Err(compute::ArrowError::InvalidArgument("lcm requires integer arrays of the same type".to_string()))
    }

    // ========== Trigonometric Functions ==========

    fn sin(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.sin())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.sin())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("sin requires a float array".to_string()))
    }

    fn cos(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.cos())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.cos())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("cos requires a float array".to_string()))
    }

    fn tan(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.tan())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.tan())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("tan requires a float array".to_string()))
    }

    fn asin(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.asin())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.asin())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("asin requires a float array".to_string()))
    }

    fn acos(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.acos())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.acos())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("acos requires a float array".to_string()))
    }

    fn atan(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.atan())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.atan())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("atan requires a float array".to_string()))
    }

    fn atan2(y: arrays::ArrayBorrow<'_>, x: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let y_impl = y.get::<ArrayImpl>();
        let x_impl = x.get::<ArrayImpl>();
        if let (Some(y_arr), Some(x_arr)) = (
            y_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
            x_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
        ) {
            if y_arr.len() != x_arr.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::Float64Array = y_arr.iter().zip(x_arr.iter())
                .map(|(y, x)| match (y, x) { (Some(y), Some(x)) => Some(y.atan2(x)), _ => None })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("atan2 requires Float64 arrays".to_string()))
    }

    fn sinh(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.sinh())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.sinh())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("sinh requires a float array".to_string()))
    }

    fn cosh(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.cosh())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.cosh())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("cosh requires a float array".to_string()))
    }

    fn tanh(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::Float64Array = float_arr.iter().map(|v| v.map(|x| x.tanh())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let result: arrow_array::Float32Array = float_arr.iter().map(|v| v.map(|x| x.tanh())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("tanh requires a float array".to_string()))
    }

    fn compare(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>, op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result: BooleanArray = match op {
            compute::ComparisonOp::Eq => arrow_ord::cmp::eq(&left_impl.inner, &right_impl.inner),
            compute::ComparisonOp::NotEq => arrow_ord::cmp::neq(&left_impl.inner, &right_impl.inner),
            compute::ComparisonOp::Lt => arrow_ord::cmp::lt(&left_impl.inner, &right_impl.inner),
            compute::ComparisonOp::LtEq => arrow_ord::cmp::lt_eq(&left_impl.inner, &right_impl.inner),
            compute::ComparisonOp::Gt => arrow_ord::cmp::gt(&left_impl.inner, &right_impl.inner),
            compute::ComparisonOp::GtEq => arrow_ord::cmp::gt_eq(&left_impl.inner, &right_impl.inner),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn compare_scalar_i64(arr: arrays::ArrayBorrow<'_>, scalar: i64, op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Int64Array::new_scalar(scalar);
        let result: BooleanArray = match op {
            compute::ComparisonOp::Eq => arrow_ord::cmp::eq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::NotEq => arrow_ord::cmp::neq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::Lt => arrow_ord::cmp::lt(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::LtEq => arrow_ord::cmp::lt_eq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::Gt => arrow_ord::cmp::gt(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::GtEq => arrow_ord::cmp::gt_eq(&arr_impl.inner, &scalar_arr),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn compare_scalar_f64(arr: arrays::ArrayBorrow<'_>, scalar: f64, op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Float64Array::new_scalar(scalar);
        let result: BooleanArray = match op {
            compute::ComparisonOp::Eq => arrow_ord::cmp::eq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::NotEq => arrow_ord::cmp::neq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::Lt => arrow_ord::cmp::lt(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::LtEq => arrow_ord::cmp::lt_eq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::Gt => arrow_ord::cmp::gt(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::GtEq => arrow_ord::cmp::gt_eq(&arr_impl.inner, &scalar_arr),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn compare_scalar_string(arr: arrays::ArrayBorrow<'_>, scalar: String, op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar_arr = arrow_array::Scalar::new(arrow_array::StringArray::from(vec![scalar.as_str()]));
        let result: BooleanArray = match op {
            compute::ComparisonOp::Eq => arrow_ord::cmp::eq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::NotEq => arrow_ord::cmp::neq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::Lt => arrow_ord::cmp::lt(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::LtEq => arrow_ord::cmp::lt_eq(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::Gt => arrow_ord::cmp::gt(&arr_impl.inner, &scalar_arr),
            compute::ComparisonOp::GtEq => arrow_ord::cmp::gt_eq(&arr_impl.inner, &scalar_arr),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn distinct(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_ord::cmp::distinct(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn not_distinct(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_ord::cmp::not_distinct(&left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    // ========== List Membership ==========

    fn in_list_i64(arr: arrays::ArrayBorrow<'_>, values: Vec<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let i64_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))?;

        // Create a HashSet for fast lookups
        let value_set: std::collections::HashSet<i64> = values.into_iter().collect();

        // Check each element
        let result: BooleanArray = i64_arr.iter()
            .map(|opt| opt.map(|v| value_set.contains(&v)))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn in_list_string(arr: arrays::ArrayBorrow<'_>, values: Vec<String>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected String array".to_string()))?;

        // Create a HashSet for fast lookups
        let value_set: std::collections::HashSet<String> = values.into_iter().collect();

        // Check each element
        let result: BooleanArray = str_arr.iter()
            .map(|opt| opt.map(|v| value_set.contains(v)))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn and(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let right_bool = right_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let result = arrow_arith::boolean::and(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn or(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let right_bool = right_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let result = arrow_arith::boolean::or(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn not(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bool_arr = arr_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let result = arrow_arith::boolean::not(bool_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn and_not(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let right_bool = right_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let result = arrow_arith::boolean::and_not(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn and_kleene(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let right_bool = right_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let result = arrow_arith::boolean::and_kleene(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn or_kleene(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let right_bool = right_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        let result = arrow_arith::boolean::or_kleene(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn is_null(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::boolean::is_null(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn is_not_null(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::boolean::is_not_null(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn sum_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int64Type>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))?;
        Ok(arrow_arith::aggregate::sum(prim_arr))
    }

    fn sum_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Float64Type>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Float64 array".to_string()))?;
        Ok(arrow_arith::aggregate::sum(prim_arr))
    }

    fn min_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int64Type>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))?;
        Ok(arrow_arith::aggregate::min(prim_arr))
    }

    fn min_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Float64Type>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Float64 array".to_string()))?;
        Ok(arrow_arith::aggregate::min(prim_arr))
    }

    fn min_string(arr: arrays::ArrayBorrow<'_>) -> Result<Option<String>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_string_opt::<i32>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected String array".to_string()))?;
        Ok(arrow_arith::aggregate::min_string(str_arr).map(|s| s.to_string()))
    }

    fn max_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int64Type>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))?;
        Ok(arrow_arith::aggregate::max(prim_arr))
    }

    fn max_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let prim_arr = arr_impl.inner.as_primitive_opt::<arrow_array::types::Float64Type>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Float64 array".to_string()))?;
        Ok(arrow_arith::aggregate::max(prim_arr))
    }

    fn max_string(arr: arrays::ArrayBorrow<'_>) -> Result<Option<String>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_string_opt::<i32>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected String array".to_string()))?;
        Ok(arrow_arith::aggregate::max_string(str_arr).map(|s| s.to_string()))
    }

    fn min_binary(arr: arrays::ArrayBorrow<'_>) -> Result<Option<Vec<u8>>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bin_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Binary array".to_string()))?;
        Ok(arrow_arith::aggregate::min_binary(bin_arr).map(|b| b.to_vec()))
    }

    fn max_binary(arr: arrays::ArrayBorrow<'_>) -> Result<Option<Vec<u8>>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bin_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Binary array".to_string()))?;
        Ok(arrow_arith::aggregate::max_binary(bin_arr).map(|b| b.to_vec()))
    }

    fn count(arr: arrays::ArrayBorrow<'_>) -> u64 {
        let arr_impl = arr.get::<ArrayImpl>();
        (arr_impl.inner.len() - arr_impl.inner.null_count()) as u64
    }

    fn mean(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let count = arr_impl.inner.len() - arr_impl.inner.null_count();
        if count == 0 {
            return Ok(None);
        }
        if let Some(prim_arr) = arr_impl.inner.as_primitive_opt::<arrow_array::types::Float64Type>() {
            let sum: Option<f64> = arrow_arith::aggregate::sum(prim_arr);
            return Ok(sum.map(|s| s / count as f64));
        }
        if let Some(prim_arr) = arr_impl.inner.as_primitive_opt::<arrow_array::types::Int64Type>() {
            let sum: Option<i64> = arrow_arith::aggregate::sum(prim_arr);
            return Ok(sum.map(|s| s as f64 / count as f64));
        }
        Err(compute::ArrowError::InvalidArgument("Mean requires numeric array".to_string()))
    }

    fn filter(arr: arrays::ArrayBorrow<'_>, predicate: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let pred_impl = predicate.get::<ArrayImpl>();
        let pred_bool = pred_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Predicate must be boolean".to_string()))?;
        let result = arrow_select::filter::filter(&arr_impl.inner, pred_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn filter_record_batch(batch: record_batch::RecordBatchBorrow<'_>, predicate: arrays::ArrayBorrow<'_>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let pred_impl = predicate.get::<ArrayImpl>();
        let pred_bool = pred_impl.inner.as_boolean_opt()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Predicate must be boolean".to_string()))?;
        let result = arrow_select::filter::filter_record_batch(&batch_impl.inner, pred_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn take(arr: arrays::ArrayBorrow<'_>, indices: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let indices_impl = indices.get::<ArrayImpl>();
        let result = arrow_select::take::take(&arr_impl.inner, &indices_impl.inner, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn take_record_batch(batch: record_batch::RecordBatchBorrow<'_>, indices: arrays::ArrayBorrow<'_>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let indices_impl = indices.get::<ArrayImpl>();
        let new_columns: Result<Vec<ArrayRef>, _> = batch_impl.inner.columns().iter()
            .map(|col| arrow_select::take::take(col, &indices_impl.inner, None))
            .collect();
        let new_columns = new_columns.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        let result = ArrowRecordBatch::try_new(batch_impl.inner.schema(), new_columns)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn sort_indices(arr: arrays::ArrayBorrow<'_>, options: compute::SortOptions) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_options = arrow_ord::sort::SortOptions {
            descending: options.descending,
            nulls_first: options.nulls_first,
        };
        let indices = arrow_ord::sort::sort_to_indices(&arr_impl.inner, Some(sort_options), None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(indices) }))
    }

    fn sort(arr: arrays::ArrayBorrow<'_>, options: compute::SortOptions) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_options = arrow_ord::sort::SortOptions {
            descending: options.descending,
            nulls_first: options.nulls_first,
        };
        let result = arrow_ord::sort::sort(&arr_impl.inner, Some(sort_options))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn sort_record_batch(batch: record_batch::RecordBatchBorrow<'_>, sort_columns: Vec<(String, compute::SortOptions)>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();

        // Build SortColumn array for lexsort
        let mut columns: Vec<arrow_row::SortField> = Vec::new();
        let mut arrays_to_sort: Vec<Arc<dyn arrow_array::Array>> = Vec::new();

        for (col_name, options) in &sort_columns {
            let col_idx = batch_impl.inner.schema().index_of(col_name)
                .map_err(|e| compute::ArrowError::InvalidArgument(e.to_string()))?;
            let col = batch_impl.inner.column(col_idx).clone();

            let sort_options = arrow_ord::sort::SortOptions {
                descending: options.descending,
                nulls_first: options.nulls_first,
            };

            columns.push(arrow_row::SortField::new_with_options(col.data_type().clone(), sort_options));
            arrays_to_sort.push(col);
        }

        // Use lexsort_to_indices to get sorted indices
        let sort_columns: Vec<arrow_ord::sort::SortColumn> = arrays_to_sort.iter()
            .zip(sort_columns.iter())
            .map(|(arr, (_, opts))| arrow_ord::sort::SortColumn {
                values: arr.clone(),
                options: Some(arrow_ord::sort::SortOptions {
                    descending: opts.descending,
                    nulls_first: opts.nulls_first,
                }),
            })
            .collect();

        let indices = arrow_ord::sort::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Take rows in sorted order
        let sorted_columns: Result<Vec<_>, _> = batch_impl.inner.columns().iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &indices, None))
            .collect();
        let sorted_columns = sorted_columns
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        let result = arrow_array::RecordBatch::try_new(batch_impl.inner.schema(), sorted_columns)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn lexsort(arrays: Vec<arrays::Array>, options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        if arrays.len() != options.len() {
            return Err(compute::ArrowError::InvalidArgument(
                "arrays and options must have the same length".to_string()
            ));
        }

        if arrays.is_empty() {
            return Err(compute::ArrowError::InvalidArgument(
                "lexsort requires at least one array".to_string()
            ));
        }

        // Build SortColumn array
        let sort_columns: Vec<arrow_ord::sort::SortColumn> = arrays.iter()
            .zip(options.iter())
            .map(|(arr, opts)| {
                let arr_impl = arr.get::<ArrayImpl>();
                arrow_ord::sort::SortColumn {
                    values: arr_impl.inner.clone(),
                    options: Some(arrow_ord::sort::SortOptions {
                        descending: opts.descending,
                        nulls_first: opts.nulls_first,
                    }),
                }
            })
            .collect();

        let indices = arrow_ord::sort::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(indices) }))
    }

    fn sort_limit(arr: arrays::ArrayBorrow<'_>, options: compute::SortOptions, limit: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_opts = arrow_ord::sort::SortOptions {
            descending: options.descending,
            nulls_first: options.nulls_first,
        };

        // Get sort indices with limit
        let indices = arrow_ord::sort::sort_to_indices(&*arr_impl.inner, Some(sort_opts), Some(limit as usize))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Take values at those indices
        let result = arrow_select::take::take(&*arr_impl.inner, &indices, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn sort_indices_limit(arr: arrays::ArrayBorrow<'_>, options: compute::SortOptions, limit: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_opts = arrow_ord::sort::SortOptions {
            descending: options.descending,
            nulls_first: options.nulls_first,
        };

        // Get sort indices with limit
        let indices = arrow_ord::sort::sort_to_indices(&*arr_impl.inner, Some(sort_opts), Some(limit as usize))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(indices) }))
    }

    fn limit(arr: arrays::ArrayBorrow<'_>, n: u64) -> arrays::Array {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len().min(n as usize);
        arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(0, len) })
    }

    fn skip(arr: arrays::ArrayBorrow<'_>, n: u64) -> arrays::Array {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();
        let offset = (n as usize).min(len);
        arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(offset, len - offset) })
    }

    fn shift(arr: arrays::ArrayBorrow<'_>, offset: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_select::window::shift(&*arr_impl.inner, offset)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn cast(arr: arrays::ArrayBorrow<'_>, to_type: types::DataType) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let arrow_type = convert::to_arrow_data_type(&to_type);
        let result = arrow_cast::cast(&arr_impl.inner, &arrow_type)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn can_cast_types(from_type: types::DataType, to_type: types::DataType) -> bool {
        let from_arrow = convert::to_arrow_data_type(&from_type);
        let to_arrow = convert::to_arrow_data_type(&to_type);
        arrow_cast::can_cast_types(&from_arrow, &to_arrow)
    }

    fn try_cast(arr: arrays::ArrayBorrow<'_>, to_type: types::DataType) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_cast::CastOptions;
        let arr_impl = arr.get::<ArrayImpl>();
        let arrow_type = convert::to_arrow_data_type(&to_type);

        // Use safe cast options that don't error on invalid values
        let options = CastOptions {
            safe: true, // Return null instead of error for invalid values
            ..Default::default()
        };

        let result = arrow_cast::cast_with_options(&arr_impl.inner, &arrow_type, &options)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn cast_with_options(arr: arrays::ArrayBorrow<'_>, to_type: types::DataType, options: compute::CastOptions) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_cast::CastOptions as ArrowCastOptions;
        let arr_impl = arr.get::<ArrayImpl>();
        let arrow_type = convert::to_arrow_data_type(&to_type);

        let cast_options = ArrowCastOptions {
            safe: options.safe,
            format_options: if let Some(ref fmt) = options.format_string {
                arrow_cast::display::FormatOptions::default().with_datetime_format(Some(fmt.as_str()))
            } else {
                arrow_cast::display::FormatOptions::default()
            },
            ..Default::default()
        };

        let result = arrow_cast::cast_with_options(&arr_impl.inner, &arrow_type, &cast_options)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn parse_string(arr: arrays::ArrayBorrow<'_>, to_type: types::DataType, format: Option<String>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Verify input is a string array
        let _str_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("parse_string requires a String array".to_string()))?;

        let arrow_type = convert::to_arrow_data_type(&to_type);

        // For timestamps/dates with custom format, use arrow's parse capability with format options
        let cast_options = if let Some(ref fmt) = format {
            arrow_cast::CastOptions {
                safe: true,
                format_options: arrow_cast::display::FormatOptions::default()
                    .with_datetime_format(Some(fmt.as_str())),
                ..Default::default()
            }
        } else {
            arrow_cast::CastOptions {
                safe: true,
                ..Default::default()
            }
        };

        let result = arrow_cast::cast_with_options(&arr_impl.inner, &arrow_type, &cast_options)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn string_length(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_string::length::length(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn bit_length(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_string::length::bit_length(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn string_lower(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Manual lowercase implementation
        let result: arrow_array::StringArray = string_arr
            .iter()
            .map(|opt| opt.map(|s| s.to_lowercase()))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_upper(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Manual uppercase implementation
        let result: arrow_array::StringArray = string_arr
            .iter()
            .map(|opt| opt.map(|s| s.to_uppercase()))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_trim(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Manual trim implementation
        let result: arrow_array::StringArray = string_arr
            .iter()
            .map(|opt| opt.map(|s| s.trim().to_string()))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_contains(arr: arrays::ArrayBorrow<'_>, substring: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let scalar = arrow_array::StringArray::new_scalar(&substring);
        let result = arrow_string::like::contains(string_arr, &scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_starts_with(arr: arrays::ArrayBorrow<'_>, prefix: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let scalar = arrow_array::StringArray::new_scalar(&prefix);
        let result = arrow_string::like::starts_with(string_arr, &scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_ends_with(arr: arrays::ArrayBorrow<'_>, suffix: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let scalar = arrow_array::StringArray::new_scalar(&suffix);
        let result = arrow_string::like::ends_with(string_arr, &scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_concat(arr_list: Vec<arrays::Array>) -> Result<arrays::Array, compute::ArrowError> {
        if arr_list.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("No arrays to concatenate".to_string()));
        }
        // Concatenate string arrays element-wise using concat_elements_dyn
        let array_refs: Vec<ArrayRef> = arr_list
            .iter()
            .map(|a| a.get::<ArrayImpl>().inner.clone())
            .collect();

        if array_refs.len() == 1 {
            return Ok(arrays::Array::new(ArrayImpl { inner: array_refs.into_iter().next().unwrap() }));
        }

        // Use concat_elements_dyn for element-wise string concatenation
        let mut result = array_refs[0].clone();
        for arr in &array_refs[1..] {
            result = arrow_string::concat_elements::concat_elements_dyn(&*result, &**arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        }
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn concat_elements(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = arrow_string::concat_elements::concat_elements_dyn(&*left_impl.inner, &*right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn fill_null_i64(arr: arrays::ArrayBorrow<'_>, value: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let int_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))?;

        // Manual fill_null: replace null values with the given value
        let result: arrow_array::Int64Array = int_arr
            .iter()
            .map(|opt| Some(opt.unwrap_or(value)))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn fill_null_f64(arr: arrays::ArrayBorrow<'_>, value: f64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let float_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Float64 array".to_string()))?;

        // Manual fill_null
        let result: arrow_array::Float64Array = float_arr
            .iter()
            .map(|opt| Some(opt.unwrap_or(value)))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn fill_null_string(arr: arrays::ArrayBorrow<'_>, value: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected String array".to_string()))?;

        // Manual fill_null
        let result: arrow_array::StringArray = string_arr
            .iter()
            .map(|opt| Some(opt.unwrap_or(&value).to_string()))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn fill_null_bool(arr: arrays::ArrayBorrow<'_>, value: bool) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bool_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Boolean array".to_string()))?;

        // Manual fill_null
        let result: arrow_array::BooleanArray = bool_arr
            .iter()
            .map(|opt| Some(opt.unwrap_or(value)))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn coalesce(arr_list: Vec<arrays::Array>) -> Result<arrays::Array, compute::ArrowError> {
        if arr_list.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("No arrays to coalesce".to_string()));
        }
        if arr_list.len() == 1 {
            let arr_impl = arr_list[0].get::<ArrayImpl>();
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        // For simplicity, implement coalesce for the first two arrays
        // A full implementation would iterate through all arrays
        let first = arr_list[0].get::<ArrayImpl>();
        let second = arr_list[1].get::<ArrayImpl>();

        let len = first.inner.len();
        let mut indices = Vec::with_capacity(len);
        let mut sources = Vec::with_capacity(len);

        for i in 0..len {
            if first.inner.is_valid(i) {
                indices.push(i as u32);
                sources.push(0u32);
            } else if second.inner.is_valid(i) {
                indices.push(i as u32);
                sources.push(1u32);
            } else {
                indices.push(i as u32);
                sources.push(0u32); // Take from first (will be null)
            }
        }

        // Build result by interleaving
        let result = arrow_select::interleave::interleave(
            &[&*first.inner, &*second.inner],
            &sources.iter().zip(indices.iter()).map(|(&s, &i)| (s as usize, i as usize)).collect::<Vec<_>>(),
        ).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn unique(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_ord::cmp::eq;
        let arr_impl = arr.get::<ArrayImpl>();

        // Build unique values by iterating and checking distinctness
        let len = arr_impl.inner.len();
        if len == 0 {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(0, 0) }));
        }

        // Use take with unique indices
        let mut unique_indices: Vec<u32> = Vec::new();

        for i in 0..len {
            let mut is_unique = true;
            for &idx in &unique_indices {
                // Check if current element equals any previous unique element
                let current = arr_impl.inner.slice(i, 1);
                let prev = arr_impl.inner.slice(idx as usize, 1);
                if let Ok(eq_result) = eq(&current, &prev) {
                    if eq_result.len() > 0 && eq_result.value(0) {
                        is_unique = false;
                        break;
                    }
                }
            }
            if is_unique {
                unique_indices.push(i as u32);
            }
        }

        let indices = arrow_array::UInt32Array::from(unique_indices);
        let result = arrow_select::take::take(&*arr_impl.inner, &indices, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn value_counts(arr: arrays::ArrayBorrow<'_>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        use arrow_ord::cmp::eq;
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        if len == 0 {
            let schema = arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("values", arr_impl.inner.data_type().clone(), true),
                arrow_schema::Field::new("counts", arrow_schema::DataType::UInt64, false),
            ]);
            let batch = ArrowRecordBatch::try_new(
                Arc::new(schema),
                vec![
                    arr_impl.inner.slice(0, 0),
                    Arc::new(arrow_array::UInt64Array::from(Vec::<u64>::new())),
                ],
            ).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }));
        }

        // Get unique values and their counts
        let mut unique_indices: Vec<usize> = Vec::new();
        let mut counts: Vec<u64> = Vec::new();

        for i in 0..len {
            let mut found_idx: Option<usize> = None;
            for (j, &idx) in unique_indices.iter().enumerate() {
                let current = arr_impl.inner.slice(i, 1);
                let prev = arr_impl.inner.slice(idx, 1);
                if let Ok(eq_result) = eq(&current, &prev) {
                    if eq_result.len() > 0 && eq_result.value(0) {
                        found_idx = Some(j);
                        break;
                    }
                }
            }
            if let Some(j) = found_idx {
                counts[j] += 1;
            } else {
                unique_indices.push(i);
                counts.push(1);
            }
        }

        let indices = arrow_array::UInt32Array::from(unique_indices.iter().map(|&i| i as u32).collect::<Vec<_>>());
        let values = arrow_select::take::take(&*arr_impl.inner, &indices, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        let counts_arr = Arc::new(arrow_array::UInt64Array::from(counts));

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("values", arr_impl.inner.data_type().clone(), true),
            arrow_schema::Field::new("counts", arrow_schema::DataType::UInt64, false),
        ]);

        let batch = ArrowRecordBatch::try_new(
            Arc::new(schema),
            vec![values, counts_arr],
        ).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }))
    }

    // ========== Date/Time Operations ==========

    fn date_year(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Year)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_month(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Month)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_day(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Day)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_day_of_week(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::DayOfWeekSunday0)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_day_of_year(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::DayOfYear)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_week(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Week)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_quarter(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Quarter)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn time_hour(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Hour)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn time_minute(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Minute)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn time_second(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Second)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn time_millisecond(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Millisecond)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn time_microsecond(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Microsecond)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn time_nanosecond(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_arith::temporal::date_part(&*arr_impl.inner, arrow_arith::temporal::DatePart::Nanosecond)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn date_add_days(arr: arrays::ArrayBorrow<'_>, days: i32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Handle Date32 (days since epoch)
        if let Some(date32_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>() {
            let result: arrow_array::Date32Array = date32_arr.iter()
                .map(|opt| opt.map(|v| v + days))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Handle Date64 (milliseconds since epoch)
        if let Some(date64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>() {
            let ms_per_day: i64 = 86_400_000;
            let result: arrow_array::Date64Array = date64_arr.iter()
                .map(|opt| opt.map(|v| v + (days as i64) * ms_per_day))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Date32 or Date64 array".to_string()))
    }

    fn date_add_months(arr: arrays::ArrayBorrow<'_>, months: i32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Handle Date32 (days since epoch)
        if let Some(date32_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>() {
            let result: arrow_array::Date32Array = date32_arr.iter()
                .map(|opt| {
                    opt.map(|days_since_epoch| {
                        // Convert to date components
                        let epoch = 719_163; // Days from year 0 to 1970-01-01
                        let total_days = days_since_epoch + epoch;

                        // Calculate year, month, day from total days
                        let (mut year, mut month, day) = days_to_ymd(total_days);

                        // Add months
                        let total_months = (year * 12 + (month as i32 - 1)) + months;
                        year = total_months / 12;
                        month = ((total_months % 12) + 12) % 12 + 1;
                        if total_months < 0 && total_months % 12 != 0 {
                            year -= 1;
                        }

                        // Clamp day to valid range for new month
                        let max_day = days_in_month(year, month as u32);
                        let clamped_day = day.min(max_day);

                        // Convert back to days since epoch
                        ymd_to_days(year, month as u32, clamped_day) - epoch
                    })
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Handle Date64 (milliseconds since epoch)
        if let Some(date64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>() {
            let ms_per_day: i64 = 86_400_000;
            let epoch = 719_163i64;
            let result: arrow_array::Date64Array = date64_arr.iter()
                .map(|opt| {
                    opt.map(|ms| {
                        let days_since_epoch = ms / ms_per_day;
                        let remainder_ms = ms % ms_per_day;
                        let total_days = days_since_epoch + epoch;

                        let (mut year, mut month, day) = days_to_ymd(total_days as i32);

                        let total_months = (year * 12 + (month as i32 - 1)) + months;
                        year = total_months / 12;
                        month = ((total_months % 12) + 12) % 12 + 1;
                        if total_months < 0 && total_months % 12 != 0 {
                            year -= 1;
                        }

                        let max_day = days_in_month(year, month as u32);
                        let clamped_day = day.min(max_day);

                        let new_days = ymd_to_days(year, month as u32, clamped_day) as i64 - epoch;
                        new_days * ms_per_day + remainder_ms
                    })
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Date32 or Date64 array".to_string()))
    }

    fn date_add_years(arr: arrays::ArrayBorrow<'_>, years: i32) -> Result<arrays::Array, compute::ArrowError> {
        // Add years by converting to months
        Self::date_add_months(arr, years * 12)
    }

    fn date_diff_days(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        // Handle Date32
        if let (Some(left_arr), Some(right_arr)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>(),
        ) {
            let result: arrow_array::Int32Array = left_arr.iter()
                .zip(right_arr.iter())
                .map(|(l, r)| match (l, r) {
                    (Some(l), Some(r)) => Some(l - r),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Handle Date64
        if let (Some(left_arr), Some(right_arr)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>(),
        ) {
            let ms_per_day: i64 = 86_400_000;
            let result: arrow_array::Int64Array = left_arr.iter()
                .zip(right_arr.iter())
                .map(|(l, r)| match (l, r) {
                    (Some(l), Some(r)) => Some((l - r) / ms_per_day),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Date32 or Date64 arrays of the same type".to_string()))
    }

    fn timestamp_truncate(arr: arrays::ArrayBorrow<'_>, unit: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Try various timestamp types
        macro_rules! handle_timestamp {
            ($arr_type:ty, $time_unit:expr) => {
                if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let divisor: i64 = match (unit.to_lowercase().as_str(), $time_unit) {
                        ("second", arrow_schema::TimeUnit::Nanosecond) => 1_000_000_000,
                        ("second", arrow_schema::TimeUnit::Microsecond) => 1_000_000,
                        ("second", arrow_schema::TimeUnit::Millisecond) => 1_000,
                        ("second", arrow_schema::TimeUnit::Second) => 1,
                        ("minute", arrow_schema::TimeUnit::Nanosecond) => 60_000_000_000,
                        ("minute", arrow_schema::TimeUnit::Microsecond) => 60_000_000,
                        ("minute", arrow_schema::TimeUnit::Millisecond) => 60_000,
                        ("minute", arrow_schema::TimeUnit::Second) => 60,
                        ("hour", arrow_schema::TimeUnit::Nanosecond) => 3_600_000_000_000,
                        ("hour", arrow_schema::TimeUnit::Microsecond) => 3_600_000_000,
                        ("hour", arrow_schema::TimeUnit::Millisecond) => 3_600_000,
                        ("hour", arrow_schema::TimeUnit::Second) => 3_600,
                        ("day", arrow_schema::TimeUnit::Nanosecond) => 86_400_000_000_000,
                        ("day", arrow_schema::TimeUnit::Microsecond) => 86_400_000_000,
                        ("day", arrow_schema::TimeUnit::Millisecond) => 86_400_000,
                        ("day", arrow_schema::TimeUnit::Second) => 86_400,
                        _ => return Err(compute::ArrowError::InvalidArgument(
                            format!("Unsupported truncation unit: {}", unit)
                        )),
                    };
                    let result: $arr_type = ts_arr.iter()
                        .map(|opt| opt.map(|v| (v / divisor) * divisor))
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        handle_timestamp!(arrow_array::TimestampNanosecondArray, arrow_schema::TimeUnit::Nanosecond);
        handle_timestamp!(arrow_array::TimestampMicrosecondArray, arrow_schema::TimeUnit::Microsecond);
        handle_timestamp!(arrow_array::TimestampMillisecondArray, arrow_schema::TimeUnit::Millisecond);
        handle_timestamp!(arrow_array::TimestampSecondArray, arrow_schema::TimeUnit::Second);

        Err(compute::ArrowError::InvalidArgument("Expected Timestamp array".to_string()))
    }

    fn timestamp_convert_tz(arr: arrays::ArrayBorrow<'_>, _from_tz: Option<String>, to_tz: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // For now, we support changing the timezone metadata without converting values
        // Full timezone conversion would require a timezone database

        // Handle different timestamp precisions - convert to same type with new timezone
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampNanosecondArray>() {
            let values: Vec<Option<i64>> = ts_arr.iter().collect();
            let new_arr = arrow_array::TimestampNanosecondArray::from(values).with_timezone(to_tz);
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(new_arr) }));
        }
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>() {
            let values: Vec<Option<i64>> = ts_arr.iter().collect();
            let new_arr = arrow_array::TimestampMicrosecondArray::from(values).with_timezone(to_tz);
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(new_arr) }));
        }
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMillisecondArray>() {
            let values: Vec<Option<i64>> = ts_arr.iter().collect();
            let new_arr = arrow_array::TimestampMillisecondArray::from(values).with_timezone(to_tz);
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(new_arr) }));
        }
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampSecondArray>() {
            let values: Vec<Option<i64>> = ts_arr.iter().collect();
            let new_arr = arrow_array::TimestampSecondArray::from(values).with_timezone(to_tz);
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(new_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Timestamp array".to_string()))
    }

    fn timestamp_epoch_seconds(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! to_epoch_seconds {
            ($arr_type:ty, $divisor:expr) => {
                if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let result: arrow_array::Int64Array = ts_arr.iter()
                        .map(|opt| opt.map(|v| v / $divisor))
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        to_epoch_seconds!(arrow_array::TimestampNanosecondArray, 1_000_000_000i64);
        to_epoch_seconds!(arrow_array::TimestampMicrosecondArray, 1_000_000i64);
        to_epoch_seconds!(arrow_array::TimestampMillisecondArray, 1_000i64);
        to_epoch_seconds!(arrow_array::TimestampSecondArray, 1i64);

        Err(compute::ArrowError::InvalidArgument("Expected Timestamp array".to_string()))
    }

    fn timestamp_epoch_millis(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! to_epoch_millis {
            ($arr_type:ty, $divisor:expr) => {
                if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let result: arrow_array::Int64Array = ts_arr.iter()
                        .map(|opt| opt.map(|v| v / $divisor))
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        to_epoch_millis!(arrow_array::TimestampNanosecondArray, 1_000_000i64);
        to_epoch_millis!(arrow_array::TimestampMicrosecondArray, 1_000i64);
        to_epoch_millis!(arrow_array::TimestampMillisecondArray, 1i64);

        // For seconds, multiply to get milliseconds
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampSecondArray>() {
            let result: arrow_array::Int64Array = ts_arr.iter()
                .map(|opt| opt.map(|v| v * 1_000))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Timestamp array".to_string()))
    }

    fn timestamp_from_epoch_seconds(arr: arrays::ArrayBorrow<'_>, timezone: Option<String>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let values: Vec<Option<i64>> = int_arr.iter().collect();
            let result = match timezone {
                Some(tz) => arrow_array::TimestampSecondArray::from(values).with_timezone(tz),
                None => arrow_array::TimestampSecondArray::from(values),
            };
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))
    }

    fn timestamp_from_epoch_millis(arr: arrays::ArrayBorrow<'_>, timezone: Option<String>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let values: Vec<Option<i64>> = int_arr.iter().collect();
            let result = match timezone {
                Some(tz) => arrow_array::TimestampMillisecondArray::from(values).with_timezone(tz),
                None => arrow_array::TimestampMillisecondArray::from(values),
            };
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Int64 array".to_string()))
    }

    fn date_is_weekend(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Helper to convert days since epoch to day of week (0=Sunday, 6=Saturday)
        fn days_to_dow(days: i32) -> u32 {
            // Unix epoch (1970-01-01) was Thursday (4)
            ((days + 4) % 7).unsigned_abs()
        }

        if let Some(date_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>() {
            let result: arrow_array::BooleanArray = date_arr.iter()
                .map(|opt| opt.map(|days| {
                    let dow = days_to_dow(days);
                    dow == 0 || dow == 6  // Sunday or Saturday
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(date_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>() {
            let result: arrow_array::BooleanArray = date_arr.iter()
                .map(|opt| opt.map(|millis| {
                    let days = (millis / 86_400_000) as i32;
                    let dow = days_to_dow(days);
                    dow == 0 || dow == 6
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Date32 or Date64 array".to_string()))
    }

    fn date_is_leap_year(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Helper to check if year is leap year
        fn is_leap_year(year: i32) -> bool {
            (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
        }

        // Helper to get year from days since epoch
        fn days_to_year(days: i32) -> i32 {
            // Approximate - more accurate would use a proper date library
            let days_since_year_zero = days + 719_528; // Days from year 0 to 1970
            (days_since_year_zero as f64 / 365.2425) as i32
        }

        if let Some(date_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>() {
            let result: arrow_array::BooleanArray = date_arr.iter()
                .map(|opt| opt.map(|days| is_leap_year(days_to_year(days))))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(date_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>() {
            let result: arrow_array::BooleanArray = date_arr.iter()
                .map(|opt| opt.map(|millis| {
                    let days = (millis / 86_400_000) as i32;
                    is_leap_year(days_to_year(days))
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Date32 or Date64 array".to_string()))
    }

    fn date_days_in_month(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Days in each month (non-leap year)
        const DAYS_IN_MONTH: [i32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        fn is_leap_year(year: i32) -> bool {
            (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
        }

        // Extract year and month from days since epoch
        fn days_to_year_month(days: i32) -> (i32, u32) {
            let mut remaining_days = days + 719_528; // Days from year 0 to 1970
            let mut year = (remaining_days as f64 / 365.2425) as i32;

            // Calculate days at start of year
            fn days_in_year(y: i32) -> i32 {
                y * 365 + y / 4 - y / 100 + y / 400
            }

            let days_at_year_start = days_in_year(year);
            remaining_days = remaining_days - days_at_year_start;

            // Adjust if we overshot
            while remaining_days < 0 {
                year -= 1;
                remaining_days += if is_leap_year(year) { 366 } else { 365 };
            }

            // Find month
            let is_leap = is_leap_year(year);
            let mut month = 0u32;
            for m in 0..12 {
                let days_in_month = if m == 1 && is_leap { 29 } else { DAYS_IN_MONTH[m as usize] };
                if remaining_days < days_in_month {
                    month = m + 1;
                    break;
                }
                remaining_days -= days_in_month;
                if m == 11 { month = 12; }
            }

            (year, month)
        }

        fn get_days_in_month(year: i32, month: u32) -> i32 {
            let is_leap = is_leap_year(year);
            let idx = (month.saturating_sub(1)) as usize;
            if idx < 12 {
                if idx == 1 && is_leap { 29 } else { DAYS_IN_MONTH[idx] }
            } else {
                31
            }
        }

        if let Some(date_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>() {
            let result: arrow_array::Int32Array = date_arr.iter()
                .map(|opt| opt.map(|days| {
                    let (year, month) = days_to_year_month(days);
                    get_days_in_month(year, month)
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(date_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>() {
            let result: arrow_array::Int32Array = date_arr.iter()
                .map(|opt| opt.map(|millis| {
                    let days = (millis / 86_400_000) as i32;
                    let (year, month) = days_to_year_month(days);
                    get_days_in_month(year, month)
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Date32 or Date64 array".to_string()))
    }

    fn timestamp_add_interval(arr: arrays::ArrayBorrow<'_>, months: i32, days: i32, nanos: i64) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Convert interval to nanoseconds (for time portion) and days (for date portion)
        let days_to_add = days + (months * 30); // Approximate months to days
        let nanos_per_day: i64 = 86_400_000_000_000;
        let total_nanos = (days_to_add as i64) * nanos_per_day + nanos;

        macro_rules! add_interval_impl {
            ($arr_type:ty, $scale:expr) => {
                if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let offset = total_nanos / $scale;
                    let result: $arr_type = ts_arr.iter()
                        .map(|opt| opt.map(|v| v + offset))
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        add_interval_impl!(arrow_array::TimestampNanosecondArray, 1i64);
        add_interval_impl!(arrow_array::TimestampMicrosecondArray, 1_000i64);
        add_interval_impl!(arrow_array::TimestampMillisecondArray, 1_000_000i64);
        add_interval_impl!(arrow_array::TimestampSecondArray, 1_000_000_000i64);

        Err(compute::ArrowError::InvalidArgument("Expected Timestamp array".to_string()))
    }

    fn timestamp_diff(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>, unit: types::TimeUnit) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        // Calculate divisor based on target unit
        let divisor: i64 = match unit {
            types::TimeUnit::Second => 1_000_000_000,
            types::TimeUnit::Millisecond => 1_000_000,
            types::TimeUnit::Microsecond => 1_000,
            types::TimeUnit::Nanosecond => 1,
        };

        macro_rules! diff_impl {
            ($arr_type:ty, $scale:expr) => {
                if let (Some(l), Some(r)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    let result: arrow_array::Int64Array = l.iter().zip(r.iter())
                        .map(|(lv, rv)| {
                            match (lv, rv) {
                                (Some(l), Some(r)) => Some(((l - r) * $scale) / divisor),
                                _ => None,
                            }
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        diff_impl!(arrow_array::TimestampNanosecondArray, 1i64);
        diff_impl!(arrow_array::TimestampMicrosecondArray, 1_000i64);
        diff_impl!(arrow_array::TimestampMillisecondArray, 1_000_000i64);
        diff_impl!(arrow_array::TimestampSecondArray, 1_000_000_000i64);

        Err(compute::ArrowError::InvalidArgument("Expected matching Timestamp arrays".to_string()))
    }

    fn make_date(year: arrays::ArrayBorrow<'_>, month: arrays::ArrayBorrow<'_>, day: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let year_impl = year.get::<ArrayImpl>();
        let month_impl = month.get::<ArrayImpl>();
        let day_impl = day.get::<ArrayImpl>();

        let year_arr = year_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("year must be Int32 array".to_string()))?;
        let month_arr = month_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("month must be Int32 array".to_string()))?;
        let day_arr = day_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("day must be Int32 array".to_string()))?;

        // Convert year/month/day to days since epoch
        fn ymd_to_days(year: i32, month: i32, day: i32) -> i32 {
            // Adjust for months out of range
            let (y, m) = if month <= 0 {
                let years_sub = (-month) / 12 + 1;
                (year - years_sub, month + years_sub * 12)
            } else if month > 12 {
                let years_add = (month - 1) / 12;
                (year + years_add, month - years_add * 12)
            } else {
                (year, month)
            };

            // Days from year 1 to year y
            let days_before_year = (y - 1) * 365 + (y - 1) / 4 - (y - 1) / 100 + (y - 1) / 400;

            // Days in months before current month
            const DAYS_BEFORE_MONTH: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
            let is_leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
            let month_idx = ((m - 1).max(0).min(11)) as usize;
            let days_before_month = DAYS_BEFORE_MONTH[month_idx] + if is_leap && m > 2 { 1 } else { 0 };

            // Total days minus Unix epoch offset
            days_before_year + days_before_month + day - 719_529
        }

        let result: arrow_array::Date32Array = year_arr.iter()
            .zip(month_arr.iter())
            .zip(day_arr.iter())
            .map(|((y, m), d)| {
                match (y, m, d) {
                    (Some(year), Some(month), Some(day)) => Some(ymd_to_days(year, month, day)),
                    _ => None,
                }
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn make_timestamp(
        year: arrays::ArrayBorrow<'_>,
        month: arrays::ArrayBorrow<'_>,
        day: arrays::ArrayBorrow<'_>,
        hour: arrays::ArrayBorrow<'_>,
        minute: arrays::ArrayBorrow<'_>,
        second: arrays::ArrayBorrow<'_>,
        timezone: Option<String>,
    ) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let year_impl = year.get::<ArrayImpl>();
        let month_impl = month.get::<ArrayImpl>();
        let day_impl = day.get::<ArrayImpl>();
        let hour_impl = hour.get::<ArrayImpl>();
        let minute_impl = minute.get::<ArrayImpl>();
        let second_impl = second.get::<ArrayImpl>();

        let year_arr = year_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("year must be Int32 array".to_string()))?;
        let month_arr = month_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("month must be Int32 array".to_string()))?;
        let day_arr = day_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("day must be Int32 array".to_string()))?;
        let hour_arr = hour_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("hour must be Int32 array".to_string()))?;
        let minute_arr = minute_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("minute must be Int32 array".to_string()))?;
        let second_arr = second_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("second must be Int32 array".to_string()))?;

        // Convert to epoch seconds
        fn ymdhms_to_epoch(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: i32) -> i64 {
            // Days from year 1 to year y
            let (y, m) = if month <= 0 {
                let years_sub = (-month) / 12 + 1;
                (year - years_sub, month + years_sub * 12)
            } else if month > 12 {
                let years_add = (month - 1) / 12;
                (year + years_add, month - years_add * 12)
            } else {
                (year, month)
            };

            let days_before_year = (y - 1) * 365 + (y - 1) / 4 - (y - 1) / 100 + (y - 1) / 400;
            const DAYS_BEFORE_MONTH: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
            let is_leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
            let month_idx = ((m - 1).max(0).min(11)) as usize;
            let days_before_month = DAYS_BEFORE_MONTH[month_idx] + if is_leap && m > 2 { 1 } else { 0 };
            let total_days = days_before_year + days_before_month + day - 719_529;

            (total_days as i64) * 86_400 + (hour as i64) * 3600 + (minute as i64) * 60 + (second as i64)
        }

        let values: Vec<Option<i64>> = year_arr.iter()
            .zip(month_arr.iter())
            .zip(day_arr.iter())
            .zip(hour_arr.iter())
            .zip(minute_arr.iter())
            .zip(second_arr.iter())
            .map(|(((((y, mo), d), h), mi), s)| {
                match (y, mo, d, h, mi, s) {
                    (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) =>
                        Some(ymdhms_to_epoch(year, month, day, hour, minute, second)),
                    _ => None,
                }
            })
            .collect();

        let result = match timezone {
            Some(tz) => arrow_array::TimestampSecondArray::from(values).with_timezone(tz),
            None => arrow_array::TimestampSecondArray::from(values),
        };

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    // ========== Interval Operations ==========

    fn make_interval_month_day_nano(months: arrays::ArrayBorrow<'_>, days: arrays::ArrayBorrow<'_>, nanos: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let months_impl = months.get::<ArrayImpl>();
        let days_impl = days.get::<ArrayImpl>();
        let nanos_impl = nanos.get::<ArrayImpl>();

        let months_arr = months_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("months must be Int32 array".to_string()))?;
        let days_arr = days_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("days must be Int32 array".to_string()))?;
        let nanos_arr = nanos_impl.inner.as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("nanos must be Int64 array".to_string()))?;

        if months_arr.len() != days_arr.len() || days_arr.len() != nanos_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("All arrays must have the same length".to_string()));
        }

        let mut builder = arrow_array::builder::IntervalMonthDayNanoBuilder::with_capacity(months_arr.len());

        for i in 0..months_arr.len() {
            if months_arr.is_null(i) || days_arr.is_null(i) || nanos_arr.is_null(i) {
                builder.append_null();
            } else {
                let interval = arrow_buffer::IntervalMonthDayNano::new(
                    months_arr.value(i),
                    days_arr.value(i),
                    nanos_arr.value(i),
                );
                builder.append_value(interval);
            }
        }

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn interval_months(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>() {
            let result: arrow_array::Int32Array = (0..interval_arr.len())
                .map(|i| {
                    if interval_arr.is_null(i) {
                        None
                    } else {
                        Some(interval_arr.value(i).months)
                    }
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalYearMonthArray>() {
            let result: arrow_array::Int32Array = interval_arr.iter()
                .map(|opt| opt)
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Interval array".to_string()))
    }

    fn interval_days(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>() {
            let result: arrow_array::Int32Array = (0..interval_arr.len())
                .map(|i| {
                    if interval_arr.is_null(i) {
                        None
                    } else {
                        Some(interval_arr.value(i).days)
                    }
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalDayTimeArray>() {
            let result: arrow_array::Int32Array = interval_arr.iter()
                .map(|opt| opt.map(|v| v.days))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Interval array".to_string()))
    }

    fn interval_nanos(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>() {
            let result: arrow_array::Int64Array = (0..interval_arr.len())
                .map(|i| {
                    if interval_arr.is_null(i) {
                        None
                    } else {
                        Some(interval_arr.value(i).nanoseconds)
                    }
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalDayTimeArray>() {
            // Convert milliseconds to nanoseconds
            let result: arrow_array::Int64Array = interval_arr.iter()
                .map(|opt| opt.map(|v| (v.milliseconds as i64) * 1_000_000))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected Interval array".to_string()))
    }

    // ========== Regex Operations ==========

    fn regex_match(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let result = arrow_string::regexp::regexp_is_match_scalar(string_arr, &pattern, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn regex_extract(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let re = regex::Regex::new(&pattern)
            .map_err(|e| compute::ArrowError::InvalidArgument(format!("Invalid regex: {}", e)))?;

        // Extract first capture group or full match
        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| {
                opt.and_then(|s| {
                    re.captures(s).and_then(|caps| {
                        // Return first capture group if it exists, otherwise full match
                        caps.get(1).or_else(|| caps.get(0)).map(|m| m.as_str().to_string())
                    })
                })
            })
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn regex_extract_group(arr: arrays::ArrayBorrow<'_>, pattern: String, group: u32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let re = regex::Regex::new(&pattern)
            .map_err(|e| compute::ArrowError::InvalidArgument(format!("Invalid regex: {}", e)))?;

        let group_idx = group as usize;
        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| {
                opt.and_then(|s| {
                    re.captures(s).and_then(|caps| {
                        caps.get(group_idx).map(|m| m.as_str().to_string())
                    })
                })
            })
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn regex_replace(arr: arrays::ArrayBorrow<'_>, pattern: String, replacement: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let re = regex::Regex::new(&pattern)
            .map_err(|e| compute::ArrowError::InvalidArgument(format!("Invalid regex: {}", e)))?;

        // Replace first occurrence only
        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| re.replace(s, &replacement).into_owned()))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn regex_replace_all(arr: arrays::ArrayBorrow<'_>, pattern: String, replacement: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let re = regex::Regex::new(&pattern)
            .map_err(|e| compute::ArrowError::InvalidArgument(format!("Invalid regex: {}", e)))?;

        // Replace all occurrences
        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| re.replace_all(s, &replacement).into_owned()))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn regex_count(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let re = regex::Regex::new(&pattern)
            .map_err(|e| compute::ArrowError::InvalidArgument(format!("Invalid regex: {}", e)))?;

        let result: arrow_array::Int64Array = string_arr.iter()
            .map(|opt| opt.map(|s| re.find_iter(s).count() as i64))
            .collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn regex_split(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let re = regex::Regex::new(&pattern)
            .map_err(|e| compute::ArrowError::InvalidArgument(format!("Invalid regex: {}", e)))?;

        // Build a ListArray where each element is a list of split strings
        let mut list_builder = arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new());

        for opt in string_arr.iter() {
            match opt {
                None => list_builder.append_null(),
                Some(s) => {
                    let parts: Vec<&str> = re.split(s).collect();
                    let values_builder = list_builder.values();
                    for part in parts {
                        values_builder.append_value(part);
                    }
                    list_builder.append(true);
                }
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_builder.finish()) }))
    }

    // ========== Base64 Operations ==========

    fn b64_encode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bin_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected Binary array".to_string()))?;
        let result = arrow_cast::base64::b64_encode(&arrow_cast::base64::BASE64_STANDARD, bin_arr);
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn b64_decode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        // b64_decode expects a BinaryArray, so we need to convert from StringArray
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            // Convert StringArray to BinaryArray
            let bin_arr: arrow_array::BinaryArray = str_arr.iter()
                .map(|opt| opt.map(|s| s.as_bytes()))
                .collect();
            let result = arrow_cast::base64::b64_decode(&arrow_cast::base64::BASE64_STANDARD, &bin_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            let result = arrow_cast::base64::b64_decode(&arrow_cast::base64::BASE64_STANDARD, bin_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::InvalidArgument("Expected String or Binary array".to_string()))
    }

    // ========== Window Functions ==========
    //
    // Window functions operate over partitions and order within partitions.
    // If partition_by is empty, the entire array is one partition.
    // If order_by is empty, rows are processed in original order.

    fn window_row_number(partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        // Get array length from first partition_by or order_by array
        let len = if !partition_by.is_empty() {
            partition_by[0].get::<ArrayImpl>().inner.len()
        } else if !order_by.is_empty() {
            order_by[0].get::<ArrayImpl>().inner.len()
        } else {
            return Err(compute::ArrowError::InvalidArgument("Need at least one array to determine length".to_string()));
        };

        // Compute partitions and sort indices
        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        // Result array
        let mut result = vec![0u64; len];

        for (start, end) in partitions {
            for (row_num, i) in (start..end).enumerate() {
                let original_idx = sort_indices[i];
                result[original_idx] = (row_num + 1) as u64;
            }
        }

        let result_arr: arrow_array::UInt64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_rank(partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let len = if !partition_by.is_empty() {
            partition_by[0].get::<ArrayImpl>().inner.len()
        } else if !order_by.is_empty() {
            order_by[0].get::<ArrayImpl>().inner.len()
        } else {
            return Err(compute::ArrowError::InvalidArgument("Need at least one array".to_string()));
        };

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;
        let order_arrays: Vec<_> = order_by.iter().map(|a| a.get::<ArrayImpl>().inner.clone()).collect();

        let mut result = vec![0u64; len];

        for (start, end) in partitions {
            let mut rank = 1u64;
            for i in start..end {
                let original_idx = sort_indices[i];
                if i > start {
                    let prev_idx = sort_indices[i - 1];
                    // Check if current row differs from previous in order_by columns
                    if !rows_equal_for_ordering(&order_arrays, original_idx, prev_idx) {
                        rank = (i - start + 1) as u64;
                    }
                }
                result[original_idx] = rank;
            }
        }

        let result_arr: arrow_array::UInt64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_dense_rank(partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let len = if !partition_by.is_empty() {
            partition_by[0].get::<ArrayImpl>().inner.len()
        } else if !order_by.is_empty() {
            order_by[0].get::<ArrayImpl>().inner.len()
        } else {
            return Err(compute::ArrowError::InvalidArgument("Need at least one array".to_string()));
        };

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;
        let order_arrays: Vec<_> = order_by.iter().map(|a| a.get::<ArrayImpl>().inner.clone()).collect();

        let mut result = vec![0u64; len];

        for (start, end) in partitions {
            let mut dense_rank = 1u64;
            for i in start..end {
                let original_idx = sort_indices[i];
                if i > start {
                    let prev_idx = sort_indices[i - 1];
                    if !rows_equal_for_ordering(&order_arrays, original_idx, prev_idx) {
                        dense_rank += 1;
                    }
                }
                result[original_idx] = dense_rank;
            }
        }

        let result_arr: arrow_array::UInt64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_percent_rank(partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let len = if !partition_by.is_empty() {
            partition_by[0].get::<ArrayImpl>().inner.len()
        } else if !order_by.is_empty() {
            order_by[0].get::<ArrayImpl>().inner.len()
        } else {
            return Err(compute::ArrowError::InvalidArgument("Need at least one array".to_string()));
        };

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;
        let order_arrays: Vec<_> = order_by.iter().map(|a| a.get::<ArrayImpl>().inner.clone()).collect();

        let mut result = vec![0.0f64; len];

        for (start, end) in partitions {
            let partition_size = end - start;
            if partition_size <= 1 {
                // percent_rank = 0 for single row partition
                for i in start..end {
                    result[sort_indices[i]] = 0.0;
                }
                continue;
            }

            let mut rank = 1u64;
            for i in start..end {
                let original_idx = sort_indices[i];
                if i > start {
                    let prev_idx = sort_indices[i - 1];
                    if !rows_equal_for_ordering(&order_arrays, original_idx, prev_idx) {
                        rank = (i - start + 1) as u64;
                    }
                }
                // percent_rank = (rank - 1) / (partition_size - 1)
                result[original_idx] = (rank - 1) as f64 / (partition_size - 1) as f64;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_cume_dist(partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let len = if !partition_by.is_empty() {
            partition_by[0].get::<ArrayImpl>().inner.len()
        } else if !order_by.is_empty() {
            order_by[0].get::<ArrayImpl>().inner.len()
        } else {
            return Err(compute::ArrowError::InvalidArgument("Need at least one array".to_string()));
        };

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;
        let order_arrays: Vec<_> = order_by.iter().map(|a| a.get::<ArrayImpl>().inner.clone()).collect();

        let mut result = vec![0.0f64; len];

        for (start, end) in partitions {
            let partition_size = end - start;
            // cume_dist = number of rows with value <= current / partition_size
            // Since we're sorted, count rows up to and including current group

            let mut i = start;
            while i < end {
                let current_idx = sort_indices[i];
                // Find end of current group (same order_by values)
                let mut group_end = i + 1;
                while group_end < end {
                    let next_idx = sort_indices[group_end];
                    if !rows_equal_for_ordering(&order_arrays, current_idx, next_idx) {
                        break;
                    }
                    group_end += 1;
                }

                let cume = group_end - start;
                let cume_dist = cume as f64 / partition_size as f64;

                // Assign same cume_dist to all rows in this group
                for j in i..group_end {
                    result[sort_indices[j]] = cume_dist;
                }
                i = group_end;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_ntile(partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, n: u32) -> Result<arrays::Array, compute::ArrowError> {
        if n == 0 {
            return Err(compute::ArrowError::InvalidArgument("ntile n must be > 0".to_string()));
        }

        let len = if !partition_by.is_empty() {
            partition_by[0].get::<ArrayImpl>().inner.len()
        } else if !order_by.is_empty() {
            order_by[0].get::<ArrayImpl>().inner.len()
        } else {
            return Err(compute::ArrowError::InvalidArgument("Need at least one array".to_string()));
        };

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![0u32; len];

        for (start, end) in partitions {
            let partition_size = end - start;
            let n = n as usize;
            let bucket_size = partition_size / n;
            let remainder = partition_size % n;

            let mut bucket = 1u32;
            let mut count_in_bucket = 0usize;

            for i in start..end {
                result[sort_indices[i]] = bucket;
                count_in_bucket += 1;
                let current_bucket_size = bucket_size + if (bucket as usize) <= remainder { 1 } else { 0 };
                if count_in_bucket >= current_bucket_size && bucket < n as u32 {
                    bucket += 1;
                    count_in_bucket = 0;
                }
            }
        }

        let result_arr: arrow_array::UInt32Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_lead(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, offset: u32, default_value: Option<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        // Special handling for Int64Array to support default_value
        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![default_value; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    let target = i + offset as usize;
                    let original_idx = sort_indices[i];
                    if target < *end {
                        let lead_idx = sort_indices[target];
                        result[original_idx] = get_i64_opt(typed_arr, lead_idx);
                    }
                    // else: result[original_idx] already has default_value
                }
            }
            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        // Macro for other types (no default value support)
        macro_rules! impl_lead {
            ($arr_type:ty, $result_type:ty, $get_fn:ident) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Vec<Option<$result_type>> = vec![None; len];
                    for (start, end) in &partitions {
                        for i in *start..*end {
                            let target = i + offset as usize;
                            if target < *end {
                                let original_idx = sort_indices[i];
                                let lead_idx = sort_indices[target];
                                result[original_idx] = $get_fn(typed_arr, lead_idx);
                            }
                        }
                    }
                    let result_arr: $arr_type = result.into_iter().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
                }
            };
        }

        // Integer types (except Int64 which is handled above)
        impl_lead!(arrow_array::Int32Array, i32, get_i32_opt);
        impl_lead!(arrow_array::Int16Array, i16, get_i16_opt);
        impl_lead!(arrow_array::Int8Array, i8, get_i8_opt);
        impl_lead!(arrow_array::UInt64Array, u64, get_u64_opt);
        impl_lead!(arrow_array::UInt32Array, u32, get_u32_opt);
        impl_lead!(arrow_array::UInt16Array, u16, get_u16_opt);
        impl_lead!(arrow_array::UInt8Array, u8, get_u8_opt);

        // Float types
        impl_lead!(arrow_array::Float64Array, f64, get_f64_opt);
        impl_lead!(arrow_array::Float32Array, f32, get_f32_opt);

        // Boolean type
        impl_lead!(arrow_array::BooleanArray, bool, get_bool_opt);

        // String type
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut result: Vec<Option<String>> = vec![None; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    let target = i + offset as usize;
                    if target < *end {
                        let original_idx = sort_indices[i];
                        let lead_idx = sort_indices[target];
                        result[original_idx] = get_string_opt(str_arr, lead_idx);
                    }
                }
            }
            let result_arr: arrow_array::StringArray = result.iter().map(|s| s.as_deref()).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_lead: unsupported array type".to_string()))
    }

    fn window_lag(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, offset: u32, default_value: Option<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        // Special handling for Int64Array to support default_value
        if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![default_value; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    let original_idx = sort_indices[i];
                    if i >= *start + offset as usize {
                        let target = i - offset as usize;
                        let lag_idx = sort_indices[target];
                        result[original_idx] = get_i64_opt(typed_arr, lag_idx);
                    }
                    // else: result[original_idx] already has default_value
                }
            }
            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        // Macro for other types (no default value support)
        macro_rules! impl_lag {
            ($arr_type:ty, $result_type:ty, $get_fn:ident) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Vec<Option<$result_type>> = vec![None; len];
                    for (start, end) in &partitions {
                        for i in *start..*end {
                            if i >= *start + offset as usize {
                                let target = i - offset as usize;
                                let original_idx = sort_indices[i];
                                let lag_idx = sort_indices[target];
                                result[original_idx] = $get_fn(typed_arr, lag_idx);
                            }
                        }
                    }
                    let result_arr: $arr_type = result.into_iter().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
                }
            };
        }

        // Integer types (except Int64 which is handled above)
        impl_lag!(arrow_array::Int32Array, i32, get_i32_opt);
        impl_lag!(arrow_array::Int16Array, i16, get_i16_opt);
        impl_lag!(arrow_array::Int8Array, i8, get_i8_opt);
        impl_lag!(arrow_array::UInt64Array, u64, get_u64_opt);
        impl_lag!(arrow_array::UInt32Array, u32, get_u32_opt);
        impl_lag!(arrow_array::UInt16Array, u16, get_u16_opt);
        impl_lag!(arrow_array::UInt8Array, u8, get_u8_opt);

        // Float types
        impl_lag!(arrow_array::Float64Array, f64, get_f64_opt);
        impl_lag!(arrow_array::Float32Array, f32, get_f32_opt);

        // Boolean type
        impl_lag!(arrow_array::BooleanArray, bool, get_bool_opt);

        // String type
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut result: Vec<Option<String>> = vec![None; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    if i >= *start + offset as usize {
                        let target = i - offset as usize;
                        let original_idx = sort_indices[i];
                        let lag_idx = sort_indices[target];
                        result[original_idx] = get_string_opt(str_arr, lag_idx);
                    }
                }
            }
            let result_arr: arrow_array::StringArray = result.iter().map(|s| s.as_deref()).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_lag: unsupported array type".to_string()))
    }

    fn window_first_value(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        // Macro to avoid repeating the same logic for each type
        macro_rules! impl_first_value {
            ($arr_type:ty, $result_type:ty, $get_fn:ident) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Vec<Option<$result_type>> = vec![None; len];
                    for (start, end) in &partitions {
                        for i in *start..*end {
                            let current_pos = i - *start;
                            let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, *start, *end);
                            if frame_start < frame_end {
                                let first_idx = sort_indices[frame_start];
                                result[sort_indices[i]] = $get_fn(typed_arr, first_idx);
                            }
                        }
                    }
                    let result_arr: $arr_type = result.into_iter().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
                }
            };
        }

        // Integer types
        impl_first_value!(arrow_array::Int64Array, i64, get_i64_opt);
        impl_first_value!(arrow_array::Int32Array, i32, get_i32_opt);
        impl_first_value!(arrow_array::Int16Array, i16, get_i16_opt);
        impl_first_value!(arrow_array::Int8Array, i8, get_i8_opt);
        impl_first_value!(arrow_array::UInt64Array, u64, get_u64_opt);
        impl_first_value!(arrow_array::UInt32Array, u32, get_u32_opt);
        impl_first_value!(arrow_array::UInt16Array, u16, get_u16_opt);
        impl_first_value!(arrow_array::UInt8Array, u8, get_u8_opt);

        // Float types
        impl_first_value!(arrow_array::Float64Array, f64, get_f64_opt);
        impl_first_value!(arrow_array::Float32Array, f32, get_f32_opt);

        // Boolean type
        impl_first_value!(arrow_array::BooleanArray, bool, get_bool_opt);

        // String type
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut result: Vec<Option<String>> = vec![None; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    let current_pos = i - *start;
                    let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, *start, *end);
                    if frame_start < frame_end {
                        let first_idx = sort_indices[frame_start];
                        result[sort_indices[i]] = get_string_opt(str_arr, first_idx);
                    }
                }
            }
            let result_arr: arrow_array::StringArray = result.iter().map(|s| s.as_deref()).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_first_value: unsupported array type".to_string()))
    }

    fn window_last_value(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        // Macro to avoid repeating the same logic for each type
        macro_rules! impl_last_value {
            ($arr_type:ty, $result_type:ty, $get_fn:ident) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Vec<Option<$result_type>> = vec![None; len];
                    for (start, end) in &partitions {
                        for i in *start..*end {
                            let current_pos = i - *start;
                            let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, *start, *end);
                            if frame_start < frame_end {
                                let last_idx = sort_indices[frame_end - 1];
                                result[sort_indices[i]] = $get_fn(typed_arr, last_idx);
                            }
                        }
                    }
                    let result_arr: $arr_type = result.into_iter().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
                }
            };
        }

        // Integer types
        impl_last_value!(arrow_array::Int64Array, i64, get_i64_opt);
        impl_last_value!(arrow_array::Int32Array, i32, get_i32_opt);
        impl_last_value!(arrow_array::Int16Array, i16, get_i16_opt);
        impl_last_value!(arrow_array::Int8Array, i8, get_i8_opt);
        impl_last_value!(arrow_array::UInt64Array, u64, get_u64_opt);
        impl_last_value!(arrow_array::UInt32Array, u32, get_u32_opt);
        impl_last_value!(arrow_array::UInt16Array, u16, get_u16_opt);
        impl_last_value!(arrow_array::UInt8Array, u8, get_u8_opt);

        // Float types
        impl_last_value!(arrow_array::Float64Array, f64, get_f64_opt);
        impl_last_value!(arrow_array::Float32Array, f32, get_f32_opt);

        // Boolean type
        impl_last_value!(arrow_array::BooleanArray, bool, get_bool_opt);

        // String type
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut result: Vec<Option<String>> = vec![None; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    let current_pos = i - *start;
                    let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, *start, *end);
                    if frame_start < frame_end {
                        let last_idx = sort_indices[frame_end - 1];
                        result[sort_indices[i]] = get_string_opt(str_arr, last_idx);
                    }
                }
            }
            let result_arr: arrow_array::StringArray = result.iter().map(|s| s.as_deref()).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_last_value: unsupported array type".to_string()))
    }

    fn window_nth_value(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, n: u32, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        if n == 0 {
            return Err(compute::ArrowError::InvalidArgument("nth_value n must be >= 1".to_string()));
        }

        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;
        let nth_offset = (n - 1) as usize;

        // Macro to avoid repeating the same logic for each type
        macro_rules! impl_nth_value {
            ($arr_type:ty, $result_type:ty, $get_fn:ident) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Vec<Option<$result_type>> = vec![None; len];
                    for (start, end) in &partitions {
                        for i in *start..*end {
                            let current_pos = i - *start;
                            let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, *start, *end);
                            if frame_start + nth_offset < frame_end {
                                let nth_idx = sort_indices[frame_start + nth_offset];
                                result[sort_indices[i]] = $get_fn(typed_arr, nth_idx);
                            }
                        }
                    }
                    let result_arr: $arr_type = result.into_iter().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
                }
            };
        }

        // Integer types
        impl_nth_value!(arrow_array::Int64Array, i64, get_i64_opt);
        impl_nth_value!(arrow_array::Int32Array, i32, get_i32_opt);
        impl_nth_value!(arrow_array::Int16Array, i16, get_i16_opt);
        impl_nth_value!(arrow_array::Int8Array, i8, get_i8_opt);
        impl_nth_value!(arrow_array::UInt64Array, u64, get_u64_opt);
        impl_nth_value!(arrow_array::UInt32Array, u32, get_u32_opt);
        impl_nth_value!(arrow_array::UInt16Array, u16, get_u16_opt);
        impl_nth_value!(arrow_array::UInt8Array, u8, get_u8_opt);

        // Float types
        impl_nth_value!(arrow_array::Float64Array, f64, get_f64_opt);
        impl_nth_value!(arrow_array::Float32Array, f32, get_f32_opt);

        // Boolean type
        impl_nth_value!(arrow_array::BooleanArray, bool, get_bool_opt);

        // String type
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut result: Vec<Option<String>> = vec![None; len];
            for (start, end) in &partitions {
                for i in *start..*end {
                    let current_pos = i - *start;
                    let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, *start, *end);
                    if frame_start + nth_offset < frame_end {
                        let nth_idx = sort_indices[frame_start + nth_offset];
                        result[sort_indices[i]] = get_string_opt(str_arr, nth_idx);
                    }
                }
            }
            let result_arr: arrow_array::StringArray = result.iter().map(|s| s.as_deref()).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_nth_value: unsupported array type".to_string()))
    }

    fn window_sum(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![0.0f64; len];

        for (start, end) in partitions {
            for i in start..end {
                let current_pos = i - start;
                let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, start, end);

                let mut sum = 0.0f64;
                for j in frame_start..frame_end {
                    let idx = sort_indices[j];
                    sum += values[idx];
                }

                let original_idx = sort_indices[i];
                result[original_idx] = sum;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_avg(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![f64::NAN; len];

        for (start, end) in partitions {
            for i in start..end {
                let current_pos = i - start;
                let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, start, end);

                let frame_len = frame_end - frame_start;
                if frame_len > 0 {
                    let mut sum = 0.0f64;
                    for j in frame_start..frame_end {
                        let idx = sort_indices[j];
                        sum += values[idx];
                    }
                    let original_idx = sort_indices[i];
                    result[original_idx] = sum / frame_len as f64;
                }
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_min(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![f64::NAN; len];

        for (start, end) in partitions {
            for i in start..end {
                let current_pos = i - start;
                let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, start, end);

                if frame_end > frame_start {
                    let mut min_val = f64::INFINITY;
                    for j in frame_start..frame_end {
                        let idx = sort_indices[j];
                        min_val = min_val.min(values[idx]);
                    }
                    let original_idx = sort_indices[i];
                    result[original_idx] = min_val;
                }
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_max(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![f64::NAN; len];

        for (start, end) in partitions {
            for i in start..end {
                let current_pos = i - start;
                let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, start, end);

                if frame_end > frame_start {
                    let mut max_val = f64::NEG_INFINITY;
                    for j in frame_start..frame_end {
                        let idx = sort_indices[j];
                        max_val = max_val.max(values[idx]);
                    }
                    let original_idx = sort_indices[i];
                    result[original_idx] = max_val;
                }
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_count(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![0u64; len];

        for (start, end) in partitions {
            for i in start..end {
                let current_pos = i - start;
                let (frame_start, frame_end) = compute_frame_bounds(&frame, current_pos, start, end);

                let mut count = 0u64;
                for j in frame_start..frame_end {
                    let idx = sort_indices[j];
                    if !arr_impl.inner.is_null(idx) {
                        count += 1;
                    }
                }
                let original_idx = sort_indices[i];
                result[original_idx] = count;
            }
        }

        let result_arr: arrow_array::UInt64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    // ========== Additional Aggregations ==========

    fn variance(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        // Sample variance: sum((x - mean)^2) / (n - 1)
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        if values.len() < 2 {
            return Ok(None);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Ok(Some(variance))
    }

    fn variance_pop(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        // Population variance: sum((x - mean)^2) / n
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        if values.is_empty() {
            return Ok(None);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Ok(Some(variance))
    }

    fn stddev(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        // Sample standard deviation: sqrt(variance)
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        if values.len() < 2 {
            return Ok(None);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Ok(Some(variance.sqrt()))
    }

    fn stddev_pop(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        // Population standard deviation: sqrt(variance_pop)
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        if values.is_empty() {
            return Ok(None);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        Ok(Some(variance.sqrt()))
    }

    fn median(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let mut values = collect_f64_values(&arr_impl.inner)?;
        if values.is_empty() {
            return Ok(None);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len();
        if n % 2 == 0 {
            Ok(Some((values[n / 2 - 1] + values[n / 2]) / 2.0))
        } else {
            Ok(Some(values[n / 2]))
        }
    }

    fn percentile(arr: arrays::ArrayBorrow<'_>, percentile: f64) -> Result<Option<f64>, compute::ArrowError> {
        if percentile < 0.0 || percentile > 100.0 {
            return Err(compute::ArrowError::InvalidArgument("Percentile must be between 0 and 100".to_string()));
        }
        let arr_impl = arr.get::<ArrayImpl>();
        let mut values = collect_f64_values(&arr_impl.inner)?;
        if values.is_empty() {
            return Ok(None);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len();
        let index = (percentile / 100.0) * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        if lower == upper || upper >= n {
            Ok(Some(values[lower.min(n - 1)]))
        } else {
            let weight = index - lower as f64;
            Ok(Some(values[lower] * (1.0 - weight) + values[upper] * weight))
        }
    }

    fn bool_any(arr: arrays::ArrayBorrow<'_>) -> Result<bool, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bool_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        Ok(bool_arr.iter().any(|v| v == Some(true)))
    }

    fn bool_all(arr: arrays::ArrayBorrow<'_>) -> Result<bool, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bool_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected boolean array".to_string()))?;
        Ok(bool_arr.iter().all(|v| v == Some(true)))
    }

    fn first_value(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        for i in 0..arr_impl.inner.len() {
            if arr_impl.inner.is_valid(i) {
                return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(i, 1) }));
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(0, 0) }))
    }

    fn last_value(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        for i in (0..arr_impl.inner.len()).rev() {
            if arr_impl.inner.is_valid(i) {
                return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(i, 1) }));
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(0, 0) }))
    }

    // ========== Advanced Aggregations (Phase 10.5) ==========

    fn sum_checked_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut sum: i64 = 0;
            let mut has_value = false;
            for opt in int_arr.iter() {
                if let Some(v) = opt {
                    sum = sum.checked_add(v)
                        .ok_or_else(|| compute::ArrowError::ComputeError("Integer overflow in sum".to_string()))?;
                    has_value = true;
                }
            }
            return Ok(if has_value { Some(sum) } else { None });
        }

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let mut sum: i64 = 0;
            let mut has_value = false;
            for opt in int_arr.iter() {
                if let Some(v) = opt {
                    sum = sum.checked_add(v as i64)
                        .ok_or_else(|| compute::ArrowError::ComputeError("Integer overflow in sum".to_string()))?;
                    has_value = true;
                }
            }
            return Ok(if has_value { Some(sum) } else { None });
        }

        Err(compute::ArrowError::InvalidArgument("Expected integer array".to_string()))
    }

    fn sum_checked_i32(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i32>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let mut sum: i32 = 0;
            let mut has_value = false;
            for opt in int_arr.iter() {
                if let Some(v) = opt {
                    sum = sum.checked_add(v)
                        .ok_or_else(|| compute::ArrowError::ComputeError("Integer overflow in sum".to_string()))?;
                    has_value = true;
                }
            }
            return Ok(if has_value { Some(sum) } else { None });
        }

        Err(compute::ArrowError::InvalidArgument("Expected Int32 array".to_string()))
    }

    fn product_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! compute_product {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut product: i64 = 1;
                    let mut has_value = false;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            product = product.wrapping_mul(v as i64);
                            has_value = true;
                        }
                    }
                    return Ok(if has_value { Some(product) } else { None });
                }
            };
        }

        compute_product!(arrow_array::Int64Array);
        compute_product!(arrow_array::Int32Array);
        compute_product!(arrow_array::Int16Array);
        compute_product!(arrow_array::Int8Array);
        compute_product!(arrow_array::UInt64Array);
        compute_product!(arrow_array::UInt32Array);
        compute_product!(arrow_array::UInt16Array);
        compute_product!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("Expected integer array".to_string()))
    }

    fn product_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! compute_product {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut product: f64 = 1.0;
                    let mut has_value = false;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            product *= v as f64;
                            has_value = true;
                        }
                    }
                    return Ok(if has_value { Some(product) } else { None });
                }
            };
        }

        compute_product!(arrow_array::Float64Array);
        compute_product!(arrow_array::Float32Array);
        compute_product!(arrow_array::Int64Array);
        compute_product!(arrow_array::Int32Array);

        Err(compute::ArrowError::InvalidArgument("Expected numeric array".to_string()))
    }

    fn geometric_mean(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! compute_geom_mean {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut log_sum: f64 = 0.0;
                    let mut count: u64 = 0;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            let f = v as f64;
                            if f <= 0.0 {
                                return Err(compute::ArrowError::ComputeError(
                                    "Geometric mean requires positive values".to_string()
                                ));
                            }
                            log_sum += f.ln();
                            count += 1;
                        }
                    }
                    if count == 0 {
                        return Ok(None);
                    }
                    return Ok(Some((log_sum / count as f64).exp()));
                }
            };
        }

        compute_geom_mean!(arrow_array::Float64Array);
        compute_geom_mean!(arrow_array::Float32Array);
        compute_geom_mean!(arrow_array::Int64Array);
        compute_geom_mean!(arrow_array::Int32Array);

        Err(compute::ArrowError::InvalidArgument("Expected numeric array".to_string()))
    }

    fn harmonic_mean(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! compute_harm_mean {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut recip_sum: f64 = 0.0;
                    let mut count: u64 = 0;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            let f = v as f64;
                            if f == 0.0 {
                                return Err(compute::ArrowError::ComputeError(
                                    "Harmonic mean cannot include zero values".to_string()
                                ));
                            }
                            recip_sum += 1.0 / f;
                            count += 1;
                        }
                    }
                    if count == 0 {
                        return Ok(None);
                    }
                    return Ok(Some(count as f64 / recip_sum));
                }
            };
        }

        compute_harm_mean!(arrow_array::Float64Array);
        compute_harm_mean!(arrow_array::Float32Array);
        compute_harm_mean!(arrow_array::Int64Array);
        compute_harm_mean!(arrow_array::Int32Array);

        Err(compute::ArrowError::InvalidArgument("Expected numeric array".to_string()))
    }

    fn rms(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! compute_rms {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut sq_sum: f64 = 0.0;
                    let mut count: u64 = 0;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            let f = v as f64;
                            sq_sum += f * f;
                            count += 1;
                        }
                    }
                    if count == 0 {
                        return Ok(None);
                    }
                    return Ok(Some((sq_sum / count as f64).sqrt()));
                }
            };
        }

        compute_rms!(arrow_array::Float64Array);
        compute_rms!(arrow_array::Float32Array);
        compute_rms!(arrow_array::Int64Array);
        compute_rms!(arrow_array::Int32Array);

        Err(compute::ArrowError::InvalidArgument("Expected numeric array".to_string()))
    }

    fn count_distinct(arr: arrays::ArrayBorrow<'_>) -> Result<u64, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        use std::collections::HashSet;

        macro_rules! count_distinct_impl {
            ($arr_type:ty, $hash_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut set: HashSet<$hash_type> = HashSet::new();
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            set.insert(v as $hash_type);
                        }
                    }
                    return Ok(set.len() as u64);
                }
            };
        }

        count_distinct_impl!(arrow_array::Int64Array, i64);
        count_distinct_impl!(arrow_array::Int32Array, i32);
        count_distinct_impl!(arrow_array::Int16Array, i16);
        count_distinct_impl!(arrow_array::Int8Array, i8);
        count_distinct_impl!(arrow_array::UInt64Array, u64);
        count_distinct_impl!(arrow_array::UInt32Array, u32);
        count_distinct_impl!(arrow_array::UInt16Array, u16);
        count_distinct_impl!(arrow_array::UInt8Array, u8);

        if let Some(string_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut set: HashSet<&str> = HashSet::new();
            for opt in string_arr.iter() {
                if let Some(v) = opt {
                    set.insert(v);
                }
            }
            return Ok(set.len() as u64);
        }

        if let Some(bool_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>() {
            let mut set: HashSet<bool> = HashSet::new();
            for opt in bool_arr.iter() {
                if let Some(v) = opt {
                    set.insert(v);
                }
            }
            return Ok(set.len() as u64);
        }

        Err(compute::ArrowError::InvalidArgument("Unsupported array type for count_distinct".to_string()))
    }

    fn count_distinct_approx(arr: arrays::ArrayBorrow<'_>) -> Result<u64, compute::ArrowError> {
        // For now, use exact count. A proper implementation would use HyperLogLog.
        // This is a placeholder that provides correct results but not the memory savings
        // of a probabilistic data structure.
        Self::count_distinct(arr)
    }

    // ========== Selection & Merge Operations (Phase 10.2) ==========

    fn lexsort_limit(input_arrays: Vec<arrays::Array>, limit: u64, descending: Vec<bool>, nulls_first: Vec<bool>) -> Result<Vec<arrays::Array>, compute::ArrowError> {
        if input_arrays.is_empty() {
            return Ok(vec![]);
        }

        let len = input_arrays.len();
        if descending.len() != len || nulls_first.len() != len {
            return Err(compute::ArrowError::InvalidArgument(
                "descending and nulls_first must have same length as arrays".to_string()
            ));
        }

        // Build sort columns
        let sort_columns: Vec<arrow_ord::sort::SortColumn> = input_arrays.iter()
            .zip(descending.iter().zip(nulls_first.iter()))
            .map(|(arr, (&desc, &nf))| {
                let arr_impl = arr.get::<ArrayImpl>();
                arrow_ord::sort::SortColumn {
                    values: arr_impl.inner.clone(),
                    options: Some(arrow_ord::sort::SortOptions {
                        descending: desc,
                        nulls_first: nf,
                    }),
                }
            })
            .collect();

        // Get lexsort indices with limit
        let indices = arrow_ord::sort::lexsort_to_indices(&sort_columns, Some(limit as usize))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Take from each array using the sorted indices
        let mut results = Vec::with_capacity(len);
        for arr in &input_arrays {
            let arr_impl = arr.get::<ArrayImpl>();
            let taken = arrow_select::take::take(&*arr_impl.inner, &indices, None)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            results.push(arrays::Array::new(ArrayImpl { inner: taken }));
        }

        Ok(results)
    }

    fn partition_ranges(arr: arrays::ArrayBorrow<'_>) -> Result<Vec<(u64, u64)>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        if len == 0 {
            return Ok(vec![]);
        }

        // Find boundaries where values change
        let mut ranges = Vec::new();
        let mut start = 0usize;

        for i in 1..len {
            if !arrays_equal_at_index(&arr_impl.inner, i, i - 1) {
                ranges.push((start as u64, (i - start) as u64));
                start = i;
            }
        }
        // Add final range
        ranges.push((start as u64, (len - start) as u64));

        Ok(ranges)
    }

    fn merge_sorted(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>, descending: bool) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        // Concatenate arrays first
        let combined = arrow_select::concat::concat(&[&*left_impl.inner, &*right_impl.inner])
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Sort the combined array
        let options = arrow_ord::sort::SortOptions { descending, nulls_first: false };
        let indices = arrow_ord::sort::sort_to_indices(&combined, Some(options), None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        let result = arrow_select::take::take(&*combined, &indices, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn zip_select(mask: arrays::ArrayBorrow<'_>, left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let mask_impl = mask.get::<ArrayImpl>();
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let bool_mask = mask_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("mask must be boolean array".to_string()))?;

        // Try different array types
        macro_rules! try_zip {
            ($arr_type:ty) => {
                if let (Some(l), Some(r)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    let result = arrow_select::zip::zip(bool_mask, l, r)
                        .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
                    return Ok(arrays::Array::new(ArrayImpl { inner: result }));
                }
            };
        }

        try_zip!(arrow_array::Int64Array);
        try_zip!(arrow_array::Int32Array);
        try_zip!(arrow_array::Int16Array);
        try_zip!(arrow_array::Int8Array);
        try_zip!(arrow_array::UInt64Array);
        try_zip!(arrow_array::UInt32Array);
        try_zip!(arrow_array::UInt16Array);
        try_zip!(arrow_array::UInt8Array);
        try_zip!(arrow_array::Float64Array);
        try_zip!(arrow_array::Float32Array);
        try_zip!(arrow_array::BooleanArray);
        try_zip!(arrow_array::StringArray);

        Err(compute::ArrowError::InvalidArgument("zip_select: unsupported array type or mismatched types".to_string()))
    }

    // ========== Interval Arithmetic ==========

    fn interval_add(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_buffer::IntervalMonthDayNano;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
        ) {
            use arrow_array::Array as _;
            if l.len() != r.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::IntervalMonthDayNanoArray = l.iter()
                .zip(r.iter())
                .map(|(lv, rv)| match (lv, rv) {
                    (Some(li), Some(ri)) => {
                        let months = li.months.wrapping_add(ri.months);
                        let days = li.days.wrapping_add(ri.days);
                        let nanos = li.nanoseconds.wrapping_add(ri.nanoseconds);
                        Some(IntervalMonthDayNano::new(months, days, nanos))
                    }
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_add requires IntervalMonthDayNano arrays".to_string()))
    }

    fn interval_subtract(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_buffer::IntervalMonthDayNano;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
        ) {
            use arrow_array::Array as _;
            if l.len() != r.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::IntervalMonthDayNanoArray = l.iter()
                .zip(r.iter())
                .map(|(lv, rv)| match (lv, rv) {
                    (Some(li), Some(ri)) => {
                        let months = li.months.wrapping_sub(ri.months);
                        let days = li.days.wrapping_sub(ri.days);
                        let nanos = li.nanoseconds.wrapping_sub(ri.nanoseconds);
                        Some(IntervalMonthDayNano::new(months, days, nanos))
                    }
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_subtract requires IntervalMonthDayNano arrays".to_string()))
    }

    fn interval_multiply_scalar(arr: arrays::ArrayBorrow<'_>, factor: i64) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_buffer::IntervalMonthDayNano;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>() {
            let result: arrow_array::IntervalMonthDayNanoArray = interval_arr.iter()
                .map(|opt| opt.map(|i| {
                    let new_months = i.months.wrapping_mul(factor as i32);
                    let new_days = i.days.wrapping_mul(factor as i32);
                    let new_nanos = i.nanoseconds.wrapping_mul(factor);
                    IntervalMonthDayNano::new(new_months, new_days, new_nanos)
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_multiply_scalar requires IntervalMonthDayNano array".to_string()))
    }

    fn interval_negate(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_buffer::IntervalMonthDayNano;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(interval_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>() {
            let result: arrow_array::IntervalMonthDayNanoArray = interval_arr.iter()
                .map(|opt| opt.map(|i| {
                    let new_months = i.months.wrapping_neg();
                    let new_days = i.days.wrapping_neg();
                    let new_nanos = i.nanoseconds.wrapping_neg();
                    IntervalMonthDayNano::new(new_months, new_days, new_nanos)
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_negate requires IntervalMonthDayNano array".to_string()))
    }

    fn interval_eq(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
        ) {
            use arrow_array::Array as _;
            if l.len() != r.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::BooleanArray = l.iter()
                .zip(r.iter())
                .map(|(lv, rv)| match (lv, rv) {
                    (Some(li), Some(ri)) => Some(li == ri),
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_eq requires IntervalMonthDayNano arrays".to_string()))
    }

    fn interval_lt(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
        ) {
            use arrow_array::Array as _;
            if l.len() != r.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            // Compare by total: months*30*24*60*60*1e9 + days*24*60*60*1e9 + nanos (approximate)
            let result: arrow_array::BooleanArray = l.iter()
                .zip(r.iter())
                .map(|(lv, rv)| match (lv, rv) {
                    (Some(li), Some(ri)) => {
                        // Approximate comparison (months as 30 days)
                        let l_total_nanos = li.months as i64 * 30 * 86400_000_000_000 + li.days as i64 * 86400_000_000_000 + li.nanoseconds;
                        let r_total_nanos = ri.months as i64 * 30 * 86400_000_000_000 + ri.days as i64 * 86400_000_000_000 + ri.nanoseconds;
                        Some(l_total_nanos < r_total_nanos)
                    }
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_lt requires IntervalMonthDayNano arrays".to_string()))
    }

    fn interval_gt(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
        ) {
            use arrow_array::Array as _;
            if l.len() != r.len() {
                return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
            }
            let result: arrow_array::BooleanArray = l.iter()
                .zip(r.iter())
                .map(|(lv, rv)| match (lv, rv) {
                    (Some(li), Some(ri)) => {
                        let l_total_nanos = li.months as i64 * 30 * 86400_000_000_000 + li.days as i64 * 86400_000_000_000 + li.nanoseconds;
                        let r_total_nanos = ri.months as i64 * 30 * 86400_000_000_000 + ri.days as i64 * 86400_000_000_000 + ri.nanoseconds;
                        Some(l_total_nanos > r_total_nanos)
                    }
                    _ => None,
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("interval_gt requires IntervalMonthDayNano arrays".to_string()))
    }

    // ========== Array Sampling Operations ==========

    fn sample_n(arr: arrays::ArrayBorrow<'_>, n: u64, seed: Option<u64>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        if n == 0 {
            // Return empty array with same type
            return Ok(arrays::Array::new(ArrayImpl {
                inner: arr_impl.inner.slice(0, 0),
            }));
        }

        if n >= len as u64 {
            // Return all elements
            return Ok(arrays::Array::new(ArrayImpl {
                inner: arr_impl.inner.clone(),
            }));
        }

        // Simple PRNG based on seed
        let mut state = seed.unwrap_or(42);
        let mut lcg = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };

        // Generate random indices using Fisher-Yates shuffle approach
        let mut indices: Vec<u64> = (0..len as u64).collect();
        for i in (1..len).rev() {
            let j = (lcg() as usize) % (i + 1);
            indices.swap(i, j);
        }
        indices.truncate(n as usize);

        // Take the sampled elements
        let indices_arr: arrow_array::UInt64Array = indices.into_iter().map(Some).collect();
        let result = arrow_select::take::take(&*arr_impl.inner, &indices_arr, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn sample_fraction(arr: arrays::ArrayBorrow<'_>, fraction: f64, seed: Option<u64>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        if fraction <= 0.0 {
            // Return empty array
            return Ok(arrays::Array::new(ArrayImpl {
                inner: arr_impl.inner.slice(0, 0),
            }));
        }

        if fraction >= 1.0 {
            // Return all elements
            return Ok(arrays::Array::new(ArrayImpl {
                inner: arr_impl.inner.clone(),
            }));
        }

        let n = ((len as f64) * fraction).round() as u64;
        Self::sample_n(arr, n, seed)
    }

    fn shuffle(arr: arrays::ArrayBorrow<'_>, seed: Option<u64>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        if len <= 1 {
            return Ok(arrays::Array::new(ArrayImpl {
                inner: arr_impl.inner.clone(),
            }));
        }

        // Simple PRNG based on seed
        let mut state = seed.unwrap_or(42);
        let mut lcg = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };

        // Generate shuffled indices using Fisher-Yates shuffle
        let mut indices: Vec<u64> = (0..len as u64).collect();
        for i in (1..len).rev() {
            let j = (lcg() as usize) % (i + 1);
            indices.swap(i, j);
        }

        // Take elements in shuffled order
        let indices_arr: arrow_array::UInt64Array = indices.into_iter().map(Some).collect();
        let result = arrow_select::take::take(&*arr_impl.inner, &indices_arr, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    // ========== Bitwise Operations ==========

    fn bitwise_and(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        // Try Int64 first
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_and(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Try Int32
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_and(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_and requires integer arrays".to_string()))
    }

    fn bitwise_or(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_or(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_or(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_or requires integer arrays".to_string()))
    }

    fn bitwise_xor(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_xor(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_xor(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_xor requires integer arrays".to_string()))
    }

    fn bitwise_not(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result = arrow_arith::bitwise::bitwise_not(a)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result = arrow_arith::bitwise::bitwise_not(a)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_not requires integer array".to_string()))
    }

    fn bitwise_and_not(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_and_not(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
        ) {
            let result = arrow_arith::bitwise::bitwise_and_not(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_and_not requires integer arrays".to_string()))
    }

    fn bitwise_shift_left(arr: arrays::ArrayBorrow<'_>, n: u32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let shift_arr = arrow_array::Int64Array::from(vec![n as i64; a.len()]);
            let result = arrow_arith::bitwise::bitwise_shift_left(a, &shift_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let shift_arr = arrow_array::Int32Array::from(vec![n as i32; a.len()]);
            let result = arrow_arith::bitwise::bitwise_shift_left(a, &shift_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_shift_left requires integer array".to_string()))
    }

    fn bitwise_shift_right(arr: arrays::ArrayBorrow<'_>, n: u32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let shift_arr = arrow_array::Int64Array::from(vec![n as i64; a.len()]);
            let result = arrow_arith::bitwise::bitwise_shift_right(a, &shift_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let shift_arr = arrow_array::Int32Array::from(vec![n as i32; a.len()]);
            let result = arrow_arith::bitwise::bitwise_shift_right(a, &shift_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_shift_right requires integer array".to_string()))
    }

    fn bitwise_and_scalar(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result = arrow_arith::bitwise::bitwise_and_scalar(a, scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result = arrow_arith::bitwise::bitwise_and_scalar(a, scalar as i32)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_and_scalar requires integer array".to_string()))
    }

    fn bitwise_or_scalar(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result = arrow_arith::bitwise::bitwise_or_scalar(a, scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result = arrow_arith::bitwise::bitwise_or_scalar(a, scalar as i32)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_or_scalar requires integer array".to_string()))
    }

    fn bitwise_xor_scalar(arr: arrays::ArrayBorrow<'_>, scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result = arrow_arith::bitwise::bitwise_xor_scalar(a, scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(a) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result = arrow_arith::bitwise::bitwise_xor_scalar(a, scalar as i32)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("bitwise_xor_scalar requires integer array".to_string()))
    }

    fn bitwise_and_agg(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Macro for bitwise AND aggregate
        macro_rules! bitwise_and_agg_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let Some(a) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Option<$val_type> = None;
                    for opt in a.iter() {
                        if let Some(v) = opt {
                            result = match result {
                                None => Some(v),
                                Some(acc) => Some(acc & v),
                            };
                        }
                    }
                    return Ok(result.map(|v| v as i64));
                }
            };
        }

        bitwise_and_agg_impl!(arrow_array::Int64Array, i64);
        bitwise_and_agg_impl!(arrow_array::Int32Array, i32);
        bitwise_and_agg_impl!(arrow_array::Int16Array, i16);
        bitwise_and_agg_impl!(arrow_array::Int8Array, i8);
        bitwise_and_agg_impl!(arrow_array::UInt64Array, u64);
        bitwise_and_agg_impl!(arrow_array::UInt32Array, u32);
        bitwise_and_agg_impl!(arrow_array::UInt16Array, u16);
        bitwise_and_agg_impl!(arrow_array::UInt8Array, u8);

        Err(compute::ArrowError::InvalidArgument("bitwise_and_agg requires integer array".to_string()))
    }

    fn bitwise_or_agg(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Macro for bitwise OR aggregate
        macro_rules! bitwise_or_agg_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let Some(a) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Option<$val_type> = None;
                    for opt in a.iter() {
                        if let Some(v) = opt {
                            result = match result {
                                None => Some(v),
                                Some(acc) => Some(acc | v),
                            };
                        }
                    }
                    return Ok(result.map(|v| v as i64));
                }
            };
        }

        bitwise_or_agg_impl!(arrow_array::Int64Array, i64);
        bitwise_or_agg_impl!(arrow_array::Int32Array, i32);
        bitwise_or_agg_impl!(arrow_array::Int16Array, i16);
        bitwise_or_agg_impl!(arrow_array::Int8Array, i8);
        bitwise_or_agg_impl!(arrow_array::UInt64Array, u64);
        bitwise_or_agg_impl!(arrow_array::UInt32Array, u32);
        bitwise_or_agg_impl!(arrow_array::UInt16Array, u16);
        bitwise_or_agg_impl!(arrow_array::UInt8Array, u8);

        Err(compute::ArrowError::InvalidArgument("bitwise_or_agg requires integer array".to_string()))
    }

    fn bitwise_xor_agg(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Macro for bitwise XOR aggregate
        macro_rules! bitwise_xor_agg_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let Some(a) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut result: Option<$val_type> = None;
                    for opt in a.iter() {
                        if let Some(v) = opt {
                            result = match result {
                                None => Some(v),
                                Some(acc) => Some(acc ^ v),
                            };
                        }
                    }
                    return Ok(result.map(|v| v as i64));
                }
            };
        }

        bitwise_xor_agg_impl!(arrow_array::Int64Array, i64);
        bitwise_xor_agg_impl!(arrow_array::Int32Array, i32);
        bitwise_xor_agg_impl!(arrow_array::Int16Array, i16);
        bitwise_xor_agg_impl!(arrow_array::Int8Array, i8);
        bitwise_xor_agg_impl!(arrow_array::UInt64Array, u64);
        bitwise_xor_agg_impl!(arrow_array::UInt32Array, u32);
        bitwise_xor_agg_impl!(arrow_array::UInt16Array, u16);
        bitwise_xor_agg_impl!(arrow_array::UInt8Array, u8);

        Err(compute::ArrowError::InvalidArgument("bitwise_xor_agg requires integer array".to_string()))
    }

    // ========== Additional String Operations ==========

    fn substring(arr: arrays::ArrayBorrow<'_>, start: i64, length: Option<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let result = arrow_string::substring::substring(
            string_arr,
            start,
            length.map(|l| l as u64),
        ).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_like(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let pattern_scalar = arrow_array::Scalar::new(arrow_array::StringArray::from(vec![pattern.as_str()]));
        let result = arrow_string::like::like(string_arr, &pattern_scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_ilike(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let pattern_scalar = arrow_array::Scalar::new(arrow_array::StringArray::from(vec![pattern.as_str()]));
        let result = arrow_string::like::ilike(string_arr, &pattern_scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_nlike(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let pattern_scalar = arrow_array::Scalar::new(arrow_array::StringArray::from(vec![pattern.as_str()]));
        let result = arrow_string::like::nlike(string_arr, &pattern_scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_nilike(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;
        let pattern_scalar = arrow_array::Scalar::new(arrow_array::StringArray::from(vec![pattern.as_str()]));
        let result = arrow_string::like::nilike(string_arr, &pattern_scalar)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_lpad(arr: arrays::ArrayBorrow<'_>, length: u64, fill: String) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Build result manually since arrow_string::pad may not be available
        let mut builder = arrow_array::builder::StringBuilder::new();
        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let char_count = s.chars().count();
                let target_len = length as usize;
                if char_count >= target_len {
                    builder.append_value(s);
                } else {
                    let pad_count = target_len - char_count;
                    let fill_chars: Vec<char> = fill.chars().collect();
                    if fill_chars.is_empty() {
                        builder.append_value(s);
                    } else {
                        let mut result = String::with_capacity(target_len);
                        for i in 0..pad_count {
                            result.push(fill_chars[i % fill_chars.len()]);
                        }
                        result.push_str(s);
                        builder.append_value(&result);
                    }
                }
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_rpad(arr: arrays::ArrayBorrow<'_>, length: u64, fill: String) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::StringBuilder::new();
        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let char_count = s.chars().count();
                let target_len = length as usize;
                if char_count >= target_len {
                    builder.append_value(s);
                } else {
                    let pad_count = target_len - char_count;
                    let fill_chars: Vec<char> = fill.chars().collect();
                    if fill_chars.is_empty() {
                        builder.append_value(s);
                    } else {
                        let mut result = String::with_capacity(target_len);
                        result.push_str(s);
                        for i in 0..pad_count {
                            result.push(fill_chars[i % fill_chars.len()]);
                        }
                        builder.append_value(&result);
                    }
                }
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_ltrim(arr: arrays::ArrayBorrow<'_>, chars: Option<String>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::StringBuilder::new();
        let trim_chars: Vec<char> = chars.as_ref().map(|s| s.chars().collect()).unwrap_or_else(|| vec![' ', '\t', '\n', '\r']);

        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let trimmed = s.trim_start_matches(|c| trim_chars.contains(&c));
                builder.append_value(trimmed);
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_rtrim(arr: arrays::ArrayBorrow<'_>, chars: Option<String>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::StringBuilder::new();
        let trim_chars: Vec<char> = chars.as_ref().map(|s| s.chars().collect()).unwrap_or_else(|| vec![' ', '\t', '\n', '\r']);

        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let trimmed = s.trim_end_matches(|c| trim_chars.contains(&c));
                builder.append_value(trimmed);
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_repeat(arr: arrays::ArrayBorrow<'_>, count: u64) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::StringBuilder::new();
        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let repeated = s.repeat(count as usize);
                builder.append_value(&repeated);
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_reverse(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::StringBuilder::new();
        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let reversed: String = s.chars().rev().collect();
                builder.append_value(&reversed);
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_ascii(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::Int32Builder::new();
        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                if let Some(first_char) = s.chars().next() {
                    builder.append_value(first_char as i32);
                } else {
                    builder.append_value(0); // Empty string returns 0
                }
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_replace(arr: arrays::ArrayBorrow<'_>, pattern: String, replacement: String) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let mut builder = arrow_array::builder::StringBuilder::new();
        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                builder.append_null();
            } else {
                let s = string_arr.value(i);
                let replaced = s.replace(&pattern, &replacement);
                builder.append_value(&replaced);
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(builder.finish()) }))
    }

    fn string_split(arr: arrays::ArrayBorrow<'_>, delimiter: String) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Build a list array of strings
        let mut list_builder = arrow_array::builder::ListBuilder::new(arrow_array::builder::StringBuilder::new());

        for i in 0..string_arr.len() {
            if string_arr.is_null(i) {
                list_builder.append_null();
            } else {
                let s = string_arr.value(i);
                let parts: Vec<&str> = s.split(&delimiter).collect();
                let values_builder = list_builder.values();
                for part in parts {
                    values_builder.append_value(part);
                }
                list_builder.append(true);
            }
        }
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(list_builder.finish()) }))
    }

    // ========== Advanced String Operations ==========

    fn string_left(arr: arrays::ArrayBorrow<'_>, n: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| {
                let chars: Vec<char> = s.chars().collect();
                chars.iter().take(n as usize).collect::<String>()
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_right(arr: arrays::ArrayBorrow<'_>, n: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| {
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len();
                let start = if len > n as usize { len - n as usize } else { 0 };
                chars[start..].iter().collect::<String>()
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_initcap(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| {
                let mut result = String::with_capacity(s.len());
                let mut capitalize_next = true;
                for c in s.chars() {
                    if c.is_whitespace() || !c.is_alphanumeric() {
                        result.push(c);
                        capitalize_next = true;
                    } else if capitalize_next {
                        result.extend(c.to_uppercase());
                        capitalize_next = false;
                    } else {
                        result.extend(c.to_lowercase());
                    }
                }
                result
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_position(arr: arrays::ArrayBorrow<'_>, substring: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Return 1-indexed position (0 if not found, like SQL)
        let result: arrow_array::UInt64Array = string_arr.iter()
            .map(|opt| opt.map(|s| {
                match s.find(&substring) {
                    Some(idx) => {
                        // Convert byte index to character index
                        let char_idx = s[..idx].chars().count();
                        (char_idx + 1) as u64 // 1-indexed
                    }
                    None => 0
                }
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_position_from(arr: arrays::ArrayBorrow<'_>, substring: String, start: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Convert 1-indexed start to 0-indexed
        let start_idx = if start > 0 { start as usize - 1 } else { 0 };

        let result: arrow_array::UInt64Array = string_arr.iter()
            .map(|opt| opt.map(|s| {
                let chars: Vec<char> = s.chars().collect();
                if start_idx >= chars.len() {
                    return 0;
                }
                // Get substring from start position
                let search_str: String = chars[start_idx..].iter().collect();
                match search_str.find(&substring) {
                    Some(idx) => {
                        // Convert byte index to character index in the substring
                        let char_idx = search_str[..idx].chars().count();
                        (start_idx + char_idx + 1) as u64 // 1-indexed
                    }
                    None => 0
                }
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_translate(arr: arrays::ArrayBorrow<'_>, from_chars: String, to_chars: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Build translation map
        let from: Vec<char> = from_chars.chars().collect();
        let to: Vec<char> = to_chars.chars().collect();
        let mut translation: HashMap<char, Option<char>> = HashMap::new();
        for (i, &fc) in from.iter().enumerate() {
            if i < to.len() {
                translation.insert(fc, Some(to[i]));
            } else {
                // If to is shorter, characters are deleted
                translation.insert(fc, None);
            }
        }

        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| {
                s.chars()
                    .filter_map(|c| {
                        match translation.get(&c) {
                            Some(Some(replacement)) => Some(*replacement),
                            Some(None) => None, // Delete character
                            None => Some(c), // Keep original
                        }
                    })
                    .collect::<String>()
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_concat_ws(separator: String, input_arrays: Vec<arrays::Array>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;

        if input_arrays.is_empty() {
            return Ok(arrays::Array::new(ArrayImpl {
                inner: Arc::new(arrow_array::StringArray::from(Vec::<Option<&str>>::new()))
            }));
        }

        // Get all string arrays
        let string_arrays: Vec<&arrow_array::StringArray> = input_arrays.iter()
            .map(|arr| {
                let arr_impl = arr.get::<ArrayImpl>();
                arr_impl.inner.as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| compute::ArrowError::InvalidArgument("All arrays must be string arrays".to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Check all arrays have the same length
        let len = string_arrays[0].len();
        for arr in &string_arrays {
            if arr.len() != len {
                return Err(compute::ArrowError::InvalidArgument("All arrays must have the same length".to_string()));
            }
        }

        // Concatenate
        let result: arrow_array::StringArray = (0..len)
            .map(|i| {
                let parts: Vec<&str> = string_arrays.iter()
                    .filter_map(|arr| {
                        if arr.is_null(i) {
                            None
                        } else {
                            Some(arr.value(i))
                        }
                    })
                    .collect();
                if parts.is_empty() {
                    None
                } else {
                    Some(parts.join(&separator))
                }
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_split_part(arr: arrays::ArrayBorrow<'_>, delimiter: String, part: u32) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        // Part is 1-indexed (like SQL SPLIT_PART)
        let part_idx = if part > 0 { part as usize - 1 } else { 0 };

        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.and_then(|s| {
                let parts: Vec<&str> = s.split(&delimiter).collect();
                parts.get(part_idx).map(|&p| p.to_string())
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn substring_by_char(arr: arrays::ArrayBorrow<'_>, start: i64, length: Option<u64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let string_arr = arr_impl.inner.as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Expected string array".to_string()))?;

        let result: arrow_array::StringArray = string_arr.iter()
            .map(|opt| opt.map(|s| {
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;

                // Handle negative start (from end)
                let actual_start = if start < 0 {
                    (len + start).max(0) as usize
                } else {
                    start.min(len) as usize
                };

                // Calculate end position
                let end_pos = match length {
                    Some(l) => (actual_start + l as usize).min(chars.len()),
                    None => chars.len(),
                };

                chars[actual_start..end_pos].iter().collect::<String>()
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn string_chr(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        // Support various integer types for ASCII codes
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let result: arrow_array::StringArray = int_arr.iter()
                .map(|opt| opt.and_then(|code| {
                    if code >= 0 && code <= 127 {
                        Some((code as u8 as char).to_string())
                    } else {
                        None // Non-ASCII codes return null
                    }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::StringArray = int_arr.iter()
                .map(|opt| opt.and_then(|code| {
                    if code >= 0 && code <= 127 {
                        Some((code as u8 as char).to_string())
                    } else {
                        None // Non-ASCII codes return null
                    }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::UInt8Array>() {
            let result: arrow_array::StringArray = int_arr.iter()
                .map(|opt| opt.and_then(|code| {
                    if code <= 127 {
                        Some((code as char).to_string())
                    } else {
                        None // Non-ASCII codes return null
                    }
                }))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("Expected integer array for ASCII codes".to_string()))
    }

    // ========== Array Generation ==========

    fn make_array_i64(start: i64, end: i64, step: i64) -> Result<arrays::Array, compute::ArrowError> {
        if step == 0 {
            return Err(compute::ArrowError::InvalidArgument("Step cannot be zero".to_string()));
        }

        let mut values = Vec::new();
        let mut current = start;

        if step > 0 {
            while current < end {
                values.push(current);
                current += step;
            }
        } else {
            while current > end {
                values.push(current);
                current += step;
            }
        }

        let arr = arrow_array::Int64Array::from(values);
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(arr) }))
    }

    fn array_fill_i64(value: i64, length: u64) -> arrays::Array {
        let values: Vec<i64> = vec![value; length as usize];
        let arr = arrow_array::Int64Array::from(values);
        arrays::Array::new(ArrayImpl { inner: Arc::new(arr) })
    }

    fn array_fill_f64(value: f64, length: u64) -> arrays::Array {
        let values: Vec<f64> = vec![value; length as usize];
        let arr = arrow_array::Float64Array::from(values);
        arrays::Array::new(ArrayImpl { inner: Arc::new(arr) })
    }

    fn array_fill_string(value: String, length: u64) -> arrays::Array {
        let values: Vec<&str> = vec![value.as_str(); length as usize];
        let arr = arrow_array::StringArray::from(values);
        arrays::Array::new(ArrayImpl { inner: Arc::new(arr) })
    }

    fn array_fill_null(data_type: types::DataType, length: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arrow_type = convert::to_arrow_data_type(&data_type);
        let arr = arrow_array::new_null_array(&arrow_type, length as usize);
        Ok(arrays::Array::new(ArrayImpl { inner: arr }))
    }

    // ========== Conditional Operations ==========

    fn nullif(arr: arrays::ArrayBorrow<'_>, condition: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let cond_impl = condition.get::<ArrayImpl>();

        let bool_arr = cond_impl.inner.as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Condition must be boolean array".to_string()))?;

        let result = arrow_select::nullif::nullif(&*arr_impl.inner, bool_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn if_else(condition: arrays::ArrayBorrow<'_>, truthy: arrays::ArrayBorrow<'_>, falsy: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let cond_impl = condition.get::<ArrayImpl>();
        let truthy_impl = truthy.get::<ArrayImpl>();
        let falsy_impl = falsy.get::<ArrayImpl>();

        let bool_arr = cond_impl.inner.as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("Condition must be boolean array".to_string()))?;

        // Try Int64 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Int32 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Float64 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try String arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Boolean arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Int16 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Int16Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Int16Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Int8 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Int8Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Int8Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try UInt64 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::UInt64Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::UInt64Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try UInt32 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::UInt32Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::UInt32Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try UInt16 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::UInt16Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::UInt16Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try UInt8 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::UInt8Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::UInt8Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Float32 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Binary arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Date32 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Date32Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Date64 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Date64Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try TimestampSecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampSecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampSecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try TimestampMillisecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMillisecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMillisecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try TimestampMicrosecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try TimestampNanosecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampNanosecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::TimestampNanosecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Decimal128 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Decimal128Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try LargeBinary arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try LargeString arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Decimal256 arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Decimal256Array>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Time32Second arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Time32SecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Time32SecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Time32Millisecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Time32MillisecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Time32MillisecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Time64Microsecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Time64MicrosecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Time64MicrosecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Time64Nanosecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::Time64NanosecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::Time64NanosecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try DurationSecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::DurationSecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::DurationSecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try DurationMillisecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::DurationMillisecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::DurationMillisecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try DurationMicrosecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::DurationMicrosecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::DurationMicrosecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try DurationNanosecond arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::DurationNanosecondArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::DurationNanosecondArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try FixedSizeBinary arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeBinaryArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeBinaryArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try IntervalYearMonth arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::IntervalYearMonthArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::IntervalYearMonthArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try IntervalDayTime arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::IntervalDayTimeArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::IntervalDayTimeArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try IntervalMonthDayNano arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::IntervalMonthDayNanoArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try List arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::ListArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try LargeList arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::LargeListArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try FixedSizeList arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeListArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::FixedSizeListArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Struct arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::StructArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::StructArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        // Try Null arrays
        if let (Some(t), Some(f)) = (
            truthy_impl.inner.as_any().downcast_ref::<arrow_array::NullArray>(),
            falsy_impl.inner.as_any().downcast_ref::<arrow_array::NullArray>(),
        ) {
            let result = arrow_select::zip::zip(bool_arr, t, f)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: result }));
        }

        Err(compute::ArrowError::NotImplemented("if_else not implemented for this array type".to_string()))
    }

    // ========== SQL Functions ==========

    fn between_i64(arr: arrays::ArrayBorrow<'_>, low: i64, high: i64) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result: arrow_array::BooleanArray = int_arr.iter()
                .map(|v| v.map(|x| x >= low && x <= high))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("between_i64 requires an Int64 array".to_string()))
    }

    fn between_f64(arr: arrays::ArrayBorrow<'_>, low: f64, high: f64) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let result: arrow_array::BooleanArray = float_arr.iter()
                .map(|v| v.map(|x| x >= low && x <= high))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("between_f64 requires a Float64 array".to_string()))
    }

    fn between_string(arr: arrays::ArrayBorrow<'_>, low: String, high: String) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let result: arrow_array::BooleanArray = str_arr.iter()
                .map(|v| v.map(|x| x >= low.as_str() && x <= high.as_str()))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::InvalidArgument("between_string requires a String array".to_string()))
    }

    fn greatest(input_arrays: Vec<arrays::Array>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        use arrow_ord::cmp::gt;

        if input_arrays.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("greatest requires at least one array".to_string()));
        }
        if input_arrays.len() == 1 {
            let arr_impl = input_arrays[0].get::<ArrayImpl>();
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        // Start with first array and iteratively take greater values
        let first = input_arrays[0].get::<ArrayImpl>();
        let mut result = first.inner.clone();

        for arr in input_arrays.iter().skip(1) {
            let arr_impl = arr.get::<ArrayImpl>();
            let gt_mask = gt(&result, &arr_impl.inner)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

            // Use zip: if result > arr_impl, keep result; else use arr_impl
            result = arrow_select::zip::zip(&gt_mask, &result, &arr_impl.inner)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        }

        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn least(input_arrays: Vec<arrays::Array>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        use arrow_ord::cmp::lt;

        if input_arrays.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("least requires at least one array".to_string()));
        }
        if input_arrays.len() == 1 {
            let arr_impl = input_arrays[0].get::<ArrayImpl>();
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        // Start with first array and iteratively take lesser values
        let first = input_arrays[0].get::<ArrayImpl>();
        let mut result = first.inner.clone();

        for arr in input_arrays.iter().skip(1) {
            let arr_impl = arr.get::<ArrayImpl>();
            let lt_mask = lt(&result, &arr_impl.inner)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

            // Use zip: if result < arr_impl, keep result; else use arr_impl
            result = arrow_select::zip::zip(&lt_mask, &result, &arr_impl.inner)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        }

        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn nullif_eq(arr: arrays::ArrayBorrow<'_>, compare: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        use arrow_ord::cmp::eq;

        let arr_impl = arr.get::<ArrayImpl>();
        let compare_impl = compare.get::<ArrayImpl>();

        // Compare arrays for equality
        let eq_mask = eq(&arr_impl.inner, &compare_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Create null array of same type for null replacement
        let null_arr = arrow_array::new_null_array(arr_impl.inner.data_type(), arr_impl.inner.len());

        // Use zip: if equal, use null; else use original value
        let result = arrow_select::zip::zip(&eq_mask, &null_arr, &arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn string_agg(arr: arrays::ArrayBorrow<'_>, separator: String) -> Result<Option<String>, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let non_null_values: Vec<&str> = str_arr.iter()
                .filter_map(|v| v)
                .collect();

            if non_null_values.is_empty() {
                return Ok(None);
            }

            return Ok(Some(non_null_values.join(&separator)));
        }

        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let non_null_values: Vec<&str> = str_arr.iter()
                .filter_map(|v| v)
                .collect();

            if non_null_values.is_empty() {
                return Ok(None);
            }

            return Ok(Some(non_null_values.join(&separator)));
        }

        Err(compute::ArrowError::InvalidArgument("string_agg requires a String array".to_string()))
    }

    // ========== Advanced Null Handling ==========

    fn fill_forward(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Forward fill: carry last valid value forward
        macro_rules! fill_forward_impl {
            ($arr:expr, $arr_type:ty) => {{
                let typed_arr = $arr.as_any().downcast_ref::<$arr_type>().unwrap();
                let mut last_value: Option<_> = None;
                let result: $arr_type = typed_arr.iter()
                    .map(|v| {
                        if v.is_some() {
                            last_value = v;
                        }
                        last_value
                    })
                    .collect();
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
            }};
        }

        // Integer types
        if arr_impl.inner.as_any().is::<arrow_array::Int64Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Int64Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Int32Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Int32Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Int16Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Int16Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Int8Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Int8Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt64Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::UInt64Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt32Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::UInt32Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt16Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::UInt16Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt8Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::UInt8Array);
        }

        // Float types
        if arr_impl.inner.as_any().is::<arrow_array::Float64Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Float64Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Float32Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Float32Array);
        }

        // Boolean
        if arr_impl.inner.as_any().is::<arrow_array::BooleanArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::BooleanArray);
        }

        // Date types
        if arr_impl.inner.as_any().is::<arrow_array::Date32Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Date32Array);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Date64Array>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Date64Array);
        }

        // String arrays need special handling
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut last_value: Option<String> = None;
            let result: arrow_array::StringArray = str_arr.iter()
                .map(|v| {
                    if let Some(s) = v {
                        last_value = Some(s.to_string());
                    }
                    last_value.clone()
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // LargeString arrays
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let mut last_value: Option<String> = None;
            let result: arrow_array::LargeStringArray = str_arr.iter()
                .map(|v| {
                    if let Some(s) = v {
                        last_value = Some(s.to_string());
                    }
                    last_value.clone()
                })
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Timestamp types
        if arr_impl.inner.as_any().is::<arrow_array::TimestampSecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::TimestampSecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::TimestampMillisecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::TimestampMillisecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::TimestampMicrosecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::TimestampMicrosecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::TimestampNanosecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::TimestampNanosecondArray);
        }

        // Duration types
        if arr_impl.inner.as_any().is::<arrow_array::DurationSecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::DurationSecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::DurationMillisecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::DurationMillisecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::DurationMicrosecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::DurationMicrosecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::DurationNanosecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::DurationNanosecondArray);
        }

        // Time types
        if arr_impl.inner.as_any().is::<arrow_array::Time32SecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Time32SecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Time32MillisecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Time32MillisecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Time64MicrosecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Time64MicrosecondArray);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Time64NanosecondArray>() {
            fill_forward_impl!(arr_impl.inner, arrow_array::Time64NanosecondArray);
        }

        Err(compute::ArrowError::NotImplemented("fill_forward not implemented for this array type".to_string()))
    }

    fn fill_backward(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Backward fill: carry next valid value backward
        // We iterate from end to start, then reverse

        // Macro for backward fill implementation
        macro_rules! fill_backward_impl {
            ($arr:expr, $arr_type:ty, $val_type:ty) => {{
                let typed_arr = $arr.as_any().downcast_ref::<$arr_type>().unwrap();
                let mut next_value: Option<$val_type> = None;
                let mut values: Vec<Option<$val_type>> = typed_arr.iter()
                    .rev()
                    .map(|v| {
                        if v.is_some() {
                            next_value = v;
                        }
                        next_value
                    })
                    .collect();
                values.reverse();
                let result: $arr_type = values.into_iter().collect();
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
            }};
        }

        // Integer types
        if arr_impl.inner.as_any().is::<arrow_array::Int64Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Int64Array, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Int32Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Int32Array, i32);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Int16Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Int16Array, i16);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Int8Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Int8Array, i8);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt64Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::UInt64Array, u64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt32Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::UInt32Array, u32);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt16Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::UInt16Array, u16);
        }
        if arr_impl.inner.as_any().is::<arrow_array::UInt8Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::UInt8Array, u8);
        }

        // Float types
        if arr_impl.inner.as_any().is::<arrow_array::Float64Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Float64Array, f64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Float32Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Float32Array, f32);
        }

        // Boolean
        if arr_impl.inner.as_any().is::<arrow_array::BooleanArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::BooleanArray, bool);
        }

        // Date types
        if arr_impl.inner.as_any().is::<arrow_array::Date32Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Date32Array, i32);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Date64Array>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Date64Array, i64);
        }

        // String arrays need special handling
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut next_value: Option<String> = None;
            let mut values: Vec<Option<String>> = str_arr.iter()
                .rev()
                .map(|v| {
                    if let Some(s) = v {
                        next_value = Some(s.to_string());
                    }
                    next_value.clone()
                })
                .collect();
            values.reverse();
            let result: arrow_array::StringArray = values.into_iter().map(|v| v.as_deref().map(|s| s.to_string())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // LargeString arrays
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let mut next_value: Option<String> = None;
            let mut values: Vec<Option<String>> = str_arr.iter()
                .rev()
                .map(|v| {
                    if let Some(s) = v {
                        next_value = Some(s.to_string());
                    }
                    next_value.clone()
                })
                .collect();
            values.reverse();
            let result: arrow_array::LargeStringArray = values.into_iter().map(|v| v.as_deref().map(|s| s.to_string())).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Timestamp types
        if arr_impl.inner.as_any().is::<arrow_array::TimestampSecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::TimestampSecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::TimestampMillisecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::TimestampMillisecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::TimestampMicrosecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::TimestampMicrosecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::TimestampNanosecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::TimestampNanosecondArray, i64);
        }

        // Duration types
        if arr_impl.inner.as_any().is::<arrow_array::DurationSecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::DurationSecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::DurationMillisecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::DurationMillisecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::DurationMicrosecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::DurationMicrosecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::DurationNanosecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::DurationNanosecondArray, i64);
        }

        // Time types
        if arr_impl.inner.as_any().is::<arrow_array::Time32SecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Time32SecondArray, i32);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Time32MillisecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Time32MillisecondArray, i32);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Time64MicrosecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Time64MicrosecondArray, i64);
        }
        if arr_impl.inner.as_any().is::<arrow_array::Time64NanosecondArray>() {
            fill_backward_impl!(arr_impl.inner, arrow_array::Time64NanosecondArray, i64);
        }

        Err(compute::ArrowError::NotImplemented("fill_backward not implemented for this array type".to_string()))
    }

    fn drop_nulls(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Create a boolean filter array where true = non-null
        let len = arr_impl.inner.len();
        let filter: BooleanArray = (0..len)
            .map(|i| Some(arr_impl.inner.is_valid(i)))
            .collect();

        let result = arrow_select::filter::filter(&*arr_impl.inner, &filter)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    // ========== Set Operations ==========

    fn set_union(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use std::collections::HashSet;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        macro_rules! set_union_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let (Some(l), Some(r)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    let mut set: HashSet<Option<$val_type>> = HashSet::new();
                    for v in l.iter() { set.insert(v); }
                    for v in r.iter() { set.insert(v); }
                    let result: $arr_type = set.into_iter().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        // Integer types
        set_union_impl!(arrow_array::Int64Array, i64);
        set_union_impl!(arrow_array::Int32Array, i32);
        set_union_impl!(arrow_array::Int16Array, i16);
        set_union_impl!(arrow_array::Int8Array, i8);
        set_union_impl!(arrow_array::UInt64Array, u64);
        set_union_impl!(arrow_array::UInt32Array, u32);
        set_union_impl!(arrow_array::UInt16Array, u16);
        set_union_impl!(arrow_array::UInt8Array, u8);

        // Boolean
        set_union_impl!(arrow_array::BooleanArray, bool);

        // String arrays
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
        ) {
            let mut set: HashSet<Option<String>> = HashSet::new();
            for v in l.iter() { set.insert(v.map(|s| s.to_string())); }
            for v in r.iter() { set.insert(v.map(|s| s.to_string())); }
            let result: arrow_array::StringArray = set.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Date types (stored as i32/i64 internally)
        set_union_impl!(arrow_array::Date32Array, i32);
        set_union_impl!(arrow_array::Date64Array, i64);

        // LargeString arrays
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
        ) {
            let mut set: HashSet<Option<String>> = HashSet::new();
            for v in l.iter() { set.insert(v.map(|s| s.to_string())); }
            for v in r.iter() { set.insert(v.map(|s| s.to_string())); }
            let result: arrow_array::LargeStringArray = set.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::NotImplemented("set_union not implemented for this array type".to_string()))
    }

    fn set_intersection(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use std::collections::HashSet;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        macro_rules! set_intersection_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let (Some(l), Some(r)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    let left_set: HashSet<Option<$val_type>> = l.iter().collect();
                    let right_set: HashSet<Option<$val_type>> = r.iter().collect();
                    let result: $arr_type = left_set.intersection(&right_set).cloned().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        // Integer types
        set_intersection_impl!(arrow_array::Int64Array, i64);
        set_intersection_impl!(arrow_array::Int32Array, i32);
        set_intersection_impl!(arrow_array::Int16Array, i16);
        set_intersection_impl!(arrow_array::Int8Array, i8);
        set_intersection_impl!(arrow_array::UInt64Array, u64);
        set_intersection_impl!(arrow_array::UInt32Array, u32);
        set_intersection_impl!(arrow_array::UInt16Array, u16);
        set_intersection_impl!(arrow_array::UInt8Array, u8);

        // Boolean
        set_intersection_impl!(arrow_array::BooleanArray, bool);

        // String arrays
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
        ) {
            let left_set: HashSet<Option<String>> = l.iter().map(|v| v.map(|s| s.to_string())).collect();
            let right_set: HashSet<Option<String>> = r.iter().map(|v| v.map(|s| s.to_string())).collect();
            let result: arrow_array::StringArray = left_set.intersection(&right_set).cloned().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Date types (stored as i32/i64 internally)
        set_intersection_impl!(arrow_array::Date32Array, i32);
        set_intersection_impl!(arrow_array::Date64Array, i64);

        // LargeString arrays
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
        ) {
            let left_set: HashSet<Option<String>> = l.iter().map(|v| v.map(|s| s.to_string())).collect();
            let right_set: HashSet<Option<String>> = r.iter().map(|v| v.map(|s| s.to_string())).collect();
            let result: arrow_array::LargeStringArray = left_set.intersection(&right_set).cloned().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::NotImplemented("set_intersection not implemented for this array type".to_string()))
    }

    fn set_difference(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use std::collections::HashSet;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        macro_rules! set_difference_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let (Some(l), Some(r)) = (
                    left_impl.inner.as_any().downcast_ref::<$arr_type>(),
                    right_impl.inner.as_any().downcast_ref::<$arr_type>(),
                ) {
                    let left_set: HashSet<Option<$val_type>> = l.iter().collect();
                    let right_set: HashSet<Option<$val_type>> = r.iter().collect();
                    let result: $arr_type = left_set.difference(&right_set).cloned().collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        // Integer types
        set_difference_impl!(arrow_array::Int64Array, i64);
        set_difference_impl!(arrow_array::Int32Array, i32);
        set_difference_impl!(arrow_array::Int16Array, i16);
        set_difference_impl!(arrow_array::Int8Array, i8);
        set_difference_impl!(arrow_array::UInt64Array, u64);
        set_difference_impl!(arrow_array::UInt32Array, u32);
        set_difference_impl!(arrow_array::UInt16Array, u16);
        set_difference_impl!(arrow_array::UInt8Array, u8);

        // Boolean
        set_difference_impl!(arrow_array::BooleanArray, bool);

        // String arrays
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>(),
        ) {
            let left_set: HashSet<Option<String>> = l.iter().map(|v| v.map(|s| s.to_string())).collect();
            let right_set: HashSet<Option<String>> = r.iter().map(|v| v.map(|s| s.to_string())).collect();
            let result: arrow_array::StringArray = left_set.difference(&right_set).cloned().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Date types (stored as i32/i64 internally)
        set_difference_impl!(arrow_array::Date32Array, i32);
        set_difference_impl!(arrow_array::Date64Array, i64);

        // LargeString arrays
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>(),
        ) {
            let left_set: HashSet<Option<String>> = l.iter().map(|v| v.map(|s| s.to_string())).collect();
            let right_set: HashSet<Option<String>> = r.iter().map(|v| v.map(|s| s.to_string())).collect();
            let result: arrow_array::LargeStringArray = left_set.difference(&right_set).cloned().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::NotImplemented("set_difference not implemented for this array type".to_string()))
    }

    // ========== Grouping & Aggregation ==========

    fn group_indices(arr: arrays::ArrayBorrow<'_>) -> Result<(arrays::Array, Vec<Vec<u64>>), compute::ArrowError> {
        use std::collections::HashMap;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! group_indices_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut groups: HashMap<Option<$val_type>, Vec<u64>> = HashMap::new();
                    for (i, v) in typed_arr.iter().enumerate() {
                        groups.entry(v).or_default().push(i as u64);
                    }
                    let unique_values: Vec<Option<$val_type>> = groups.keys().cloned().collect();
                    let unique_arr: $arr_type = unique_values.iter().cloned().collect();
                    let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
                    return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
                }
            };
        }

        // Integer types
        group_indices_impl!(arrow_array::Int64Array, i64);
        group_indices_impl!(arrow_array::Int32Array, i32);
        group_indices_impl!(arrow_array::Int16Array, i16);
        group_indices_impl!(arrow_array::Int8Array, i8);
        group_indices_impl!(arrow_array::UInt64Array, u64);
        group_indices_impl!(arrow_array::UInt32Array, u32);
        group_indices_impl!(arrow_array::UInt16Array, u16);
        group_indices_impl!(arrow_array::UInt8Array, u8);

        // Boolean
        group_indices_impl!(arrow_array::BooleanArray, bool);

        // Date types
        group_indices_impl!(arrow_array::Date32Array, i32);
        group_indices_impl!(arrow_array::Date64Array, i64);

        // String arrays
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut groups: HashMap<Option<String>, Vec<u64>> = HashMap::new();
            for (i, v) in str_arr.iter().enumerate() {
                groups.entry(v.map(|s| s.to_string())).or_default().push(i as u64);
            }
            let unique_values: Vec<Option<String>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::StringArray = unique_values.iter().cloned().collect();
            let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        // LargeString arrays
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let mut groups: HashMap<Option<String>, Vec<u64>> = HashMap::new();
            for (i, v) in str_arr.iter().enumerate() {
                groups.entry(v.map(|s| s.to_string())).or_default().push(i as u64);
            }
            let unique_values: Vec<Option<String>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::LargeStringArray = unique_values.iter().cloned().collect();
            let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        // Timestamp types - use i64 for grouping (internal representation)
        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampSecondArray>() {
            let mut groups: HashMap<Option<i64>, Vec<u64>> = HashMap::new();
            for (i, v) in ts_arr.iter().enumerate() {
                groups.entry(v).or_default().push(i as u64);
            }
            let unique_values: Vec<Option<i64>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::TimestampSecondArray = unique_values.iter().cloned().collect();
            let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMillisecondArray>() {
            let mut groups: HashMap<Option<i64>, Vec<u64>> = HashMap::new();
            for (i, v) in ts_arr.iter().enumerate() {
                groups.entry(v).or_default().push(i as u64);
            }
            let unique_values: Vec<Option<i64>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::TimestampMillisecondArray = unique_values.iter().cloned().collect();
            let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>() {
            let mut groups: HashMap<Option<i64>, Vec<u64>> = HashMap::new();
            for (i, v) in ts_arr.iter().enumerate() {
                groups.entry(v).or_default().push(i as u64);
            }
            let unique_values: Vec<Option<i64>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::TimestampMicrosecondArray = unique_values.iter().cloned().collect();
            let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        if let Some(ts_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::TimestampNanosecondArray>() {
            let mut groups: HashMap<Option<i64>, Vec<u64>> = HashMap::new();
            for (i, v) in ts_arr.iter().enumerate() {
                groups.entry(v).or_default().push(i as u64);
            }
            let unique_values: Vec<Option<i64>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::TimestampNanosecondArray = unique_values.iter().cloned().collect();
            let indices: Vec<Vec<u64>> = unique_values.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        // Float types - use bits representation for hashing
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut groups: HashMap<Option<u64>, Vec<u64>> = HashMap::new();
            for (i, v) in float_arr.iter().enumerate() {
                let key = v.map(|f| f.to_bits());
                groups.entry(key).or_default().push(i as u64);
            }
            let unique_keys: Vec<Option<u64>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::Float64Array = unique_keys.iter()
                .map(|k| k.map(f64::from_bits))
                .collect();
            let indices: Vec<Vec<u64>> = unique_keys.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let mut groups: HashMap<Option<u32>, Vec<u64>> = HashMap::new();
            for (i, v) in float_arr.iter().enumerate() {
                let key = v.map(|f| f.to_bits());
                groups.entry(key).or_default().push(i as u64);
            }
            let unique_keys: Vec<Option<u32>> = groups.keys().cloned().collect();
            let unique_arr: arrow_array::Float32Array = unique_keys.iter()
                .map(|k| k.map(f32::from_bits))
                .collect();
            let indices: Vec<Vec<u64>> = unique_keys.iter().map(|k| groups.get(k).cloned().unwrap_or_default()).collect();
            return Ok((arrays::Array::new(ArrayImpl { inner: Arc::new(unique_arr) }), indices));
        }

        Err(compute::ArrowError::NotImplemented("group_indices not implemented for this array type".to_string()))
    }

    fn group_by(
        batch: record_batch::RecordBatchBorrow<'_>,
        group_columns: Vec<u32>,
        aggregates: Vec<compute::AggregateSpec>,
    ) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        use std::collections::HashMap;
        let batch_impl = batch.get::<RecordBatchImpl>();

        if group_columns.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("At least one group column required".to_string()));
        }

        // Build composite key for grouping
        let num_rows = batch_impl.inner.num_rows();
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row in 0..num_rows {
            let mut key = Vec::new();
            for &col_idx in &group_columns {
                let col = batch_impl.inner.column(col_idx as usize);
                // Convert value to string for key comparison
                let val = if col.is_null(row) {
                    "NULL".to_string()
                } else if let Some(str_arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                    str_arr.value(row).to_string()
                } else if let Some(int_arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                    int_arr.value(row).to_string()
                } else if let Some(int_arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                    int_arr.value(row).to_string()
                } else if let Some(float_arr) = col.as_any().downcast_ref::<arrow_array::Float64Array>() {
                    float_arr.value(row).to_string()
                } else {
                    format!("{:?}", row) // Fallback
                };
                key.push(val);
            }
            groups.entry(key).or_default().push(row);
        }

        // Build result columns
        let mut result_columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        let mut result_fields: Vec<arrow_schema::Field> = Vec::new();
        let schema = batch_impl.inner.schema();

        // Add group columns
        for &col_idx in &group_columns {
            let src_col = batch_impl.inner.column(col_idx as usize);
            let src_field = schema.field(col_idx as usize);

            // Extract first value from each group for group columns
            let indices: Vec<u64> = groups.values().map(|rows| rows[0] as u64).collect();
            let indices_arr = arrow_array::UInt64Array::from(indices);
            let taken = arrow_select::take::take(src_col.as_ref(), &indices_arr, None)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            result_columns.push(taken);
            result_fields.push(src_field.clone());
        }

        // Add aggregate columns
        for agg in &aggregates {
            let src_col = batch_impl.inner.column(agg.column_index as usize);

            let values: Vec<Option<f64>> = groups.values().map(|rows| {
                aggregate_values(src_col.as_ref(), rows, &agg.function)
            }).collect();

            let result_arr: arrow_array::Float64Array = values.into_iter().collect();
            result_columns.push(Arc::new(result_arr));
            result_fields.push(arrow_schema::Field::new(&agg.output_name, arrow_schema::DataType::Float64, true));
        }

        let schema = Arc::new(arrow_schema::Schema::new(result_fields));
        let result = ArrowRecordBatch::try_new(schema, result_columns)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    // ========== Join Operations ==========

    fn join(
        left: record_batch::RecordBatchBorrow<'_>,
        right: record_batch::RecordBatchBorrow<'_>,
        left_on: Vec<String>,
        right_on: Vec<String>,
        join_type: compute::JoinType,
    ) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        use std::collections::HashMap;
        use compute::JoinType;

        let left_impl = left.get::<RecordBatchImpl>();
        let right_impl = right.get::<RecordBatchImpl>();

        if left_on.len() != right_on.len() || left_on.is_empty() {
            return Err(compute::ArrowError::InvalidArgument(
                "Join columns must have equal non-zero length".to_string()
            ));
        }

        // Get column indices
        let left_indices: Vec<usize> = left_on.iter().map(|name| {
            left_impl.inner.schema().index_of(name)
                .map_err(|e| compute::ArrowError::InvalidArgument(e.to_string()))
        }).collect::<Result<Vec<_>, _>>()?;

        let right_indices: Vec<usize> = right_on.iter().map(|name| {
            right_impl.inner.schema().index_of(name)
                .map_err(|e| compute::ArrowError::InvalidArgument(e.to_string()))
        }).collect::<Result<Vec<_>, _>>()?;

        // Build hash map from right side
        let mut right_map: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for row in 0..right_impl.inner.num_rows() {
            let key = build_join_key(&right_impl.inner, row, &right_indices);
            right_map.entry(key).or_default().push(row);
        }

        // Build result indices
        let mut left_result_indices: Vec<Option<u64>> = Vec::new();
        let mut right_result_indices: Vec<Option<u64>> = Vec::new();

        for left_row in 0..left_impl.inner.num_rows() {
            let key = build_join_key(&left_impl.inner, left_row, &left_indices);
            let right_matches = right_map.get(&key);

            match (&join_type, right_matches) {
                (JoinType::Inner, Some(matches)) => {
                    for &right_row in matches {
                        left_result_indices.push(Some(left_row as u64));
                        right_result_indices.push(Some(right_row as u64));
                    }
                }
                (JoinType::Left, Some(matches)) => {
                    for &right_row in matches {
                        left_result_indices.push(Some(left_row as u64));
                        right_result_indices.push(Some(right_row as u64));
                    }
                }
                (JoinType::Left, None) => {
                    left_result_indices.push(Some(left_row as u64));
                    right_result_indices.push(None);
                }
                (JoinType::LeftSemi, Some(_)) => {
                    left_result_indices.push(Some(left_row as u64));
                }
                (JoinType::LeftAnti, None) => {
                    left_result_indices.push(Some(left_row as u64));
                }
                (JoinType::Full, Some(matches)) => {
                    for &right_row in matches {
                        left_result_indices.push(Some(left_row as u64));
                        right_result_indices.push(Some(right_row as u64));
                    }
                }
                (JoinType::Full, None) => {
                    left_result_indices.push(Some(left_row as u64));
                    right_result_indices.push(None);
                }
                _ => {}
            }
        }

        // For right join, add unmatched right rows
        if matches!(join_type, JoinType::Right | JoinType::Full) {
            let mut matched_right: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for left_row in 0..left_impl.inner.num_rows() {
                let key = build_join_key(&left_impl.inner, left_row, &left_indices);
                if let Some(matches) = right_map.get(&key) {
                    for &right_row in matches {
                        matched_right.insert(right_row);
                    }
                }
            }

            for right_row in 0..right_impl.inner.num_rows() {
                if !matched_right.contains(&right_row) {
                    left_result_indices.push(None);
                    right_result_indices.push(Some(right_row as u64));
                }
            }
        }

        // Build result batch
        build_join_result(&left_impl.inner, &right_impl.inner, &left_result_indices, &right_result_indices, &join_type)
    }

    fn cross_join(
        left: record_batch::RecordBatchBorrow<'_>,
        right: record_batch::RecordBatchBorrow<'_>,
    ) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        let left_impl = left.get::<RecordBatchImpl>();
        let right_impl = right.get::<RecordBatchImpl>();

        let left_rows = left_impl.inner.num_rows();
        let right_rows = right_impl.inner.num_rows();

        // Build indices for cartesian product
        let mut left_indices: Vec<Option<u64>> = Vec::with_capacity(left_rows * right_rows);
        let mut right_indices: Vec<Option<u64>> = Vec::with_capacity(left_rows * right_rows);

        for l in 0..left_rows {
            for r in 0..right_rows {
                left_indices.push(Some(l as u64));
                right_indices.push(Some(r as u64));
            }
        }

        build_join_result(&left_impl.inner, &right_impl.inner, &left_indices, &right_indices, &compute::JoinType::Inner)
    }

    // ========== Arrow-Row Operations ==========

    fn row_distinct(
        batch: record_batch::RecordBatchBorrow<'_>,
        columns: Vec<String>,
    ) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        use arrow_row::{RowConverter, SortField};
        use std::collections::HashSet;

        let batch_impl = batch.get::<RecordBatchImpl>();
        let inner = &batch_impl.inner;

        if columns.is_empty() {
            return Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: inner.clone() }));
        }

        // Get the columns to use for distinct checking
        let sort_fields: Vec<SortField> = columns.iter()
            .filter_map(|name| {
                inner.schema().field_with_name(name).ok()
                    .map(|field| SortField::new(field.data_type().clone()))
            })
            .collect();

        if sort_fields.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("No valid columns found".to_string()));
        }

        // Get column arrays
        let arrays_for_rows: Vec<ArrayRef> = columns.iter()
            .filter_map(|name| inner.column_by_name(name).cloned())
            .collect();

        // Convert to row format
        let converter = RowConverter::new(sort_fields)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        let rows = converter.convert_columns(&arrays_for_rows)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Find distinct rows
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut indices: Vec<u64> = Vec::new();

        for (i, row) in rows.iter().enumerate() {
            let row_bytes = row.as_ref().to_vec();
            if seen.insert(row_bytes) {
                indices.push(i as u64);
            }
        }

        // Take the distinct rows
        let indices_arr: arrow_array::UInt64Array = indices.into_iter().map(Some).collect();
        let columns_result: Result<Vec<ArrayRef>, _> = inner.columns().iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &indices_arr, None))
            .collect();
        let columns_result = columns_result
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        let result = ArrowRecordBatch::try_new(inner.schema().clone(), columns_result)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn row_deduplicate(
        batch: record_batch::RecordBatchBorrow<'_>,
        columns: Vec<String>,
    ) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        // row_deduplicate is the same as row_distinct - it removes duplicates
        // preserving first occurrence
        Self::row_distinct(batch, columns)
    }

    // ========== Additional Statistics ==========

    fn mode(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use std::collections::HashMap;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! mode_impl {
            ($arr_type:ty, $val_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut counts: HashMap<$val_type, usize> = HashMap::new();
                    for v in typed_arr.iter().flatten() {
                        *counts.entry(v).or_insert(0) += 1;
                    }
                    let max_count = counts.values().cloned().max().unwrap_or(0);
                    let modes: Vec<$val_type> = counts.into_iter().filter(|(_, c)| *c == max_count).map(|(v, _)| v).collect();
                    let result: $arr_type = modes.into_iter().map(Some).collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        // Integer types
        mode_impl!(arrow_array::Int64Array, i64);
        mode_impl!(arrow_array::Int32Array, i32);
        mode_impl!(arrow_array::Int16Array, i16);
        mode_impl!(arrow_array::Int8Array, i8);
        mode_impl!(arrow_array::UInt64Array, u64);
        mode_impl!(arrow_array::UInt32Array, u32);
        mode_impl!(arrow_array::UInt16Array, u16);
        mode_impl!(arrow_array::UInt8Array, u8);

        // Boolean
        mode_impl!(arrow_array::BooleanArray, bool);

        // Float64 arrays - treat similar values as same (rounded)
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut counts: HashMap<i64, (f64, usize)> = HashMap::new();
            for v in float_arr.iter().flatten() {
                let key = (v * 1e10) as i64; // Round to 10 decimal places for comparison
                counts.entry(key).or_insert((v, 0)).1 += 1;
            }
            let max_count = counts.values().map(|(_, c)| *c).max().unwrap_or(0);
            let modes: Vec<f64> = counts.into_iter().filter(|(_, (_, c))| *c == max_count).map(|(_, (v, _))| v).collect();
            let result: arrow_array::Float64Array = modes.into_iter().map(Some).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // Float32 arrays
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float32Array>() {
            let mut counts: HashMap<i32, (f32, usize)> = HashMap::new();
            for v in float_arr.iter().flatten() {
                let key = (v * 1e6) as i32; // Round to 6 decimal places for comparison
                counts.entry(key).or_insert((v, 0)).1 += 1;
            }
            let max_count = counts.values().map(|(_, c)| *c).max().unwrap_or(0);
            let modes: Vec<f32> = counts.into_iter().filter(|(_, (_, c))| *c == max_count).map(|(_, (v, _))| v).collect();
            let result: arrow_array::Float32Array = modes.into_iter().map(Some).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        // String arrays
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for v in str_arr.iter().flatten() {
                *counts.entry(v.to_string()).or_insert(0) += 1;
            }
            let max_count = counts.values().cloned().max().unwrap_or(0);
            let modes: Vec<String> = counts.into_iter().filter(|(_, c)| *c == max_count).map(|(v, _)| v).collect();
            let result: arrow_array::StringArray = modes.into_iter().map(Some).collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }

        Err(compute::ArrowError::NotImplemented("mode not implemented for this array type".to_string()))
    }

    fn quantile(arr: arrays::ArrayBorrow<'_>, q: f64) -> Result<Option<f64>, compute::ArrowError> {
        if q < 0.0 || q > 1.0 {
            return Err(compute::ArrowError::InvalidArgument("Quantile must be between 0.0 and 1.0".to_string()));
        }

        let values = extract_float64_values(&arr)?;
        if values.is_empty() {
            return Ok(None);
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = q * (sorted.len() - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        let frac = idx - lower as f64;

        if lower == upper {
            Ok(Some(sorted[lower]))
        } else {
            Ok(Some(sorted[lower] * (1.0 - frac) + sorted[upper] * frac))
        }
    }

    fn quantiles(arr: arrays::ArrayBorrow<'_>, qs: Vec<f64>) -> Result<Vec<Option<f64>>, compute::ArrowError> {
        let values = extract_float64_values(&arr)?;
        if values.is_empty() {
            return Ok(qs.iter().map(|_| None).collect());
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Ok(qs.iter().map(|&q| {
            if q < 0.0 || q > 1.0 {
                None
            } else {
                let idx = q * (sorted.len() - 1) as f64;
                let lower = idx.floor() as usize;
                let upper = idx.ceil() as usize;
                let frac = idx - lower as f64;
                if lower == upper {
                    Some(sorted[lower])
                } else {
                    Some(sorted[lower] * (1.0 - frac) + sorted[upper] * frac)
                }
            }
        }).collect())
    }

    fn iqr(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let values = extract_float64_values(&arr)?;
        if values.is_empty() {
            return Ok(None);
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = 0.25 * (sorted.len() - 1) as f64;
        let q3_idx = 0.75 * (sorted.len() - 1) as f64;

        let q1 = interpolate_quantile(&sorted, q1_idx);
        let q3 = interpolate_quantile(&sorted, q3_idx);

        Ok(Some(q3 - q1))
    }

    fn skewness(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let values = extract_float64_values(&arr)?;
        if values.len() < 3 {
            return Ok(None);
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std == 0.0 {
            return Ok(Some(0.0));
        }

        let m3 = values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
        Ok(Some(m3))
    }

    fn kurtosis(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let values = extract_float64_values(&arr)?;
        if values.len() < 4 {
            return Ok(None);
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std == 0.0 {
            return Ok(Some(0.0));
        }

        let m4 = values.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
        Ok(Some(m4 - 3.0)) // Excess kurtosis (normal = 0)
    }

    fn covariance(x: arrays::ArrayBorrow<'_>, y: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let x_vals = extract_float64_values(&x)?;
        let y_vals = extract_float64_values(&y)?;

        if x_vals.len() != y_vals.len() || x_vals.len() < 2 {
            return Ok(None);
        }

        let n = x_vals.len() as f64;
        let x_mean = x_vals.iter().sum::<f64>() / n;
        let y_mean = y_vals.iter().sum::<f64>() / n;

        let cov = x_vals.iter().zip(y_vals.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f64>() / (n - 1.0);

        Ok(Some(cov))
    }

    fn correlation(x: arrays::ArrayBorrow<'_>, y: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let x_vals = extract_float64_values(&x)?;
        let y_vals = extract_float64_values(&y)?;

        if x_vals.len() != y_vals.len() || x_vals.len() < 2 {
            return Ok(None);
        }

        let n = x_vals.len() as f64;
        let x_mean = x_vals.iter().sum::<f64>() / n;
        let y_mean = y_vals.iter().sum::<f64>() / n;

        let cov = x_vals.iter().zip(y_vals.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f64>();

        let x_var = x_vals.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>();
        let y_var = y_vals.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>();

        if x_var == 0.0 || y_var == 0.0 {
            return Ok(None);
        }

        Ok(Some(cov / (x_var.sqrt() * y_var.sqrt())))
    }

    // ========== Extended Statistical Functions ==========

    fn index_of_max(arr: arrays::ArrayBorrow<'_>) -> Result<Option<u64>, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! index_of_max_impl {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    if typed_arr.len() == 0 {
                        return Ok(None);
                    }
                    let mut max_idx = None;
                    let mut max_val = None;
                    for (i, opt) in typed_arr.iter().enumerate() {
                        if let Some(v) = opt {
                            match max_val {
                                None => {
                                    max_val = Some(v);
                                    max_idx = Some(i as u64);
                                }
                                Some(mv) => {
                                    if v > mv {
                                        max_val = Some(v);
                                        max_idx = Some(i as u64);
                                    }
                                }
                            }
                        }
                    }
                    return Ok(max_idx);
                }
            };
        }

        index_of_max_impl!(arrow_array::Float64Array);
        index_of_max_impl!(arrow_array::Float32Array);
        index_of_max_impl!(arrow_array::Int64Array);
        index_of_max_impl!(arrow_array::Int32Array);
        index_of_max_impl!(arrow_array::Int16Array);
        index_of_max_impl!(arrow_array::Int8Array);
        index_of_max_impl!(arrow_array::UInt64Array);
        index_of_max_impl!(arrow_array::UInt32Array);
        index_of_max_impl!(arrow_array::UInt16Array);
        index_of_max_impl!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("index_of_max requires a numeric array".to_string()))
    }

    fn index_of_min(arr: arrays::ArrayBorrow<'_>) -> Result<Option<u64>, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! index_of_min_impl {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    if typed_arr.len() == 0 {
                        return Ok(None);
                    }
                    let mut min_idx = None;
                    let mut min_val = None;
                    for (i, opt) in typed_arr.iter().enumerate() {
                        if let Some(v) = opt {
                            match min_val {
                                None => {
                                    min_val = Some(v);
                                    min_idx = Some(i as u64);
                                }
                                Some(mv) => {
                                    if v < mv {
                                        min_val = Some(v);
                                        min_idx = Some(i as u64);
                                    }
                                }
                            }
                        }
                    }
                    return Ok(min_idx);
                }
            };
        }

        index_of_min_impl!(arrow_array::Float64Array);
        index_of_min_impl!(arrow_array::Float32Array);
        index_of_min_impl!(arrow_array::Int64Array);
        index_of_min_impl!(arrow_array::Int32Array);
        index_of_min_impl!(arrow_array::Int16Array);
        index_of_min_impl!(arrow_array::Int8Array);
        index_of_min_impl!(arrow_array::UInt64Array);
        index_of_min_impl!(arrow_array::UInt32Array);
        index_of_min_impl!(arrow_array::UInt16Array);
        index_of_min_impl!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("index_of_min requires a numeric array".to_string()))
    }

    fn is_monotonic_increasing(arr: arrays::ArrayBorrow<'_>) -> Result<bool, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! monotonic_increasing_impl {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    if typed_arr.len() < 2 {
                        return Ok(true);
                    }
                    let mut prev = None;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            if let Some(p) = prev {
                                if v < p {
                                    return Ok(false);
                                }
                            }
                            prev = Some(v);
                        }
                    }
                    return Ok(true);
                }
            };
        }

        monotonic_increasing_impl!(arrow_array::Float64Array);
        monotonic_increasing_impl!(arrow_array::Float32Array);
        monotonic_increasing_impl!(arrow_array::Int64Array);
        monotonic_increasing_impl!(arrow_array::Int32Array);
        monotonic_increasing_impl!(arrow_array::Int16Array);
        monotonic_increasing_impl!(arrow_array::Int8Array);
        monotonic_increasing_impl!(arrow_array::UInt64Array);
        monotonic_increasing_impl!(arrow_array::UInt32Array);
        monotonic_increasing_impl!(arrow_array::UInt16Array);
        monotonic_increasing_impl!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("is_monotonic_increasing requires a numeric array".to_string()))
    }

    fn is_monotonic_decreasing(arr: arrays::ArrayBorrow<'_>) -> Result<bool, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! monotonic_decreasing_impl {
            ($arr_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    if typed_arr.len() < 2 {
                        return Ok(true);
                    }
                    let mut prev = None;
                    for opt in typed_arr.iter() {
                        if let Some(v) = opt {
                            if let Some(p) = prev {
                                if v > p {
                                    return Ok(false);
                                }
                            }
                            prev = Some(v);
                        }
                    }
                    return Ok(true);
                }
            };
        }

        monotonic_decreasing_impl!(arrow_array::Float64Array);
        monotonic_decreasing_impl!(arrow_array::Float32Array);
        monotonic_decreasing_impl!(arrow_array::Int64Array);
        monotonic_decreasing_impl!(arrow_array::Int32Array);
        monotonic_decreasing_impl!(arrow_array::Int16Array);
        monotonic_decreasing_impl!(arrow_array::Int8Array);
        monotonic_decreasing_impl!(arrow_array::UInt64Array);
        monotonic_decreasing_impl!(arrow_array::UInt32Array);
        monotonic_decreasing_impl!(arrow_array::UInt16Array);
        monotonic_decreasing_impl!(arrow_array::UInt8Array);

        Err(compute::ArrowError::InvalidArgument("is_monotonic_decreasing requires a numeric array".to_string()))
    }

    fn top_n(arr: arrays::ArrayBorrow<'_>, n: u64) -> Result<arrays::Array, compute::ArrowError> {
        // Use existing sort_limit with descending order
        let options = compute::SortOptions {
            descending: true,
            nulls_first: false,
        };
        Self::sort_limit(arr, options, n)
    }

    fn bottom_n(arr: arrays::ArrayBorrow<'_>, n: u64) -> Result<arrays::Array, compute::ArrowError> {
        // Use existing sort_limit with ascending order
        let options = compute::SortOptions {
            descending: false,
            nulls_first: false,
        };
        Self::sort_limit(arr, options, n)
    }

    fn top_n_indices(arr: arrays::ArrayBorrow<'_>, n: u64) -> Result<arrays::Array, compute::ArrowError> {
        // Use existing sort_indices_limit with descending order
        let options = compute::SortOptions {
            descending: true,
            nulls_first: false,
        };
        Self::sort_indices_limit(arr, options, n)
    }

    fn bottom_n_indices(arr: arrays::ArrayBorrow<'_>, n: u64) -> Result<arrays::Array, compute::ArrowError> {
        // Use existing sort_indices_limit with ascending order
        let options = compute::SortOptions {
            descending: false,
            nulls_first: false,
        };
        Self::sort_indices_limit(arr, options, n)
    }

    fn entropy(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        if arr_impl.inner.len() == 0 {
            return Ok(None);
        }

        // Extract values and count occurrences manually
        let values = extract_float64_values(&arr)?;
        if values.is_empty() {
            return Ok(None);
        }

        // Count occurrences using a HashMap
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for v in &values {
            // Convert float to bits for hashing (handles exact equality)
            let key = v.to_bits();
            *counts.entry(key).or_insert(0) += 1;
        }

        let total = values.len() as f64;

        // Calculate Shannon entropy: -sum(p * log2(p))
        let entropy: f64 = counts.values()
            .filter_map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 {
                    Some(-p * p.log2())
                } else {
                    None
                }
            })
            .sum();

        Ok(Some(entropy))
    }

    fn histogram(arr: arrays::ArrayBorrow<'_>, bins: u32) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        if bins == 0 {
            return Err(compute::ArrowError::InvalidArgument("bins must be greater than 0".to_string()));
        }

        // Extract float values
        let values = extract_float64_values(&arr)?;
        if values.is_empty() {
            // Return empty histogram
            let schema = Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("bin_min", arrow_schema::DataType::Float64, false),
                arrow_schema::Field::new("bin_max", arrow_schema::DataType::Float64, false),
                arrow_schema::Field::new("count", arrow_schema::DataType::UInt64, false),
            ]));
            let batch = ArrowRecordBatch::try_new(schema, vec![
                Arc::new(arrow_array::Float64Array::from(Vec::<f64>::new())),
                Arc::new(arrow_array::Float64Array::from(Vec::<f64>::new())),
                Arc::new(arrow_array::UInt64Array::from(Vec::<u64>::new())),
            ]).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }));
        }

        // Find min/max
        let min_val = values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_val = values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        // Handle case where all values are the same
        let (bin_width, actual_min) = if min_val == max_val {
            (1.0, min_val - 0.5)
        } else {
            ((max_val - min_val) / bins as f64, min_val)
        };

        // Initialize bin counts
        let mut counts = vec![0u64; bins as usize];

        // Bin the values
        for &v in &values {
            let bin_idx = if v == max_val && min_val != max_val {
                bins as usize - 1 // Include max value in last bin
            } else {
                let idx = ((v - actual_min) / bin_width) as usize;
                idx.min(bins as usize - 1)
            };
            counts[bin_idx] += 1;
        }

        // Create arrays
        let bin_mins: Vec<f64> = (0..bins).map(|i| actual_min + (i as f64) * bin_width).collect();
        let bin_maxs: Vec<f64> = (0..bins).map(|i| actual_min + ((i + 1) as f64) * bin_width).collect();

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("bin_min", arrow_schema::DataType::Float64, false),
            arrow_schema::Field::new("bin_max", arrow_schema::DataType::Float64, false),
            arrow_schema::Field::new("count", arrow_schema::DataType::UInt64, false),
        ]));

        let batch = ArrowRecordBatch::try_new(schema, vec![
            Arc::new(arrow_array::Float64Array::from(bin_mins)),
            Arc::new(arrow_array::Float64Array::from(bin_maxs)),
            Arc::new(arrow_array::UInt64Array::from(counts)),
        ]).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }))
    }

    fn rolling_sum(arr: arrays::ArrayBorrow<'_>, options: compute::RollingOptions) -> Result<arrays::Array, compute::ArrowError> {
        rolling_agg(&arr, &options, |window| window.iter().sum())
    }

    fn rolling_mean(arr: arrays::ArrayBorrow<'_>, options: compute::RollingOptions) -> Result<arrays::Array, compute::ArrowError> {
        rolling_agg(&arr, &options, |window| window.iter().sum::<f64>() / window.len() as f64)
    }

    fn rolling_min(arr: arrays::ArrayBorrow<'_>, options: compute::RollingOptions) -> Result<arrays::Array, compute::ArrowError> {
        rolling_agg(&arr, &options, |window| {
            window.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(f64::NAN)
        })
    }

    fn rolling_max(arr: arrays::ArrayBorrow<'_>, options: compute::RollingOptions) -> Result<arrays::Array, compute::ArrowError> {
        rolling_agg(&arr, &options, |window| {
            window.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(f64::NAN)
        })
    }

    fn rolling_std(arr: arrays::ArrayBorrow<'_>, options: compute::RollingOptions) -> Result<arrays::Array, compute::ArrowError> {
        rolling_agg(&arr, &options, |window| {
            if window.len() < 2 {
                return f64::NAN;
            }
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64;
            variance.sqrt()
        })
    }

    // ========== Cumulative/Scan Operations ==========

    fn cumulative_sum(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        // Handle different numeric types with appropriate precision
        macro_rules! cumulative_sum_impl {
            ($arr_type:ty, $result_type:ty, $acc_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut acc: $acc_type = Default::default();
                    let result: $result_type = typed_arr.iter()
                        .map(|opt| {
                            match opt {
                                Some(v) => {
                                    acc += v as $acc_type;
                                    Some(acc)
                                }
                                None => Some(acc), // Nulls don't change the running sum
                            }
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        // Integer types -> Int64 for precision
        cumulative_sum_impl!(arrow_array::Int8Array, arrow_array::Int64Array, i64);
        cumulative_sum_impl!(arrow_array::Int16Array, arrow_array::Int64Array, i64);
        cumulative_sum_impl!(arrow_array::Int32Array, arrow_array::Int64Array, i64);
        cumulative_sum_impl!(arrow_array::Int64Array, arrow_array::Int64Array, i64);

        // Unsigned types -> UInt64 for precision
        cumulative_sum_impl!(arrow_array::UInt8Array, arrow_array::UInt64Array, u64);
        cumulative_sum_impl!(arrow_array::UInt16Array, arrow_array::UInt64Array, u64);
        cumulative_sum_impl!(arrow_array::UInt32Array, arrow_array::UInt64Array, u64);
        cumulative_sum_impl!(arrow_array::UInt64Array, arrow_array::UInt64Array, u64);

        // Float types -> Float64
        cumulative_sum_impl!(arrow_array::Float32Array, arrow_array::Float64Array, f64);
        cumulative_sum_impl!(arrow_array::Float64Array, arrow_array::Float64Array, f64);

        Err(compute::ArrowError::InvalidArgument(
            "cumulative_sum requires numeric array (Int8-64, UInt8-64, Float32/64)".to_string()
        ))
    }

    fn cumulative_prod(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! cumulative_prod_impl {
            ($arr_type:ty, $result_type:ty, $acc_type:ty, $identity:expr) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut acc: $acc_type = $identity;
                    let result: $result_type = typed_arr.iter()
                        .map(|opt| {
                            match opt {
                                Some(v) => {
                                    acc *= v as $acc_type;
                                    Some(acc)
                                }
                                None => Some(acc), // Nulls don't change the running product
                            }
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        // Integer types -> Int64 for precision
        cumulative_prod_impl!(arrow_array::Int8Array, arrow_array::Int64Array, i64, 1i64);
        cumulative_prod_impl!(arrow_array::Int16Array, arrow_array::Int64Array, i64, 1i64);
        cumulative_prod_impl!(arrow_array::Int32Array, arrow_array::Int64Array, i64, 1i64);
        cumulative_prod_impl!(arrow_array::Int64Array, arrow_array::Int64Array, i64, 1i64);

        // Unsigned types -> UInt64 for precision
        cumulative_prod_impl!(arrow_array::UInt8Array, arrow_array::UInt64Array, u64, 1u64);
        cumulative_prod_impl!(arrow_array::UInt16Array, arrow_array::UInt64Array, u64, 1u64);
        cumulative_prod_impl!(arrow_array::UInt32Array, arrow_array::UInt64Array, u64, 1u64);
        cumulative_prod_impl!(arrow_array::UInt64Array, arrow_array::UInt64Array, u64, 1u64);

        // Float types -> Float64
        cumulative_prod_impl!(arrow_array::Float32Array, arrow_array::Float64Array, f64, 1.0f64);
        cumulative_prod_impl!(arrow_array::Float64Array, arrow_array::Float64Array, f64, 1.0f64);

        Err(compute::ArrowError::InvalidArgument(
            "cumulative_prod requires numeric array (Int8-64, UInt8-64, Float32/64)".to_string()
        ))
    }

    fn cumulative_min(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! cumulative_min_impl {
            ($arr_type:ty, $result_type:ty, $native_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut min_val: Option<$native_type> = None;
                    let result: $result_type = typed_arr.iter()
                        .map(|opt| {
                            match opt {
                                Some(v) => {
                                    min_val = Some(match min_val {
                                        Some(m) => if v < m { v } else { m },
                                        None => v,
                                    });
                                    min_val
                                }
                                None => min_val, // Return current min on null
                            }
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        cumulative_min_impl!(arrow_array::Int8Array, arrow_array::Int8Array, i8);
        cumulative_min_impl!(arrow_array::Int16Array, arrow_array::Int16Array, i16);
        cumulative_min_impl!(arrow_array::Int32Array, arrow_array::Int32Array, i32);
        cumulative_min_impl!(arrow_array::Int64Array, arrow_array::Int64Array, i64);
        cumulative_min_impl!(arrow_array::UInt8Array, arrow_array::UInt8Array, u8);
        cumulative_min_impl!(arrow_array::UInt16Array, arrow_array::UInt16Array, u16);
        cumulative_min_impl!(arrow_array::UInt32Array, arrow_array::UInt32Array, u32);
        cumulative_min_impl!(arrow_array::UInt64Array, arrow_array::UInt64Array, u64);
        cumulative_min_impl!(arrow_array::Float32Array, arrow_array::Float32Array, f32);
        cumulative_min_impl!(arrow_array::Float64Array, arrow_array::Float64Array, f64);

        Err(compute::ArrowError::InvalidArgument(
            "cumulative_min requires numeric array (Int8-64, UInt8-64, Float32/64)".to_string()
        ))
    }

    fn cumulative_max(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();

        macro_rules! cumulative_max_impl {
            ($arr_type:ty, $result_type:ty, $native_type:ty) => {
                if let Some(typed_arr) = arr_impl.inner.as_any().downcast_ref::<$arr_type>() {
                    let mut max_val: Option<$native_type> = None;
                    let result: $result_type = typed_arr.iter()
                        .map(|opt| {
                            match opt {
                                Some(v) => {
                                    max_val = Some(match max_val {
                                        Some(m) => if v > m { v } else { m },
                                        None => v,
                                    });
                                    max_val
                                }
                                None => max_val, // Return current max on null
                            }
                        })
                        .collect();
                    return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
                }
            };
        }

        cumulative_max_impl!(arrow_array::Int8Array, arrow_array::Int8Array, i8);
        cumulative_max_impl!(arrow_array::Int16Array, arrow_array::Int16Array, i16);
        cumulative_max_impl!(arrow_array::Int32Array, arrow_array::Int32Array, i32);
        cumulative_max_impl!(arrow_array::Int64Array, arrow_array::Int64Array, i64);
        cumulative_max_impl!(arrow_array::UInt8Array, arrow_array::UInt8Array, u8);
        cumulative_max_impl!(arrow_array::UInt16Array, arrow_array::UInt16Array, u16);
        cumulative_max_impl!(arrow_array::UInt32Array, arrow_array::UInt32Array, u32);
        cumulative_max_impl!(arrow_array::UInt64Array, arrow_array::UInt64Array, u64);
        cumulative_max_impl!(arrow_array::Float32Array, arrow_array::Float32Array, f32);
        cumulative_max_impl!(arrow_array::Float64Array, arrow_array::Float64Array, f64);

        Err(compute::ArrowError::InvalidArgument(
            "cumulative_max requires numeric array (Int8-64, UInt8-64, Float32/64)".to_string()
        ))
    }

    fn cumulative_count(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let mut count: u64 = 0;
        let result: arrow_array::UInt64Array = (0..len)
            .map(|i| {
                if arr_impl.inner.is_valid(i) {
                    count += 1;
                }
                Some(count)
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    // ========== String Distance Functions ==========

    fn levenshtein(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("levenshtein requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("levenshtein requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::UInt32Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) => Some(compute_levenshtein(l, r) as u32),
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn levenshtein_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("levenshtein_scalar requires string array".to_string()))?;

        let result: arrow_array::UInt32Array = str_arr.iter()
            .map(|opt| opt.map(|s| compute_levenshtein(s, &pattern) as u32))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn jaro(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("jaro requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("jaro requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::Float64Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) => Some(compute_jaro(l, r)),
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn jaro_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("jaro_scalar requires string array".to_string()))?;

        let result: arrow_array::Float64Array = str_arr.iter()
            .map(|opt| opt.map(|s| compute_jaro(s, &pattern)))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn jaro_winkler(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("jaro_winkler requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("jaro_winkler requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::Float64Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) => Some(compute_jaro_winkler(l, r)),
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn jaro_winkler_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("jaro_winkler_scalar requires string array".to_string()))?;

        let result: arrow_array::Float64Array = str_arr.iter()
            .map(|opt| opt.map(|s| compute_jaro_winkler(s, &pattern)))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn soundex(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("soundex requires string array".to_string()))?;

        let result: arrow_array::StringArray = str_arr.iter()
            .map(|opt| opt.map(compute_soundex))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn hamming(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("hamming requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("hamming requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::UInt32Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) if l.len() == r.len() => {
                    Some(l.chars().zip(r.chars()).filter(|(a, b)| a != b).count() as u32)
                },
                (Some(_), Some(_)) => None, // Different lengths - return null
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn hamming_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("hamming_scalar requires string array".to_string()))?;

        let pattern_len = pattern.len();
        let result: arrow_array::UInt32Array = str_arr.iter()
            .map(|opt| match opt {
                Some(s) if s.len() == pattern_len => {
                    Some(s.chars().zip(pattern.chars()).filter(|(a, b)| a != b).count() as u32)
                },
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn damerau_levenshtein(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("damerau_levenshtein requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("damerau_levenshtein requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::UInt32Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) => Some(compute_damerau_levenshtein(l, r) as u32),
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn damerau_levenshtein_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("damerau_levenshtein_scalar requires string array".to_string()))?;

        let result: arrow_array::UInt32Array = str_arr.iter()
            .map(|opt| opt.map(|s| compute_damerau_levenshtein(s, &pattern) as u32))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn lcs_length(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("lcs_length requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("lcs_length requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::UInt32Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) => Some(compute_lcs_length(l, r) as u32),
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn lcs_length_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("lcs_length_scalar requires string array".to_string()))?;

        let result: arrow_array::UInt32Array = str_arr.iter()
            .map(|opt| opt.map(|s| compute_lcs_length(s, &pattern) as u32))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn normalized_levenshtein(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();

        let left_arr = left_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("normalized_levenshtein requires string arrays".to_string()))?;
        let right_arr = right_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("normalized_levenshtein requires string arrays".to_string()))?;

        if left_arr.len() != right_arr.len() {
            return Err(compute::ArrowError::InvalidArgument("Arrays must have the same length".to_string()));
        }

        let result: arrow_array::Float64Array = left_arr.iter().zip(right_arr.iter())
            .map(|(l, r)| match (l, r) {
                (Some(l), Some(r)) => {
                    let max_len = l.len().max(r.len());
                    if max_len == 0 {
                        Some(0.0)
                    } else {
                        Some(compute_levenshtein(l, r) as f64 / max_len as f64)
                    }
                },
                _ => None,
            })
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn normalized_levenshtein_scalar(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let str_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| compute::ArrowError::InvalidArgument("normalized_levenshtein_scalar requires string array".to_string()))?;

        let pattern_len = pattern.len();
        let result: arrow_array::Float64Array = str_arr.iter()
            .map(|opt| opt.map(|s| {
                let max_len = s.len().max(pattern_len);
                if max_len == 0 {
                    0.0
                } else {
                    compute_levenshtein(s, &pattern) as f64 / max_len as f64
                }
            }))
            .collect();

        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    // ========== Array Operations ==========

    fn concat(arrays: Vec<arrays::Array>) -> Result<arrays::Array, compute::ArrowError> {
        if arrays.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("Cannot concat empty array list".to_string()));
        }

        let arr_refs: Vec<&dyn arrow_array::Array> = arrays.iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();

        let result = arrow_select::concat::concat(&arr_refs)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn concat_batches(batches: Vec<record_batch::RecordBatch>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        if batches.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("Cannot concat empty batch list".to_string()));
        }

        let first = batches[0].get::<RecordBatchImpl>();
        let schema = first.inner.schema();

        let batch_refs: Vec<&ArrowRecordBatch> = batches.iter()
            .map(|b| &b.get::<RecordBatchImpl>().inner)
            .collect();

        let result = arrow_select::concat::concat_batches(&schema, batch_refs.into_iter())
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn interleave(arrays: Vec<arrays::Array>, indices: Vec<(u32, u32)>) -> Result<arrays::Array, compute::ArrowError> {
        if arrays.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("Cannot interleave empty array list".to_string()));
        }

        let arr_refs: Vec<&dyn arrow_array::Array> = arrays.iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();

        let idx: Vec<(usize, usize)> = indices.iter()
            .map(|(arr_idx, elem_idx)| (*arr_idx as usize, *elem_idx as usize))
            .collect();

        let result = arrow_select::interleave::interleave(&arr_refs, &idx)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn reverse(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_select::take::take(
            &*arr_impl.inner,
            &arrow_array::UInt64Array::from_iter_values((0..arr_impl.inner.len() as u64).rev()),
            None
        ).map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn repeat(arr: arrays::ArrayBorrow<'_>, count: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if count == 0 {
            // Return empty array of the same type
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.slice(0, 0) }));
        }

        if count == 1 {
            return Ok(arrays::Array::new(ArrayImpl { inner: arr_impl.inner.clone() }));
        }

        // Build indices to repeat: [0, 1, 2, ..., n-1, 0, 1, 2, ..., n-1, ...]
        let len = arr_impl.inner.len();
        let indices: Vec<u64> = (0..count)
            .flat_map(|_| (0..len as u64))
            .collect();
        let indices_arr = arrow_array::UInt64Array::from(indices);

        let result = arrow_select::take::take(&*arr_impl.inner, &indices_arr, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    // ========== Partitioning Operations ==========

    fn partition(arrays: Vec<arrays::Array>) -> Result<Vec<(u64, u64)>, compute::ArrowError> {
        if arrays.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("Cannot partition empty array list".to_string()));
        }

        let arr_refs: Vec<Arc<dyn arrow_array::Array>> = arrays.iter()
            .map(|a| a.get::<ArrayImpl>().inner.clone())
            .collect();

        let partitions = arrow_ord::partition::partition(&arr_refs)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        let ranges = partitions.ranges();
        Ok(ranges.iter().map(|r| (r.start as u64, (r.end - r.start) as u64)).collect())
    }

    fn rank(arr: arrays::ArrayBorrow<'_>, options: compute::SortOptions) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        let sort_opts = arrow_ord::sort::SortOptions {
            descending: options.descending,
            nulls_first: options.nulls_first,
        };

        let ranks = arrow_ord::rank::rank(&*arr_impl.inner, Some(sort_opts))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

        // Convert Vec<u32> to UInt32Array
        let result = arrow_array::UInt32Array::from(ranks);
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }
}

// ============================================================================
// IO implementation
// ============================================================================

use std::io::Cursor;
use arrow_ipc::reader::{FileReader as IpcFileReader, StreamReader as IpcStreamReader};
use arrow_ipc::writer::{FileWriter as IpcFileWriter, StreamWriter as IpcStreamWriter};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter as ParquetArrowWriter;
use parquet::basic::Compression as ParquetCompression;
use parquet::file::properties::WriterProperties;

fn to_io_error(e: impl std::fmt::Display) -> io::ArrowError {
    io::ArrowError::IoError(e.to_string())
}

fn to_parquet_compression(comp: io::Compression) -> Result<ParquetCompression, io::ArrowError> {
    match comp {
        io::Compression::Uncompressed => Ok(ParquetCompression::UNCOMPRESSED),
        io::Compression::Snappy => Ok(ParquetCompression::SNAPPY),
        io::Compression::Lz4 => Ok(ParquetCompression::LZ4),
        io::Compression::Gzip => Ok(ParquetCompression::GZIP(Default::default())),
        // ZSTD requires C compilation which doesn't work for WASM
        // Use compression-multiplexer component for ZSTD support
        io::Compression::Zstd => Err(io::ArrowError::NotImplemented(
            "ZSTD compression requires composition with compression-multiplexer component (C bindings not supported in WASM)".to_string()
        )),
        // BZIP2 and LZMA are not supported by the Parquet format
        io::Compression::Bzip2 => Err(io::ArrowError::NotImplemented(
            "BZIP2 compression is not supported by the Parquet format".to_string()
        )),
        io::Compression::Lzma => Err(io::ArrowError::NotImplemented(
            "LZMA compression is not supported by the Parquet format".to_string()
        )),
    }
}

impl io::Guest for Component {
    type BatchReader = BatchReaderImpl;

    // ========== IPC Operations ==========

    fn ipc_read_schema(data: Vec<u8>) -> Result<types::Schema, io::ArrowError> {
        let cursor = Cursor::new(data);
        let reader = IpcStreamReader::try_new(cursor, None).map_err(to_io_error)?;
        Ok(types::Schema::new(SchemaImpl { inner: reader.schema() }))
    }

    fn ipc_read_stream(data: Vec<u8>) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let cursor = Cursor::new(data);
        let reader = IpcStreamReader::try_new(cursor, None).map_err(to_io_error)?;
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn ipc_read_file(data: Vec<u8>) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let cursor = Cursor::new(data);
        let reader = IpcFileReader::try_new(cursor, None).map_err(to_io_error)?;
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn ipc_write_stream(batches: Vec<record_batch::RecordBatch>, _options: Option<io::IpcWriteOptions>) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("No batches to write".to_string()));
        }

        let first_batch = batches[0].get::<RecordBatchImpl>();
        let schema = first_batch.inner.schema();

        let mut buffer = Vec::new();
        {
            let mut writer = IpcStreamWriter::try_new(&mut buffer, &schema).map_err(to_io_error)?;
            for batch in &batches {
                let batch_impl = batch.get::<RecordBatchImpl>();
                writer.write(&batch_impl.inner).map_err(to_io_error)?;
            }
            writer.finish().map_err(to_io_error)?;
        }
        Ok(buffer)
    }

    fn ipc_write_file(batches: Vec<record_batch::RecordBatch>, _options: Option<io::IpcWriteOptions>) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("No batches to write".to_string()));
        }

        let first_batch = batches[0].get::<RecordBatchImpl>();
        let schema = first_batch.inner.schema();

        let mut buffer = Vec::new();
        {
            let mut writer = IpcFileWriter::try_new(&mut buffer, &schema).map_err(to_io_error)?;
            for batch in &batches {
                let batch_impl = batch.get::<RecordBatchImpl>();
                writer.write(&batch_impl.inner).map_err(to_io_error)?;
            }
            writer.finish().map_err(to_io_error)?;
        }
        Ok(buffer)
    }

    // ========== Parquet Operations ==========

    fn parquet_read_schema(data: Vec<u8>) -> Result<types::Schema, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        Ok(types::Schema::new(SchemaImpl { inner: builder.schema().clone() }))
    }

    fn parquet_metadata(data: Vec<u8>) -> Result<io::ParquetFileMetadata, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let metadata = builder.metadata();

        let kv_metadata: Vec<(String, String)> = metadata
            .file_metadata()
            .key_value_metadata()
            .map(|kv| {
                kv.iter()
                    .filter_map(|kv| kv.value.as_ref().map(|v| (kv.key.clone(), v.clone())))
                    .collect()
            })
            .unwrap_or_default();

        Ok(io::ParquetFileMetadata {
            num_rows: metadata.file_metadata().num_rows() as u64,
            num_row_groups: metadata.num_row_groups() as u32,
            created_by: metadata.file_metadata().created_by().map(|s| s.to_string()),
            key_value_metadata: kv_metadata,
        })
    }

    fn parquet_row_group_count(data: Vec<u8>) -> Result<u32, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        Ok(builder.metadata().num_row_groups() as u32)
    }

    fn parquet_get_row_group_metadata(data: Vec<u8>, row_group: u32) -> Result<io::ParquetRowGroupMetadata, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let metadata = builder.metadata();

        let row_group_idx = row_group as usize;
        if row_group_idx >= metadata.num_row_groups() {
            return Err(io::ArrowError::InvalidArgument(format!(
                "Row group {} does not exist (file has {} row groups)",
                row_group, metadata.num_row_groups()
            )));
        }

        let rg_metadata = metadata.row_group(row_group_idx);
        Ok(io::ParquetRowGroupMetadata {
            num_rows: rg_metadata.num_rows() as u64,
            total_byte_size: rg_metadata.total_byte_size() as u64,
            column_count: rg_metadata.num_columns() as u32,
        })
    }

    fn parquet_get_column_statistics(data: Vec<u8>, row_group: u32, column: u32) -> Result<io::ParquetColumnStatistics, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let metadata = builder.metadata();

        let row_group_idx = row_group as usize;
        if row_group_idx >= metadata.num_row_groups() {
            return Err(io::ArrowError::InvalidArgument(format!(
                "Row group {} does not exist (file has {} row groups)",
                row_group, metadata.num_row_groups()
            )));
        }

        let rg_metadata = metadata.row_group(row_group_idx);
        let column_idx = column as usize;
        if column_idx >= rg_metadata.num_columns() {
            return Err(io::ArrowError::InvalidArgument(format!(
                "Column {} does not exist in row group {} (has {} columns)",
                column, row_group, rg_metadata.num_columns()
            )));
        }

        let col_metadata = rg_metadata.column(column_idx);
        let statistics = col_metadata.statistics();

        Ok(io::ParquetColumnStatistics {
            null_count: statistics.and_then(|s| s.null_count_opt()).map(|c| c as u64),
            distinct_count: statistics.and_then(|s| s.distinct_count_opt()).map(|c| c as u64),
            min_value: statistics.and_then(|s| s.min_bytes_opt().map(|b| b.to_vec())),
            max_value: statistics.and_then(|s| s.max_bytes_opt().map(|b| b.to_vec())),
        })
    }

    fn parquet_read(data: Vec<u8>) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let reader = builder.build().map_err(to_io_error)?;
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn parquet_read_columns(data: Vec<u8>, columns: Vec<String>) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;

        let schema = builder.schema();
        let indices: Vec<usize> = columns
            .iter()
            .filter_map(|name| schema.index_of(name).ok())
            .collect();

        let parquet_schema = builder.parquet_schema().clone();
        let reader = builder
            .with_projection(parquet::arrow::ProjectionMask::leaves(
                &parquet_schema,
                indices,
            ))
            .build()
            .map_err(to_io_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn parquet_read_row_groups(data: Vec<u8>, row_groups: Vec<u32>) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let reader = builder
            .with_row_groups(row_groups.into_iter().map(|i| i as usize).collect())
            .build()
            .map_err(to_io_error)?;
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn parquet_write(batches: Vec<record_batch::RecordBatch>, options: Option<io::ParquetWriteOptions>) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("No batches to write".to_string()));
        }

        let first_batch = batches[0].get::<RecordBatchImpl>();
        let schema = first_batch.inner.schema();

        let mut props_builder = WriterProperties::builder();

        if let Some(opts) = options {
            props_builder = props_builder.set_compression(to_parquet_compression(opts.compression)?);
            if let Some(size) = opts.max_row_group_size {
                props_builder = props_builder.set_max_row_group_size(size as usize);
            }
            if let Some(page_size) = opts.data_page_size {
                props_builder = props_builder.set_data_page_size_limit(page_size as usize);
            }
            if opts.dictionary_enabled {
                props_builder = props_builder.set_dictionary_enabled(true);
            }
            if opts.write_statistics {
                props_builder = props_builder.set_statistics_enabled(
                    parquet::file::properties::EnabledStatistics::Chunk,
                );
            }
        }

        let props = props_builder.build();
        let mut buffer = Vec::new();
        {
            let mut writer = ParquetArrowWriter::try_new(&mut buffer, schema, Some(props))
                .map_err(to_io_error)?;
            for batch in &batches {
                let batch_impl = batch.get::<RecordBatchImpl>();
                writer.write(&batch_impl.inner).map_err(to_io_error)?;
            }
            writer.close().map_err(to_io_error)?;
        }
        Ok(buffer)
    }

    // ========== CSV Operations ==========

    fn csv_infer_schema(data: Vec<u8>, options: io::CsvReadOptions) -> Result<types::Schema, io::ArrowError> {
        let cursor = Cursor::new(data);
        let format = arrow_csv::reader::Format::default()
            .with_header(options.has_header)
            .with_delimiter(options.delimiter);

        let (schema, _) = format
            .infer_schema(cursor, options.schema_infer_max_records.map(|n| n as usize))
            .map_err(to_io_error)?;

        Ok(types::Schema::new(SchemaImpl { inner: Arc::new(schema) }))
    }

    fn csv_read(data: Vec<u8>, options: io::CsvReadOptions) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        // First infer schema
        let cursor = Cursor::new(data.clone());
        let format = arrow_csv::reader::Format::default()
            .with_header(options.has_header)
            .with_delimiter(options.delimiter);

        let (schema, _) = format
            .infer_schema(cursor, options.schema_infer_max_records.map(|n| n as usize))
            .map_err(to_io_error)?;

        let cursor = Cursor::new(data);
        let mut builder = arrow_csv::ReaderBuilder::new(Arc::new(schema))
            .with_header(options.has_header)
            .with_delimiter(options.delimiter);

        // Apply optional settings
        if let Some(quote) = options.quote {
            builder = builder.with_quote(quote);
        }
        if let Some(escape) = options.escape {
            builder = builder.with_escape(escape);
        }

        let reader = builder.build(cursor).map_err(to_io_error)?;

        // Handle skip_rows by skipping initial batches/records
        let batches: Result<Vec<_>, _> = reader.collect();
        let mut batches = batches.map_err(to_io_error)?;

        // If skip_rows is specified, skip the appropriate number of rows
        if let Some(skip) = options.skip_rows {
            let mut rows_skipped = 0u64;
            let mut start_batch = 0;
            for (i, batch) in batches.iter().enumerate() {
                if rows_skipped + (batch.num_rows() as u64) <= skip {
                    rows_skipped += batch.num_rows() as u64;
                    start_batch = i + 1;
                } else {
                    // Partial batch - slice it
                    let offset = (skip - rows_skipped) as usize;
                    if offset > 0 && offset < batch.num_rows() {
                        batches[i] = batch.slice(offset, batch.num_rows() - offset);
                    }
                    start_batch = i;
                    break;
                }
            }
            batches = batches.into_iter().skip(start_batch).collect();
        }

        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn csv_read_with_schema(data: Vec<u8>, schema: types::Schema, options: io::CsvReadOptions) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();
        let cursor = Cursor::new(data);
        let reader = arrow_csv::ReaderBuilder::new(schema_impl.inner.clone())
            .with_header(options.has_header)
            .with_delimiter(options.delimiter)
            .build(cursor)
            .map_err(to_io_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn csv_write(batches: Vec<record_batch::RecordBatch>, options: io::CsvWriteOptions) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("No batches to write".to_string()));
        }

        let mut buffer = Vec::new();
        {
            let mut builder = arrow_csv::WriterBuilder::new()
                .with_header(options.has_header)
                .with_delimiter(options.delimiter);

            if let Some(fmt) = &options.date_format {
                builder = builder.with_date_format(fmt.clone());
            }
            if let Some(fmt) = &options.timestamp_format {
                builder = builder.with_timestamp_format(fmt.clone());
            }

            let mut writer = builder.build(&mut buffer);

            for batch in &batches {
                let batch_impl = batch.get::<RecordBatchImpl>();
                writer.write(&batch_impl.inner).map_err(to_io_error)?;
            }
        }
        Ok(buffer)
    }

    // ========== JSON Operations ==========

    fn json_infer_schema(data: Vec<u8>, options: io::JsonReadOptions) -> Result<types::Schema, io::ArrowError> {
        let cursor = Cursor::new(data);
        let buf_reader = std::io::BufReader::new(cursor);
        let (schema, _) = arrow_json::reader::infer_json_schema(
            buf_reader,
            options.schema_infer_max_records.map(|n| n as usize),
        )
        .map_err(to_io_error)?;

        Ok(types::Schema::new(SchemaImpl { inner: Arc::new(schema) }))
    }

    fn json_read(data: Vec<u8>, options: io::JsonReadOptions) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        // First infer schema
        let cursor = Cursor::new(data.clone());
        let buf_reader = std::io::BufReader::new(cursor);
        let (schema, _) = arrow_json::reader::infer_json_schema(
            buf_reader,
            options.schema_infer_max_records.map(|n| n as usize),
        )
        .map_err(to_io_error)?;

        let cursor = Cursor::new(data);
        let reader = arrow_json::ReaderBuilder::new(Arc::new(schema))
            .build(cursor)
            .map_err(to_io_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn json_read_with_schema(data: Vec<u8>, schema: types::Schema, _options: io::JsonReadOptions) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();
        let cursor = Cursor::new(data);
        let reader = arrow_json::ReaderBuilder::new(schema_impl.inner.clone())
            .build(cursor)
            .map_err(to_io_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn json_write(batches: Vec<record_batch::RecordBatch>) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("No batches to write".to_string()));
        }

        let mut buffer = Vec::new();
        {
            let mut writer = arrow_json::LineDelimitedWriter::new(&mut buffer);
            for batch in &batches {
                let batch_impl = batch.get::<RecordBatchImpl>();
                writer.write(&batch_impl.inner).map_err(to_io_error)?;
            }
            writer.finish().map_err(to_io_error)?;
        }
        Ok(buffer)
    }

    // ========== Avro Format ==========

    fn avro_read(data: Vec<u8>) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let cursor = std::io::BufReader::new(Cursor::new(data));
        let reader = arrow_avro::reader::ReaderBuilder::new()
            .build(cursor)
            .map_err(|e| io::ArrowError::IoError(e.to_string()))?;
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(|e| io::ArrowError::IoError(e.to_string()))?;
        Ok(batches
            .into_iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b }))
            .collect())
    }

    fn avro_infer_schema(data: Vec<u8>) -> Result<types::Schema, io::ArrowError> {
        let cursor = std::io::BufReader::new(Cursor::new(data));
        let reader = arrow_avro::reader::ReaderBuilder::new()
            .build(cursor)
            .map_err(|e| io::ArrowError::IoError(e.to_string()))?;
        Ok(types::Schema::new(SchemaImpl { inner: reader.schema() }))
    }

    fn avro_write(batches: Vec<record_batch::RecordBatch>) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("Cannot write empty batch list to Avro".to_string()));
        }

        let first_batch = batches[0].get::<RecordBatchImpl>();
        let schema = first_batch.inner.schema();

        let mut output = Vec::new();
        let mut writer = arrow_avro::writer::WriterBuilder::new((*schema).clone())
            .build::<_, arrow_avro::writer::format::AvroOcfFormat>(&mut output)
            .map_err(|e| io::ArrowError::IoError(e.to_string()))?;

        for batch in &batches {
            let batch_impl = batch.get::<RecordBatchImpl>();
            writer.write(&batch_impl.inner)
                .map_err(|e| io::ArrowError::IoError(e.to_string()))?;
        }

        writer.finish()
            .map_err(|e| io::ArrowError::IoError(e.to_string()))?;

        Ok(output)
    }

    // ========== Streaming Readers ==========

    fn ipc_stream_reader(data: Vec<u8>) -> Result<io::BatchReader, io::ArrowError> {
        let cursor = Cursor::new(data);
        let reader = IpcStreamReader::try_new(cursor, None).map_err(to_io_error)?;
        let schema = reader.schema();
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(io::BatchReader::new(BatchReaderImpl {
            batches,
            index: std::cell::Cell::new(0),
            schema,
        }))
    }

    fn parquet_stream_reader(data: Vec<u8>) -> Result<io::BatchReader, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let schema = builder.schema().clone();
        let reader = builder.build().map_err(to_io_error)?;
        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(io::BatchReader::new(BatchReaderImpl {
            batches,
            index: std::cell::Cell::new(0),
            schema,
        }))
    }

    fn csv_stream_reader(data: Vec<u8>, options: io::CsvReadOptions) -> Result<io::BatchReader, io::ArrowError> {
        // First infer schema
        let cursor = Cursor::new(data.clone());
        let format = arrow_csv::reader::Format::default()
            .with_header(options.has_header)
            .with_delimiter(options.delimiter);

        let (schema, _) = format
            .infer_schema(cursor, options.schema_infer_max_records.map(|n| n as usize))
            .map_err(to_io_error)?;

        let schema = Arc::new(schema);
        let cursor = Cursor::new(data);
        let reader = arrow_csv::ReaderBuilder::new(schema.clone())
            .with_header(options.has_header)
            .with_delimiter(options.delimiter)
            .build(cursor)
            .map_err(to_io_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(io::BatchReader::new(BatchReaderImpl {
            batches,
            index: std::cell::Cell::new(0),
            schema,
        }))
    }

    fn json_stream_reader(data: Vec<u8>, options: io::JsonReadOptions) -> Result<io::BatchReader, io::ArrowError> {
        // First infer schema
        let cursor = Cursor::new(data.clone());
        let buf_reader = std::io::BufReader::new(cursor);
        let (schema, _) = arrow_json::reader::infer_json_schema(
            buf_reader,
            options.schema_infer_max_records.map(|n| n as usize),
        )
        .map_err(to_io_error)?;

        let schema = Arc::new(schema);
        let cursor = Cursor::new(data);
        let reader = arrow_json::ReaderBuilder::new(schema.clone())
            .build(cursor)
            .map_err(to_io_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_io_error)?;
        Ok(io::BatchReader::new(BatchReaderImpl {
            batches,
            index: std::cell::Cell::new(0),
            schema,
        }))
    }
}

struct BatchReaderImpl {
    batches: Vec<ArrowRecordBatch>,
    index: std::cell::Cell<usize>,
    schema: Arc<arrow_schema::Schema>,
}

impl io::GuestBatchReader for BatchReaderImpl {
    fn schema(&self) -> types::Schema {
        types::Schema::new(SchemaImpl { inner: self.schema.clone() })
    }

    fn next(&self) -> Option<Result<record_batch::RecordBatch, io::ArrowError>> {
        let idx = self.index.get();
        if idx >= self.batches.len() {
            return None;
        }
        self.index.set(idx + 1);
        let batch = self.batches[idx].clone();
        Some(Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch })))
    }

    fn collect(&self) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let idx = self.index.get();
        let remaining: Vec<record_batch::RecordBatch> = self.batches[idx..]
            .iter()
            .map(|b| record_batch::RecordBatch::new(RecordBatchImpl { inner: b.clone() }))
            .collect();
        self.index.set(self.batches.len());
        Ok(remaining)
    }
}

// ============================================================================
// Flight implementation
// ============================================================================

use crate::bindings::exports::arrow::arrow_wasm::flight;

fn to_flight_error(e: impl std::fmt::Display) -> flight::ArrowError {
    flight::ArrowError::IoError(e.to_string())
}

impl flight::Guest for Component {
    fn encode_batch(batch: record_batch::RecordBatch, schema: types::Schema) -> Result<flight::FlightData, flight::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let schema_impl = schema.get::<SchemaImpl>();

        // Write the batch to IPC stream format (includes schema in the stream)
        let mut data_body = Vec::new();
        {
            let mut writer = IpcStreamWriter::try_new(&mut data_body, &schema_impl.inner)
                .map_err(to_flight_error)?;
            writer.write(&batch_impl.inner).map_err(to_flight_error)?;
            writer.finish().map_err(to_flight_error)?;
        }

        // For Flight, the data_header can contain schema info encoded as IPC
        // We'll encode schema as a separate IPC stream containing just schema
        let mut data_header = Vec::new();
        {
            let mut writer = IpcStreamWriter::try_new(&mut data_header, &schema_impl.inner)
                .map_err(to_flight_error)?;
            // Just write schema by finishing without writing batches
            writer.finish().map_err(to_flight_error)?;
        }

        Ok(flight::FlightData {
            descriptor: None,
            data_header,
            app_metadata: Vec::new(),
            data_body,
        })
    }

    fn encode_batches(batches: Vec<record_batch::RecordBatch>, schema: types::Schema) -> Result<Vec<flight::FlightData>, flight::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();
        let mut result = Vec::new();

        // First message includes schema in header
        let mut schema_header = Vec::new();
        {
            let mut writer = IpcStreamWriter::try_new(&mut schema_header, &schema_impl.inner)
                .map_err(to_flight_error)?;
            writer.finish().map_err(to_flight_error)?;
        }

        for (i, batch) in batches.iter().enumerate() {
            let batch_impl = batch.get::<RecordBatchImpl>();

            let mut data_body = Vec::new();
            {
                let mut writer = IpcStreamWriter::try_new(&mut data_body, &schema_impl.inner)
                    .map_err(to_flight_error)?;
                writer.write(&batch_impl.inner).map_err(to_flight_error)?;
                writer.finish().map_err(to_flight_error)?;
            }

            let data_header = if i == 0 {
                schema_header.clone()
            } else {
                Vec::new()
            };

            result.push(flight::FlightData {
                descriptor: None,
                data_header,
                app_metadata: Vec::new(),
                data_body,
            });
        }

        Ok(result)
    }

    fn decode_batch(data: flight::FlightData, _schema: types::Schema) -> Result<record_batch::RecordBatch, flight::ArrowError> {
        // Read the batch from IPC stream format (schema is embedded in the stream)
        let cursor = Cursor::new(data.data_body);
        let reader = IpcStreamReader::try_new(cursor, None)
            .map_err(to_flight_error)?;

        let batches: Result<Vec<_>, _> = reader.collect();
        let batches = batches.map_err(to_flight_error)?;

        if batches.is_empty() {
            return Err(flight::ArrowError::InvalidArgument("No batches in FlightData".to_string()));
        }

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: batches.into_iter().next().unwrap() }))
    }

    fn decode_batches(data: Vec<flight::FlightData>) -> Result<Vec<record_batch::RecordBatch>, flight::ArrowError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();

        for flight_data in data {
            if flight_data.data_body.is_empty() {
                continue;
            }

            let cursor = Cursor::new(flight_data.data_body);
            let reader = IpcStreamReader::try_new(cursor, None)
                .map_err(to_flight_error)?;

            let batches: Result<Vec<_>, _> = reader.collect();
            let batches = batches.map_err(to_flight_error)?;

            for batch in batches {
                result.push(record_batch::RecordBatch::new(RecordBatchImpl { inner: batch }));
            }
        }

        Ok(result)
    }

    fn extract_schema(info: flight::FlightInfo) -> Result<types::Schema, flight::ArrowError> {
        if info.schema_bytes.is_empty() {
            return Err(flight::ArrowError::InvalidArgument("FlightInfo has no schema bytes".to_string()));
        }

        // Schema bytes are stored as an IPC stream - read it to extract schema
        let cursor = Cursor::new(info.schema_bytes);
        let reader = IpcStreamReader::try_new(cursor, None)
            .map_err(to_flight_error)?;

        let schema = reader.schema();
        Ok(types::Schema::new(SchemaImpl { inner: schema }))
    }

    fn create_flight_info(
        schema: types::Schema,
        descriptor: Option<flight::FlightDescriptor>,
        endpoints: Vec<flight::FlightEndpoint>,
        total_records: i64,
        total_bytes: i64,
    ) -> Result<flight::FlightInfo, flight::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();

        // Encode schema as IPC stream
        let mut schema_bytes = Vec::new();
        {
            let mut writer = IpcStreamWriter::try_new(&mut schema_bytes, &schema_impl.inner)
                .map_err(to_flight_error)?;
            writer.finish().map_err(to_flight_error)?;
        }

        Ok(flight::FlightInfo {
            schema_bytes,
            schema: Some(schema),
            descriptor,
            endpoints,
            total_records,
            total_bytes,
            ordered: false,
            app_metadata: Vec::new(),
        })
    }

    fn create_path_descriptor(path: Vec<String>) -> flight::FlightDescriptor {
        flight::FlightDescriptor {
            type_: flight::DescriptorType::Path,
            cmd: None,
            path,
        }
    }

    fn create_cmd_descriptor(cmd: Vec<u8>) -> flight::FlightDescriptor {
        flight::FlightDescriptor {
            type_: flight::DescriptorType::Cmd,
            cmd: Some(cmd),
            path: Vec::new(),
        }
    }

    fn serialize_flight_info(info: flight::FlightInfo) -> Result<Vec<u8>, flight::ArrowError> {
        // Simple serialization: length-prefixed fields
        let mut buffer = Vec::new();

        // Schema bytes
        let schema_len = info.schema_bytes.len() as u32;
        buffer.extend_from_slice(&schema_len.to_le_bytes());
        buffer.extend_from_slice(&info.schema_bytes);

        // Total records and bytes
        buffer.extend_from_slice(&info.total_records.to_le_bytes());
        buffer.extend_from_slice(&info.total_bytes.to_le_bytes());

        // Ordered flag
        buffer.push(if info.ordered { 1 } else { 0 });

        // App metadata
        let meta_len = info.app_metadata.len() as u32;
        buffer.extend_from_slice(&meta_len.to_le_bytes());
        buffer.extend_from_slice(&info.app_metadata);

        // Number of endpoints
        let endpoints_len = info.endpoints.len() as u32;
        buffer.extend_from_slice(&endpoints_len.to_le_bytes());

        for endpoint in &info.endpoints {
            // Ticket
            let ticket_len = endpoint.ticket.len() as u32;
            buffer.extend_from_slice(&ticket_len.to_le_bytes());
            buffer.extend_from_slice(&endpoint.ticket);

            // Locations
            let locs_len = endpoint.locations.len() as u32;
            buffer.extend_from_slice(&locs_len.to_le_bytes());
            for loc in &endpoint.locations {
                let loc_bytes = loc.as_bytes();
                let loc_len = loc_bytes.len() as u32;
                buffer.extend_from_slice(&loc_len.to_le_bytes());
                buffer.extend_from_slice(loc_bytes);
            }

            // Expiration time
            buffer.extend_from_slice(&endpoint.expiration_time.to_le_bytes());

            // App metadata
            let ep_meta_len = endpoint.app_metadata.len() as u32;
            buffer.extend_from_slice(&ep_meta_len.to_le_bytes());
            buffer.extend_from_slice(&endpoint.app_metadata);
        }

        Ok(buffer)
    }

    fn deserialize_flight_info(data: Vec<u8>) -> Result<flight::FlightInfo, flight::ArrowError> {
        let mut cursor = std::io::Cursor::new(&data);
        use std::io::Read;

        fn read_u32(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Result<u32, flight::ArrowError> {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf).map_err(to_flight_error)?;
            Ok(u32::from_le_bytes(buf))
        }

        fn read_u64(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Result<u64, flight::ArrowError> {
            let mut buf = [0u8; 8];
            cursor.read_exact(&mut buf).map_err(to_flight_error)?;
            Ok(u64::from_le_bytes(buf))
        }

        fn read_i64(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Result<i64, flight::ArrowError> {
            let mut buf = [0u8; 8];
            cursor.read_exact(&mut buf).map_err(to_flight_error)?;
            Ok(i64::from_le_bytes(buf))
        }

        fn read_bytes(cursor: &mut std::io::Cursor<&Vec<u8>>, len: usize) -> Result<Vec<u8>, flight::ArrowError> {
            let mut buf = vec![0u8; len];
            cursor.read_exact(&mut buf).map_err(to_flight_error)?;
            Ok(buf)
        }

        // Schema bytes
        let schema_len = read_u32(&mut cursor)? as usize;
        let schema_bytes = read_bytes(&mut cursor, schema_len)?;

        // Total records and bytes
        let total_records = read_i64(&mut cursor)?;
        let total_bytes = read_i64(&mut cursor)?;

        // Ordered flag
        let mut ordered_buf = [0u8; 1];
        cursor.read_exact(&mut ordered_buf).map_err(to_flight_error)?;
        let ordered = ordered_buf[0] != 0;

        // App metadata
        let meta_len = read_u32(&mut cursor)? as usize;
        let app_metadata = read_bytes(&mut cursor, meta_len)?;

        // Endpoints
        let endpoints_len = read_u32(&mut cursor)? as usize;
        let mut endpoints = Vec::with_capacity(endpoints_len);

        for _ in 0..endpoints_len {
            // Ticket
            let ticket_len = read_u32(&mut cursor)? as usize;
            let ticket = read_bytes(&mut cursor, ticket_len)?;

            // Locations
            let locs_len = read_u32(&mut cursor)? as usize;
            let mut locations = Vec::with_capacity(locs_len);
            for _ in 0..locs_len {
                let loc_len = read_u32(&mut cursor)? as usize;
                let loc_bytes = read_bytes(&mut cursor, loc_len)?;
                let loc = String::from_utf8(loc_bytes)
                    .map_err(|e| flight::ArrowError::InvalidArgument(e.to_string()))?;
                locations.push(loc);
            }

            // Expiration time
            let expiration_time = read_u64(&mut cursor)?;

            // App metadata
            let ep_meta_len = read_u32(&mut cursor)? as usize;
            let ep_app_metadata = read_bytes(&mut cursor, ep_meta_len)?;

            endpoints.push(flight::FlightEndpoint {
                ticket,
                locations,
                expiration_time,
                app_metadata: ep_app_metadata,
            });
        }

        Ok(flight::FlightInfo {
            schema_bytes,
            schema: None, // Caller can use extract_schema if needed
            descriptor: None,
            endpoints,
            total_records,
            total_bytes,
            ordered,
            app_metadata,
        })
    }

    fn serialize_flight_data(data: flight::FlightData) -> Result<Vec<u8>, flight::ArrowError> {
        let mut buffer = Vec::new();

        // Data header
        let header_len = data.data_header.len() as u32;
        buffer.extend_from_slice(&header_len.to_le_bytes());
        buffer.extend_from_slice(&data.data_header);

        // App metadata
        let meta_len = data.app_metadata.len() as u32;
        buffer.extend_from_slice(&meta_len.to_le_bytes());
        buffer.extend_from_slice(&data.app_metadata);

        // Data body
        let body_len = data.data_body.len() as u32;
        buffer.extend_from_slice(&body_len.to_le_bytes());
        buffer.extend_from_slice(&data.data_body);

        Ok(buffer)
    }

    fn deserialize_flight_data(data: Vec<u8>) -> Result<flight::FlightData, flight::ArrowError> {
        let mut cursor = std::io::Cursor::new(&data);
        use std::io::Read;

        fn read_u32(cursor: &mut std::io::Cursor<&Vec<u8>>) -> Result<u32, flight::ArrowError> {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf).map_err(to_flight_error)?;
            Ok(u32::from_le_bytes(buf))
        }

        fn read_bytes(cursor: &mut std::io::Cursor<&Vec<u8>>, len: usize) -> Result<Vec<u8>, flight::ArrowError> {
            let mut buf = vec![0u8; len];
            cursor.read_exact(&mut buf).map_err(to_flight_error)?;
            Ok(buf)
        }

        // Data header
        let header_len = read_u32(&mut cursor)? as usize;
        let data_header = read_bytes(&mut cursor, header_len)?;

        // App metadata
        let meta_len = read_u32(&mut cursor)? as usize;
        let app_metadata = read_bytes(&mut cursor, meta_len)?;

        // Data body
        let body_len = read_u32(&mut cursor)? as usize;
        let data_body = read_bytes(&mut cursor, body_len)?;

        Ok(flight::FlightData {
            descriptor: None,
            data_header,
            app_metadata,
            data_body,
        })
    }
}
