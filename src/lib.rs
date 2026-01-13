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

// Main component struct
struct Component;

bindings::export!(Component with_types_in bindings);

// ============================================================================
// Types implementation
// ============================================================================

impl types::Guest for Component {
    type Field = FieldImpl;
    type Schema = SchemaImpl;
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

    fn concat(arr: Vec<arrays::Array>) -> Result<arrays::Array, arrays::ArrowError> {
        let refs: Vec<&dyn arrow_array::Array> = arr
            .iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();
        let result = arrow_select::concat::concat(&refs)
            .map_err(|e| arrays::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
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

    fn window_lead(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, offset: u32, _default_value: Option<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        // Build result by taking values at offset positions ahead
        // For Int64 arrays (simplest case)
        if let Some(i64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![None; len];

            for (start, end) in partitions {
                for i in start..end {
                    let target = i + offset as usize;
                    if target < end {
                        let original_idx = sort_indices[i];
                        let lead_idx = sort_indices[target];
                        result[original_idx] = get_i64_opt(i64_arr, lead_idx);
                    }
                }
            }

            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        // For Float64 arrays
        if let Some(f64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut result: Vec<Option<f64>> = vec![None; len];

            for (start, end) in partitions {
                for i in start..end {
                    let target = i + offset as usize;
                    if target < end {
                        let original_idx = sort_indices[i];
                        let lead_idx = sort_indices[target];
                        result[original_idx] = get_f64_opt(f64_arr, lead_idx);
                    }
                }
            }

            let result_arr: arrow_array::Float64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_lead supports Int64 and Float64 arrays".to_string()))
    }

    fn window_lag(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, offset: u32, _default_value: Option<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        if let Some(i64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![None; len];

            for (start, end) in partitions {
                for i in start..end {
                    if i >= start + offset as usize {
                        let target = i - offset as usize;
                        let original_idx = sort_indices[i];
                        let lag_idx = sort_indices[target];
                        result[original_idx] = get_i64_opt(i64_arr, lag_idx);
                    }
                }
            }

            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        if let Some(f64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut result: Vec<Option<f64>> = vec![None; len];

            for (start, end) in partitions {
                for i in start..end {
                    if i >= start + offset as usize {
                        let target = i - offset as usize;
                        let original_idx = sort_indices[i];
                        let lag_idx = sort_indices[target];
                        result[original_idx] = get_f64_opt(f64_arr, lag_idx);
                    }
                }
            }

            let result_arr: arrow_array::Float64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_lag supports Int64 and Float64 arrays".to_string()))
    }

    fn window_first_value(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        if let Some(i64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![None; len];

            for (start, end) in partitions {
                if start < end {
                    let first_idx = sort_indices[start];
                    let first_val = get_i64_opt(i64_arr, first_idx);
                    for i in start..end {
                        result[sort_indices[i]] = first_val;
                    }
                }
            }

            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        if let Some(f64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut result: Vec<Option<f64>> = vec![None; len];

            for (start, end) in partitions {
                if start < end {
                    let first_idx = sort_indices[start];
                    let first_val = get_f64_opt(f64_arr, first_idx);
                    for i in start..end {
                        result[sort_indices[i]] = first_val;
                    }
                }
            }

            let result_arr: arrow_array::Float64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_first_value supports Int64 and Float64 arrays".to_string()))
    }

    fn window_last_value(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        if let Some(i64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![None; len];

            for (start, end) in partitions {
                if start < end {
                    let last_idx = sort_indices[end - 1];
                    let last_val = get_i64_opt(i64_arr, last_idx);
                    for i in start..end {
                        result[sort_indices[i]] = last_val;
                    }
                }
            }

            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        if let Some(f64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut result: Vec<Option<f64>> = vec![None; len];

            for (start, end) in partitions {
                if start < end {
                    let last_idx = sort_indices[end - 1];
                    let last_val = get_f64_opt(f64_arr, last_idx);
                    for i in start..end {
                        result[sort_indices[i]] = last_val;
                    }
                }
            }

            let result_arr: arrow_array::Float64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_last_value supports Int64 and Float64 arrays".to_string()))
    }

    fn window_nth_value(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, n: u32, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        if n == 0 {
            return Err(compute::ArrowError::InvalidArgument("nth_value n must be >= 1".to_string()));
        }

        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        if let Some(i64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let mut result: Vec<Option<i64>> = vec![None; len];

            for (start, end) in partitions {
                let nth_offset = (n - 1) as usize;
                if start + nth_offset < end {
                    let nth_idx = sort_indices[start + nth_offset];
                    let nth_val = get_i64_opt(i64_arr, nth_idx);
                    for i in start..end {
                        result[sort_indices[i]] = nth_val;
                    }
                }
            }

            let result_arr: arrow_array::Int64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        if let Some(f64_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            let mut result: Vec<Option<f64>> = vec![None; len];

            for (start, end) in partitions {
                let nth_offset = (n - 1) as usize;
                if start + nth_offset < end {
                    let nth_idx = sort_indices[start + nth_offset];
                    let nth_val = get_f64_opt(f64_arr, nth_idx);
                    for i in start..end {
                        result[sort_indices[i]] = nth_val;
                    }
                }
            }

            let result_arr: arrow_array::Float64Array = result.into_iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }));
        }

        Err(compute::ArrowError::InvalidArgument("window_nth_value supports Int64 and Float64 arrays".to_string()))
    }

    fn window_sum(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![0.0f64; len];

        for (start, end) in partitions {
            let mut running_sum = 0.0f64;
            for i in start..end {
                let original_idx = sort_indices[i];
                running_sum += values[original_idx];
                result[original_idx] = running_sum;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_avg(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![0.0f64; len];

        for (start, end) in partitions {
            let mut running_sum = 0.0f64;
            for (count, i) in (start..end).enumerate() {
                let original_idx = sort_indices[i];
                running_sum += values[original_idx];
                result[original_idx] = running_sum / (count + 1) as f64;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_min(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![f64::NAN; len];

        for (start, end) in partitions {
            let mut running_min = f64::INFINITY;
            for i in start..end {
                let original_idx = sort_indices[i];
                running_min = running_min.min(values[original_idx]);
                result[original_idx] = running_min;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_max(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let values = collect_f64_values(&arr_impl.inner)?;
        let len = values.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![f64::NAN; len];

        for (start, end) in partitions {
            let mut running_max = f64::NEG_INFINITY;
            for i in start..end {
                let original_idx = sort_indices[i];
                running_max = running_max.max(values[original_idx]);
                result[original_idx] = running_max;
            }
        }

        let result_arr: arrow_array::Float64Array = result.into_iter().map(Some).collect();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    fn window_count(arr: arrays::ArrayBorrow<'_>, partition_by: Vec<arrays::Array>, order_by: Vec<arrays::Array>, order_options: Vec<compute::SortOptions>, _frame: Option<compute::WindowFrame>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();

        let (partitions, sort_indices) = compute_window_partitions_and_order(&partition_by, &order_by, &order_options)?;

        let mut result = vec![0u64; len];

        for (start, end) in partitions {
            let mut running_count = 0u64;
            for i in start..end {
                let original_idx = sort_indices[i];
                if !arr_impl.inner.is_null(original_idx) {
                    running_count += 1;
                }
                result[original_idx] = running_count;
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

        Err(compute::ArrowError::NotImplemented("if_else not implemented for this array type".to_string()))
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
        // These codecs require the compression-multiplexer component to be composed
        io::Compression::Zstd => Err(io::ArrowError::NotImplemented(
            "ZSTD compression requires composition with compression-multiplexer component".to_string()
        )),
        io::Compression::Gzip => Err(io::ArrowError::NotImplemented(
            "GZIP compression requires composition with compression-multiplexer component".to_string()
        )),
        io::Compression::Bzip2 => Err(io::ArrowError::NotImplemented(
            "BZIP2 compression requires composition with compression-multiplexer component".to_string()
        )),
        io::Compression::Lzma => Err(io::ArrowError::NotImplemented(
            "LZMA compression requires composition with compression-multiplexer component".to_string()
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
        let reader = arrow_csv::ReaderBuilder::new(Arc::new(schema))
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
