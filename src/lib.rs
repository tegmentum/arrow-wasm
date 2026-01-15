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
use std::io::Cursor;
use std::sync::Arc;

// Re-export for internal use
use arrow_array::{Array as ArrowArrayTrait, ArrayRef, RecordBatch as ArrowRecordBatch};

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

// Helper function to collect f64 values with nulls preserved
fn collect_nullable_f64_values(arr: &dyn arrow_array::Array) -> Result<Vec<Option<f64>>, compute::ArrowError> {
    if let Some(f64_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        return Ok(f64_arr.iter().collect());
    }
    if let Some(f32_arr) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        return Ok(f32_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Ok(i64_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        return Ok(i32_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    if let Some(i16_arr) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Ok(i16_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    if let Some(i8_arr) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Ok(i8_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    if let Some(u64_arr) = arr.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        return Ok(u64_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    if let Some(u32_arr) = arr.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        return Ok(u32_arr.iter().map(|v| v.map(|x| x as f64)).collect());
    }
    Err(compute::ArrowError::InvalidArgument("Expected numeric array".to_string()))
}

// Helper function to check if two rows have the same rank (same order_by values)
fn are_same_rank(sort_indices: &[usize], order_by: &[arrays::Array], i: usize, j: usize) -> bool {
    for arr in order_by {
        let arr_impl = arr.get::<ArrayImpl>();
        let idx_i = sort_indices[i];
        let idx_j = sort_indices[j];

        // Compare values at both indices
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            if str_arr.is_null(idx_i) != str_arr.is_null(idx_j) {
                return false;
            }
            if !str_arr.is_null(idx_i) && str_arr.value(idx_i) != str_arr.value(idx_j) {
                return false;
            }
        } else if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            if int_arr.is_null(idx_i) != int_arr.is_null(idx_j) {
                return false;
            }
            if !int_arr.is_null(idx_i) && int_arr.value(idx_i) != int_arr.value(idx_j) {
                return false;
            }
        } else if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            if float_arr.is_null(idx_i) != float_arr.is_null(idx_j) {
                return false;
            }
            if !float_arr.is_null(idx_i) && (float_arr.value(idx_i) - float_arr.value(idx_j)).abs() > f64::EPSILON {
                return false;
            }
        } else if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int32Array>() {
            if int_arr.is_null(idx_i) != int_arr.is_null(idx_j) {
                return false;
            }
            if !int_arr.is_null(idx_i) && int_arr.value(idx_i) != int_arr.value(idx_j) {
                return false;
            }
        }
    }
    true
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

// ========== Phase 15 Helper Functions ==========

/// Count distinct values in an array
fn count_distinct_values(arr: &Arc<dyn arrow_array::Array>) -> Result<u64, compute::ArrowError> {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();

    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        for i in 0..str_arr.len() {
            if !str_arr.is_null(i) {
                seen.insert(str_arr.value(i).to_string());
            }
        }
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        for i in 0..int_arr.len() {
            if !int_arr.is_null(i) {
                seen.insert(int_arr.value(i).to_string());
            }
        }
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        for i in 0..int_arr.len() {
            if !int_arr.is_null(i) {
                seen.insert(int_arr.value(i).to_string());
            }
        }
    } else if let Some(f64_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        for i in 0..f64_arr.len() {
            if !f64_arr.is_null(i) {
                seen.insert(format!("{:.10}", f64_arr.value(i)));
            }
        }
    } else if let Some(bool_arr) = arr.as_any().downcast_ref::<arrow_array::BooleanArray>() {
        for i in 0..bool_arr.len() {
            if !bool_arr.is_null(i) {
                seen.insert(bool_arr.value(i).to_string());
            }
        }
    }

    Ok(seen.len() as u64)
}

/// Profile a column (implementation)

/// Approximate t-distribution p-value using normal approximation for large df
fn t_distribution_pvalue(t: f64, df: f64) -> f64 {
    // For large df, t-distribution approaches normal
    // Use approximation: t * sqrt(df / (df - 2)) ~ N(0,1) for df > 30
    if df > 30.0 {
        normal_cdf(-t.abs())
    } else {
        // Simple approximation for smaller df using beta function properties
        // This is a rough approximation
        let x = df / (df + t * t);
        0.5 * incomplete_beta(df / 2.0, 0.5, x)
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Horner form approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Incomplete beta function approximation (for chi-square p-value)
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    // Simple approximation using continued fraction
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Use series expansion for small x
    let mut sum = 0.0;
    let mut term = 1.0;
    for n in 0..100 {
        if n > 0 {
            term *= (a + n as f64 - 1.0) * x / (a + b + n as f64 - 1.0) / n as f64;
        }
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
    }

    x.powf(a) * (1.0 - x).powf(b) * sum / a
}

/// Chi-square p-value using gamma function approximation
fn chi_square_pvalue(chi2: f64, df: f64) -> f64 {
    // Upper incomplete gamma function ratio
    // P(chi2, df) = gamma_inc(df/2, chi2/2) / gamma(df/2)
    1.0 - gamma_inc_ratio(df / 2.0, chi2 / 2.0)
}

/// Regularized incomplete gamma function approximation
fn gamma_inc_ratio(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    // Series expansion for small x
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 {
                break;
            }
        }
        return sum * (-x + a * x.ln() - ln_gamma(a)).exp();
    }

    // Continued fraction for large x
    1.0 - gamma_inc_upper_ratio(a, x)
}

/// Upper incomplete gamma function ratio (complement)
fn gamma_inc_upper_ratio(a: f64, x: f64) -> f64 {
    // Lentz's algorithm for continued fraction
    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 0.0;

    for n in 1..100 {
        let an = if n == 1 { 1.0 } else { (n as f64 - 1.0 - a) * (n as f64 - 1.0) };
        let bn = x + 2.0 * n as f64 - 1.0 - a;

        d = bn + an * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        d = 1.0 / d;

        c = bn + an / c;
        if c.abs() < 1e-30 { c = 1e-30; }

        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    f * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Log gamma function approximation (Stirling)
fn ln_gamma(x: f64) -> f64 {
    // Stirling's approximation
    let x = x - 1.0;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * x.ln() - x
        + 1.0 / (12.0 * x) - 1.0 / (360.0 * x.powi(3))
}

/// Compute outlier bounds based on method
/// Pearson correlation coefficient
fn compute_pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, compute::ArrowError> {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return Ok(0.0);
    }

    Ok(cov / (var_x * var_y).sqrt())
}

/// Spearman rank correlation
fn compute_spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64, compute::ArrowError> {
    // Convert to ranks
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Compute Pearson correlation on ranks
    compute_pearson_correlation(&rank_x, &rank_y)
}

/// Compute ranks for a vector
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(f64, usize)> = values.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i;
        while j < indexed.len() && (indexed[j].0 - indexed[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].1] = avg_rank;
        }
        i = j;
    }

    ranks
}

/// Extract string value at index from any array type
fn extract_string_at_index(arr: &Arc<dyn arrow_array::Array>, idx: usize) -> Option<String> {
    if arr.is_null(idx) {
        return None;
    }

    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Some(str_arr.value(idx).to_string());
    }
    if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        return Some(int_arr.value(idx).to_string());
    }
    if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        return Some(int_arr.value(idx).to_string());
    }
    if let Some(f64_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        return Some(f64_arr.value(idx).to_string());
    }
    if let Some(bool_arr) = arr.as_any().downcast_ref::<arrow_array::BooleanArray>() {
        return Some(bool_arr.value(idx).to_string());
    }

    None
}

/// Extract all string values from a string array
fn extract_string_values(arr: &Arc<dyn arrow_array::Array>) -> Result<Vec<Option<String>>, compute::ArrowError> {
    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        return Ok(str_arr.iter().map(|opt| opt.map(|s| s.to_string())).collect());
    }
    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
        return Ok(str_arr.iter().map(|opt| opt.map(|s| s.to_string())).collect());
    }
    Err(compute::ArrowError::InvalidArgument("Expected string array".to_string()))
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

/// Helper for stable sort to indices
/// Arrow's sort_to_indices is already stable, this is a wrapper with explicit stable semantics
fn sort_to_indices_stable_impl(arr: &ArrayRef, descending: bool) -> Result<arrow_array::UInt32Array, compute::ArrowError> {
    let sort_opts = arrow_ord::sort::SortOptions {
        descending,
        nulls_first: false,
    };

    // Arrow's sort_to_indices uses a stable sorting algorithm (introsort with insertion sort fallback)
    arrow_ord::sort::sort_to_indices(arr, Some(sort_opts), None)
        .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))
}

/// Helper for saturating binary operations on integer arrays
fn saturating_binary_op<F>(left: &ArrayRef, right: &ArrayRef, op: F) -> Result<arrays::Array, compute::ArrowError>
where
    F: Fn(i64, i64) -> i64,
{
    use arrow_array::Array as ArrowArrayTrait;

    if left.len() != right.len() {
        return Err(compute::ArrowError::InvalidArgument("Arrays must have same length".to_string()));
    }

    macro_rules! saturate_op {
        ($left_type:ty, $right_type:ty, $result_type:ty, $native:ty) => {{
            if let (Some(l), Some(r)) = (
                left.as_any().downcast_ref::<$left_type>(),
                right.as_any().downcast_ref::<$right_type>(),
            ) {
                let result: $result_type = l.iter().zip(r.iter())
                    .map(|(a, b)| match (a, b) {
                        (Some(av), Some(bv)) => Some(op(av as i64, bv as i64) as $native),
                        _ => None,
                    })
                    .collect();
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
            }
        }};
    }

    saturate_op!(arrow_array::Int8Array, arrow_array::Int8Array, arrow_array::Int8Array, i8);
    saturate_op!(arrow_array::Int16Array, arrow_array::Int16Array, arrow_array::Int16Array, i16);
    saturate_op!(arrow_array::Int32Array, arrow_array::Int32Array, arrow_array::Int32Array, i32);
    saturate_op!(arrow_array::Int64Array, arrow_array::Int64Array, arrow_array::Int64Array, i64);

    Err(compute::ArrowError::InvalidArgument("Saturating operations require integer arrays of the same type".to_string()))
}

/// Helper for saturating scalar operations on integer arrays
fn saturating_scalar_op<F>(arr: &ArrayRef, scalar: i64, op: F) -> Result<arrays::Array, compute::ArrowError>
where
    F: Fn(i64, i64) -> i64,
{
    use arrow_array::Array as ArrowArrayTrait;

    macro_rules! saturate_scalar_op {
        ($arr_type:ty, $result_type:ty, $native:ty) => {{
            if let Some(a) = arr.as_any().downcast_ref::<$arr_type>() {
                let result: $result_type = a.iter()
                    .map(|v| v.map(|x| op(x as i64, scalar) as $native))
                    .collect();
                return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
            }
        }};
    }

    saturate_scalar_op!(arrow_array::Int8Array, arrow_array::Int8Array, i8);
    saturate_scalar_op!(arrow_array::Int16Array, arrow_array::Int16Array, i16);
    saturate_scalar_op!(arrow_array::Int32Array, arrow_array::Int32Array, i32);
    saturate_scalar_op!(arrow_array::Int64Array, arrow_array::Int64Array, i64);

    Err(compute::ArrowError::InvalidArgument("Saturating operations require integer arrays".to_string()))
}

/// Helper for case conversion
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

    fn compare_schemas(left: types::SchemaBorrow<'_>, right: types::SchemaBorrow<'_>) -> types::SchemaDiffResult {
        let left_schema = left.get::<SchemaImpl>();
        let right_schema = right.get::<SchemaImpl>();

        let left_names: std::collections::HashSet<_> = left_schema.inner.fields().iter().map(|f| f.name().clone()).collect();
        let right_names: std::collections::HashSet<_> = right_schema.inner.fields().iter().map(|f| f.name().clone()).collect();

        // Fields only in left
        let left_only: Vec<String> = left_names.difference(&right_names).cloned().collect();

        // Fields only in right
        let right_only: Vec<String> = right_names.difference(&left_names).cloned().collect();

        // Fields in both - check for type and nullability mismatches
        let common_names: Vec<String> = left_names.intersection(&right_names).cloned().collect();

        let mut type_mismatches = Vec::new();
        let mut nullability_mismatches = Vec::new();

        for name in common_names {
            let left_field = left_schema.inner.field_with_name(&name).unwrap();
            let right_field = right_schema.inner.field_with_name(&name).unwrap();

            // Check type mismatch
            if left_field.data_type() != right_field.data_type() {
                type_mismatches.push((
                    name.clone(),
                    format!("{:?}", left_field.data_type()),
                    format!("{:?}", right_field.data_type()),
                ));
            }

            // Check nullability mismatch
            if left_field.is_nullable() != right_field.is_nullable() {
                nullability_mismatches.push(name.clone());
            }
        }

        types::SchemaDiffResult {
            left_only,
            right_only,
            type_mismatches,
            nullability_mismatches,
        }
    }

    fn schemas_compatible(left: types::SchemaBorrow<'_>, right: types::SchemaBorrow<'_>) -> bool {
        let left_schema = left.get::<SchemaImpl>();
        let right_schema = right.get::<SchemaImpl>();

        // Check if each field in left can be cast to the corresponding field in right
        for left_field in left_schema.inner.fields() {
            match right_schema.inner.field_with_name(left_field.name()) {
                Ok(right_field) => {
                    // Check if types are castable
                    // Use arrow_cast::can_cast_types
                    if !arrow_cast::can_cast_types(left_field.data_type(), right_field.data_type()) {
                        return false;
                    }
                    // If left is nullable but right is not, that's incompatible
                    if left_field.is_nullable() && !right_field.is_nullable() {
                        return false;
                    }
                }
                Err(_) => {
                    // Field doesn't exist in right schema, that's okay for compatibility
                    // (extra fields can be ignored)
                }
            }
        }

        true
    }

    fn schema_merge_two(left: types::SchemaBorrow<'_>, right: types::SchemaBorrow<'_>) -> Result<types::Schema, types::ArrowError> {
        let left_schema = left.get::<SchemaImpl>();
        let right_schema = right.get::<SchemaImpl>();

        let mut merged_fields: Vec<Arc<arrow_schema::Field>> = Vec::new();
        let mut merged_metadata: HashMap<String, String> = HashMap::new();

        // Add all fields from left
        for field in left_schema.inner.fields() {
            merged_fields.push(field.clone());
        }

        // Add fields from right that aren't in left
        for field in right_schema.inner.fields() {
            if !merged_fields.iter().any(|f| f.name() == field.name()) {
                merged_fields.push(field.clone());
            }
        }

        // Merge metadata (right overrides left)
        for (k, v) in left_schema.inner.metadata() {
            merged_metadata.insert(k.clone(), v.clone());
        }
        for (k, v) in right_schema.inner.metadata() {
            merged_metadata.insert(k.clone(), v.clone());
        }

        let merged_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            arrow_schema::Fields::from(merged_fields),
            merged_metadata,
        ));

        Ok(types::Schema::new(SchemaImpl { inner: merged_schema }))
    }

    fn schema_project(schema: types::SchemaBorrow<'_>, fields: Vec<String>) -> Result<types::Schema, types::ArrowError> {
        let schema_impl = schema.get::<SchemaImpl>();

        let mut projected_fields: Vec<Arc<arrow_schema::Field>> = Vec::new();

        for field_name in &fields {
            match schema_impl.inner.field_with_name(field_name) {
                Ok(field) => projected_fields.push(Arc::new(field.clone())),
                Err(_) => {
                    return Err(types::ArrowError::InvalidArgument(format!(
                        "Field '{}' not found in schema",
                        field_name
                    )));
                }
            }
        }

        let projected_schema = Arc::new(arrow_schema::Schema::new(
            arrow_schema::Fields::from(projected_fields),
        ));

        Ok(types::Schema::new(SchemaImpl { inner: projected_schema }))
    }

    fn schema_rename(schema: types::SchemaBorrow<'_>, old_names: Vec<String>, new_names: Vec<String>) -> Result<types::Schema, types::ArrowError> {
        if old_names.len() != new_names.len() {
            return Err(types::ArrowError::InvalidArgument(
                "old_names and new_names must have the same length".to_string()
            ));
        }

        let schema_impl = schema.get::<SchemaImpl>();

        // Create a mapping from old name to new name
        let rename_map: HashMap<String, String> = old_names.iter().cloned().zip(new_names.iter().cloned()).collect();

        // Create new fields with renamed names
        let renamed_fields: Vec<Arc<arrow_schema::Field>> = schema_impl.inner.fields()
            .iter()
            .map(|field| {
                if let Some(new_name) = rename_map.get(field.name()) {
                    Arc::new(arrow_schema::Field::new(
                        new_name.clone(),
                        field.data_type().clone(),
                        field.is_nullable(),
                    ).with_metadata(field.metadata().clone()))
                } else {
                    field.clone()
                }
            })
            .collect();

        let renamed_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
            arrow_schema::Fields::from(renamed_fields),
            schema_impl.inner.metadata().clone(),
        ));

        Ok(types::Schema::new(SchemaImpl { inner: renamed_schema }))
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

    // ========== View Array Types ==========

    fn create_string_view_array(values: Vec<Option<String>>) -> arrays::Array {
        let arr: arrow_array::StringViewArray = values.into_iter().collect();
        arrays::Array::new(ArrayImpl { inner: Arc::new(arr) })
    }

    fn create_binary_view_array(values: Vec<Option<Vec<u8>>>) -> arrays::Array {
        let arr: arrow_array::BinaryViewArray = values
            .into_iter()
            .map(|opt| opt.map(|v| v.as_slice().to_vec()))
            .collect();
        arrays::Array::new(ArrayImpl { inner: Arc::new(arr) })
    }

    fn string_to_view(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let view_arr: arrow_array::StringViewArray = str_arr.iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(view_arr) }));
        }

        if let Some(large_str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            let view_arr: arrow_array::StringViewArray = large_str_arr.iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(view_arr) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Array must be String or LargeString type".to_string()))
    }

    fn binary_to_view(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            let view_arr: arrow_array::BinaryViewArray = bin_arr.iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(view_arr) }));
        }

        if let Some(large_bin_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::LargeBinaryArray>() {
            let view_arr: arrow_array::BinaryViewArray = large_bin_arr.iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(view_arr) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Array must be Binary or LargeBinary type".to_string()))
    }

    fn view_to_string(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(view_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringViewArray>() {
            let str_arr: arrow_array::StringArray = view_arr.iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(str_arr) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Array must be StringView type".to_string()))
    }

    fn view_to_binary(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();

        if let Some(view_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BinaryViewArray>() {
            let bin_arr: arrow_array::BinaryArray = view_arr.iter().collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(bin_arr) }));
        }

        Err(arrays::ArrowError::InvalidArgument("Array must be BinaryView type".to_string()))
    }

    fn is_view_type(arr: arrays::ArrayBorrow<'_>) -> bool {
        let arr_impl = arr.get::<ArrayImpl>();
        matches!(
            arr_impl.inner.data_type(),
            arrow_schema::DataType::Utf8View | arrow_schema::DataType::BinaryView
        )
    }

    // ========== Pretty Print & Display Utilities ==========

    fn array_to_string(arr: arrays::ArrayBorrow<'_>, max_rows: Option<u32>) -> String {
        let arr_impl = arr.get::<ArrayImpl>();
        let len = arr_impl.inner.len();
        let limit = max_rows.map(|m| m as usize).unwrap_or(len).min(len);

        let mut result = String::new();
        result.push_str(&format!("{} [len={}]\n", arr_impl.inner.data_type(), len));
        result.push('[');

        for i in 0..limit {
            if i > 0 {
                result.push_str(", ");
            }
            if arr_impl.inner.is_null(i) {
                result.push_str("null");
            } else {
                // Use arrow_cast display for value formatting
                use arrow_cast::display::ArrayFormatter;
                if let Ok(formatter) = ArrayFormatter::try_new(arr_impl.inner.as_ref(), &Default::default()) {
                    result.push_str(&formatter.value(i).to_string());
                } else {
                    result.push_str("<unknown>");
                }
            }
        }

        if limit < len {
            result.push_str(&format!(", ... {} more values", len - limit));
        }
        result.push(']');
        result
    }

    fn array_value_to_string(arr: arrays::ArrayBorrow<'_>, index: u64) -> Result<Option<String>, arrays::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let idx = index as usize;

        if idx >= arr_impl.inner.len() {
            return Err(arrays::ArrowError::InvalidArgument(format!(
                "Index {} out of bounds for array of length {}",
                index, arr_impl.inner.len()
            )));
        }

        if arr_impl.inner.is_null(idx) {
            return Ok(None);
        }

        use arrow_cast::display::ArrayFormatter;
        match ArrayFormatter::try_new(arr_impl.inner.as_ref(), &Default::default()) {
            Ok(formatter) => Ok(Some(formatter.value(idx).to_string())),
            Err(e) => Err(arrays::ArrowError::InvalidArgument(format!("Failed to format value: {}", e))),
        }
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

/// Extract f64 values from a numeric array
fn extract_f64_values(arr: &Arc<dyn arrow_array::Array>) -> Result<Vec<Option<f64>>, compute::ArrowError> {
    use arrow_array::Array as ArrowArrayTrait;

    if let Some(float_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        Ok((0..float_arr.len()).map(|i| {
            if float_arr.is_null(i) { None } else { Some(float_arr.value(i)) }
        }).collect())
    } else if let Some(float_arr) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        Ok((0..float_arr.len()).map(|i| {
            if float_arr.is_null(i) { None } else { Some(float_arr.value(i) as f64) }
        }).collect())
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        Ok((0..int_arr.len()).map(|i| {
            if int_arr.is_null(i) { None } else { Some(int_arr.value(i) as f64) }
        }).collect())
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        Ok((0..int_arr.len()).map(|i| {
            if int_arr.is_null(i) { None } else { Some(int_arr.value(i) as f64) }
        }).collect())
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        Ok((0..int_arr.len()).map(|i| {
            if int_arr.is_null(i) { None } else { Some(int_arr.value(i) as f64) }
        }).collect())
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        Ok((0..int_arr.len()).map(|i| {
            if int_arr.is_null(i) { None } else { Some(int_arr.value(i) as f64) }
        }).collect())
    } else if let Some(uint_arr) = arr.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        Ok((0..uint_arr.len()).map(|i| {
            if uint_arr.is_null(i) { None } else { Some(uint_arr.value(i) as f64) }
        }).collect())
    } else if let Some(uint_arr) = arr.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        Ok((0..uint_arr.len()).map(|i| {
            if uint_arr.is_null(i) { None } else { Some(uint_arr.value(i) as f64) }
        }).collect())
    } else {
        Err(compute::ArrowError::InvalidArgument("Array must be numeric for bucketing/histogram".to_string()))
    }
}

/// Extract a string representation of a value from an array at given row
fn extract_string_value(arr: &dyn arrow_array::Array, row: usize) -> String {
    use arrow_array::Array as ArrowArrayTrait;

    if arr.is_null(row) {
        return "NULL".to_string();
    }

    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        str_arr.value(row).to_string()
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        int_arr.value(row).to_string()
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        int_arr.value(row).to_string()
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        int_arr.value(row).to_string()
    } else if let Some(int_arr) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        int_arr.value(row).to_string()
    } else if let Some(uint_arr) = arr.as_any().downcast_ref::<arrow_array::UInt64Array>() {
        uint_arr.value(row).to_string()
    } else if let Some(uint_arr) = arr.as_any().downcast_ref::<arrow_array::UInt32Array>() {
        uint_arr.value(row).to_string()
    } else if let Some(float_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        float_arr.value(row).to_string()
    } else if let Some(float_arr) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        float_arr.value(row).to_string()
    } else if let Some(bool_arr) = arr.as_any().downcast_ref::<arrow_array::BooleanArray>() {
        bool_arr.value(row).to_string()
    } else {
        format!("{}", row)
    }
}

/// Helper function to aggregate values at given row indices




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

/// Refined Soundex - more detailed phonetic code than standard Soundex
fn compute_refined_soundex(s: &str) -> String {
    let chars: Vec<char> = s.to_uppercase().chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if chars.is_empty() {
        return "0".to_string();
    }

    // Refined Soundex codes (more granular than standard Soundex)
    let refined_code = |c: char| -> char {
        match c {
            'B' | 'P' => '1',
            'F' | 'V' => '2',
            'C' | 'K' | 'S' => '3',
            'G' | 'J' => '4',
            'Q' | 'X' | 'Z' => '5',
            'D' | 'T' => '6',
            'L' => '7',
            'M' | 'N' => '8',
            'R' => '9',
            _ => '0', // A, E, I, O, U, H, W, Y
        }
    };

    let mut result = String::with_capacity(10);
    result.push(chars[0]);

    let mut last_code = refined_code(chars[0]);

    for &c in &chars[1..] {
        let code = refined_code(c);
        if code != '0' && code != last_code {
            result.push(code);
        }
        last_code = code;
    }

    result
}

/// Metaphone - phonetic algorithm for English pronunciation
fn compute_metaphone(s: &str) -> String {
    let s = s.to_uppercase();
    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if chars.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(6);
    let len = chars.len();

    let get_char = |i: usize| -> char {
        if i < len { chars[i] } else { '\0' }
    };

    let mut i = 0;

    // Handle initial letter combinations
    match (get_char(0), get_char(1)) {
        ('K', 'N') | ('G', 'N') | ('P', 'N') | ('A', 'E') | ('W', 'R') => i = 1,
        ('W', 'H') => { i = 1; result.push('W'); }
        ('X', _) => { result.push('S'); i = 1; }
        _ => {}
    }

    while i < len && result.len() < 6 {
        let c = get_char(i);
        let next = get_char(i + 1);
        let prev = if i > 0 { get_char(i - 1) } else { '\0' };

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 { result.push(c); }
            }
            'B' => {
                if prev != 'M' || i + 1 < len {
                    result.push('B');
                }
            }
            'C' => {
                if next == 'I' || next == 'E' || next == 'Y' {
                    if next == 'I' && get_char(i + 2) == 'A' {
                        result.push('X');
                    } else {
                        result.push('S');
                    }
                } else if next == 'H' {
                    result.push('X');
                    i += 1;
                } else {
                    result.push('K');
                }
            }
            'D' => {
                if next == 'G' && (get_char(i + 2) == 'E' || get_char(i + 2) == 'Y' || get_char(i + 2) == 'I') {
                    result.push('J');
                    i += 1;
                } else {
                    result.push('T');
                }
            }
            'F' => result.push('F'),
            'G' => {
                if next == 'H' {
                    if i + 2 < len && !matches!(get_char(i + 2), 'A' | 'E' | 'I' | 'O' | 'U') {
                        i += 1;
                    } else {
                        result.push('K');
                        i += 1;
                    }
                } else if next == 'N' {
                    if i + 2 >= len || (i + 2 < len && get_char(i + 2) != 'E') {
                        // skip
                    } else {
                        result.push('K');
                    }
                } else if next == 'I' || next == 'E' || next == 'Y' {
                    result.push('J');
                } else {
                    result.push('K');
                }
            }
            'H' => {
                if matches!(prev, 'A' | 'E' | 'I' | 'O' | 'U') && !matches!(next, 'A' | 'E' | 'I' | 'O' | 'U') {
                    // silent
                } else if matches!(next, 'A' | 'E' | 'I' | 'O' | 'U') {
                    result.push('H');
                }
            }
            'J' => result.push('J'),
            'K' => {
                if prev != 'C' { result.push('K'); }
            }
            'L' => result.push('L'),
            'M' => result.push('M'),
            'N' => result.push('N'),
            'P' => {
                if next == 'H' {
                    result.push('F');
                    i += 1;
                } else {
                    result.push('P');
                }
            }
            'Q' => result.push('K'),
            'R' => result.push('R'),
            'S' => {
                if next == 'H' {
                    result.push('X');
                    i += 1;
                } else if next == 'I' && (get_char(i + 2) == 'O' || get_char(i + 2) == 'A') {
                    result.push('X');
                } else {
                    result.push('S');
                }
            }
            'T' => {
                if next == 'I' && (get_char(i + 2) == 'O' || get_char(i + 2) == 'A') {
                    result.push('X');
                } else if next == 'H' {
                    result.push('0');  // TH sound
                    i += 1;
                } else if next != 'C' || get_char(i + 2) != 'H' {
                    result.push('T');
                }
            }
            'V' => result.push('F'),
            'W' | 'Y' => {
                if matches!(next, 'A' | 'E' | 'I' | 'O' | 'U') {
                    result.push(c);
                }
            }
            'X' => {
                result.push('K');
                result.push('S');
            }
            'Z' => result.push('S'),
            _ => {}
        }
        i += 1;
    }

    result
}

/// Double Metaphone - returns primary or alternate encoding
fn compute_double_metaphone(s: &str, alternate: bool) -> String {
    let s = s.to_uppercase();
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();

    if len == 0 {
        return String::new();
    }

    let mut primary = String::with_capacity(6);
    let mut secondary = String::with_capacity(6);

    let get_char = |i: usize| -> char {
        if i < len { chars[i] } else { '\0' }
    };

    let is_vowel = |i: usize| -> bool {
        matches!(get_char(i), 'A' | 'E' | 'I' | 'O' | 'U')
    };

    let mut i = 0;

    // Skip initial silent letters
    if matches!((get_char(0), get_char(1)), ('G', 'N') | ('K', 'N') | ('P', 'N') | ('W', 'R') | ('P', 'S')) {
        i = 1;
    }

    // Initial X -> S
    if get_char(0) == 'X' {
        primary.push('S');
        secondary.push('S');
        i = 1;
    }

    while i < len && (primary.len() < 6 || secondary.len() < 6) {
        let c = get_char(i);

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    primary.push('A');
                    secondary.push('A');
                }
            }
            'B' => {
                primary.push('P');
                secondary.push('P');
                if get_char(i + 1) == 'B' { i += 1; }
            }
            'C' => {
                if get_char(i + 1) == 'H' {
                    primary.push('X');
                    secondary.push('X');
                    i += 1;
                } else if get_char(i + 1) == 'K' {
                    primary.push('K');
                    secondary.push('K');
                    i += 1;
                } else if matches!(get_char(i + 1), 'I' | 'E' | 'Y') {
                    primary.push('S');
                    secondary.push('S');
                } else {
                    primary.push('K');
                    secondary.push('K');
                }
            }
            'D' => {
                if get_char(i + 1) == 'G' && matches!(get_char(i + 2), 'E' | 'I' | 'Y') {
                    primary.push('J');
                    secondary.push('J');
                    i += 2;
                } else {
                    primary.push('T');
                    secondary.push('T');
                    if get_char(i + 1) == 'D' { i += 1; }
                }
            }
            'F' => {
                primary.push('F');
                secondary.push('F');
                if get_char(i + 1) == 'F' { i += 1; }
            }
            'G' => {
                if get_char(i + 1) == 'H' {
                    if i > 0 && !is_vowel(i - 1) {
                        primary.push('K');
                        secondary.push('K');
                    } else if i == 0 {
                        primary.push('J');
                        secondary.push('J');
                    }
                    i += 1;
                } else if get_char(i + 1) == 'N' {
                    if i == 0 && is_vowel(1) {
                        primary.push('K');
                        secondary.push('N');
                    } else if get_char(i + 2) != 'E' || get_char(i + 3) != 'Y' {
                        primary.push('N');
                        secondary.push('N');
                    }
                } else if matches!(get_char(i + 1), 'I' | 'E' | 'Y') {
                    primary.push('J');
                    secondary.push('K');
                } else {
                    primary.push('K');
                    secondary.push('K');
                    if get_char(i + 1) == 'G' { i += 1; }
                }
            }
            'H' => {
                if (i == 0 || is_vowel(i - 1)) && is_vowel(i + 1) {
                    primary.push('H');
                    secondary.push('H');
                }
            }
            'J' => {
                primary.push('J');
                secondary.push('J');
                if get_char(i + 1) == 'J' { i += 1; }
            }
            'K' => {
                primary.push('K');
                secondary.push('K');
                if get_char(i + 1) == 'K' { i += 1; }
            }
            'L' => {
                primary.push('L');
                secondary.push('L');
                if get_char(i + 1) == 'L' { i += 1; }
            }
            'M' => {
                primary.push('M');
                secondary.push('M');
                if get_char(i + 1) == 'M' { i += 1; }
            }
            'N' => {
                primary.push('N');
                secondary.push('N');
                if get_char(i + 1) == 'N' { i += 1; }
            }
            'P' => {
                if get_char(i + 1) == 'H' {
                    primary.push('F');
                    secondary.push('F');
                    i += 1;
                } else {
                    primary.push('P');
                    secondary.push('P');
                    if get_char(i + 1) == 'P' { i += 1; }
                }
            }
            'Q' => {
                primary.push('K');
                secondary.push('K');
                if get_char(i + 1) == 'Q' { i += 1; }
            }
            'R' => {
                primary.push('R');
                secondary.push('R');
                if get_char(i + 1) == 'R' { i += 1; }
            }
            'S' => {
                if get_char(i + 1) == 'H' {
                    primary.push('X');
                    secondary.push('X');
                    i += 1;
                } else if get_char(i + 1) == 'C' && get_char(i + 2) == 'H' {
                    primary.push('X');
                    secondary.push('X');
                    i += 2;
                } else {
                    primary.push('S');
                    secondary.push('S');
                    if get_char(i + 1) == 'S' { i += 1; }
                }
            }
            'T' => {
                if get_char(i + 1) == 'H' {
                    primary.push('0');  // TH
                    secondary.push('T');
                    i += 1;
                } else if get_char(i + 1) == 'C' && get_char(i + 2) == 'H' {
                    // skip
                } else {
                    primary.push('T');
                    secondary.push('T');
                    if get_char(i + 1) == 'T' { i += 1; }
                }
            }
            'V' => {
                primary.push('F');
                secondary.push('F');
                if get_char(i + 1) == 'V' { i += 1; }
            }
            'W' => {
                if is_vowel(i + 1) {
                    primary.push('A');
                    secondary.push('F');
                }
            }
            'X' => {
                primary.push_str("KS");
                secondary.push_str("KS");
                if get_char(i + 1) == 'X' { i += 1; }
            }
            'Y' => {
                if is_vowel(i + 1) {
                    primary.push('A');
                    secondary.push('A');
                }
            }
            'Z' => {
                primary.push('S');
                secondary.push('S');
                if get_char(i + 1) == 'Z' { i += 1; }
            }
            _ => {}
        }
        i += 1;
    }

    if alternate { secondary } else { primary }
}

/// NYSIIS - New York State Identification and Intelligence System
fn compute_nysiis(s: &str) -> String {
    let s = s.to_uppercase();
    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if chars.is_empty() {
        return String::new();
    }

    let mut result: Vec<char> = chars.clone();

    // Handle initial patterns
    if result.len() >= 3 {
        let start: String = result.iter().take(3).collect();
        if start == "MAC" {
            result.splice(0..3, "MCC".chars());
        } else if start == "KN " || (result.len() >= 2 && result[0] == 'K' && result[1] == 'N') {
            result.remove(0);
        }
    }
    if result.len() >= 2 {
        let start: String = result.iter().take(2).collect();
        if start == "PH" {
            result.splice(0..2, "FF".chars());
        } else if start == "PF" {
            result.splice(0..2, "FF".chars());
        } else if result[0] == 'K' {
            result[0] = 'C';
        }
    }
    if result.len() >= 3 && result[0] == 'S' && result[1] == 'C' && result[2] == 'H' {
        result.splice(0..3, "SSS".chars());
    }

    // Handle ending patterns
    let len = result.len();
    if len >= 2 {
        let end: String = result.iter().skip(len - 2).collect();
        if end == "EE" || end == "IE" {
            result.truncate(len - 2);
            result.push('Y');
        } else if end == "DT" || end == "RT" || end == "RD" || end == "NT" || end == "ND" {
            result.truncate(len - 2);
            result.push('D');
        }
    }

    // First character is kept
    let first = result[0];

    // Process remaining characters
    let mut encoded = String::with_capacity(8);
    encoded.push(first);

    let mut i = 1;
    while i < result.len() {
        let c = result[i];
        let prev = if i > 0 { result[i - 1] } else { '\0' };
        let next = if i + 1 < result.len() { result[i + 1] } else { '\0' };

        let replacement = match c {
            'E' | 'I' | 'O' | 'U' => 'A',
            'Q' => 'G',
            'Z' => 'S',
            'M' => 'N',
            'K' => if next == 'N' { 'N' } else { 'C' },
            'S' if next == 'C' && i + 2 < result.len() && result[i + 2] == 'H' => 'S',
            'P' if next == 'H' => 'F',
            'H' if !matches!(prev, 'A' | 'E' | 'I' | 'O' | 'U') || !matches!(next, 'A' | 'E' | 'I' | 'O' | 'U') => prev,
            'W' if matches!(prev, 'A' | 'E' | 'I' | 'O' | 'U') => prev,
            _ => c,
        };

        // Don't add consecutive duplicates
        if encoded.chars().last() != Some(replacement) {
            encoded.push(replacement);
        }

        i += 1;
    }

    // Remove trailing S
    if encoded.len() > 1 && encoded.ends_with('S') {
        encoded.pop();
    }

    // Remove trailing A (if not the only character)
    if encoded.len() > 1 && encoded.ends_with('A') {
        encoded.pop();
    }

    // Limit to 6 characters
    encoded.truncate(6);
    encoded
}

/// Cologne Phonetics - phonetic algorithm for German language
fn compute_cologne(s: &str) -> String {
    let s = s.to_uppercase();
    let chars: Vec<char> = s.chars().filter(|c| c.is_ascii_alphabetic()).collect();

    if chars.is_empty() {
        return String::new();
    }

    let len = chars.len();

    let get_char = |i: usize| -> char {
        if i < len { chars[i] } else { '\0' }
    };

    let mut result = String::with_capacity(len);

    for i in 0..len {
        let c = get_char(i);
        let prev = if i > 0 { get_char(i - 1) } else { '\0' };
        let next = get_char(i + 1);

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' | 'J' | 'Y' => {
                result.push('0');
            }
            'H' => continue, // H is ignored
            'B' => {
                result.push('1');
            }
            'P' => {
                result.push(if next == 'H' { '3' } else { '1' });
            }
            'D' | 'T' => {
                result.push(if matches!(next, 'C' | 'S' | 'Z') { '8' } else { '2' });
            }
            'F' | 'V' | 'W' => {
                result.push('3');
            }
            'G' | 'K' | 'Q' => {
                result.push('4');
            }
            'X' => {
                if !matches!(prev, 'C' | 'K' | 'Q') {
                    result.push('4');
                    result.push('8');
                } else {
                    result.push('8');
                }
            }
            'L' => {
                result.push('5');
            }
            'M' | 'N' => {
                result.push('6');
            }
            'R' => {
                result.push('7');
            }
            'S' | 'Z' => {
                result.push('8');
            }
            'C' => {
                let code = if i == 0 {
                    if matches!(next, 'A' | 'H' | 'K' | 'L' | 'O' | 'Q' | 'R' | 'U' | 'X') { '4' }
                    else { '8' }
                } else if matches!(prev, 'S' | 'Z') {
                    '8'
                } else if matches!(next, 'A' | 'H' | 'K' | 'O' | 'Q' | 'U' | 'X') {
                    '4'
                } else {
                    '8'
                };
                result.push(code);
            }
            _ => continue,
        }
    }

    // Remove consecutive duplicates
    let mut final_result = String::with_capacity(result.len());
    let mut last = '\0';
    for c in result.chars() {
        if c != last {
            final_result.push(c);
            last = c;
        }
    }

    // Remove all '0's except if it's the only character or at the start
    if final_result.len() > 1 {
        let first = final_result.chars().next().unwrap();
        let rest: String = final_result.chars().skip(1).filter(|&c| c != '0').collect();
        final_result = String::new();
        final_result.push(first);
        final_result.push_str(&rest);
    }

    // Remove leading '0' if there are other characters
    while final_result.len() > 1 && final_result.starts_with('0') {
        final_result.remove(0);
    }

    final_result
}

/// Encode a string using the specified phonetic algorithm

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

    // ========== Batch Set Operations ==========

    fn except_batches(
        left: record_batch::RecordBatchBorrow<'_>,
        right: record_batch::RecordBatchBorrow<'_>,
        all_rows: bool,
    ) -> Result<record_batch::RecordBatch, types::ArrowError> {
        use arrow_row::{RowConverter, SortField};
        use std::collections::{HashMap, HashSet};

        let left_impl = left.get::<RecordBatchImpl>();
        let right_impl = right.get::<RecordBatchImpl>();

        // Verify schemas match
        if left_impl.inner.schema() != right_impl.inner.schema() {
            return Err(types::ArrowError::SchemaMismatch(
                "EXCEPT requires batches with identical schemas".to_string()
            ));
        }

        let schema = left_impl.inner.schema();
        let sort_fields: Vec<SortField> = schema.fields().iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();

        let converter = RowConverter::new(sort_fields)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Convert right to row format for lookup
        let right_columns: Vec<ArrayRef> = right_impl.inner.columns().to_vec();
        let right_rows = converter.convert_columns(&right_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Build set/counts from right side
        let mut right_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for row in right_rows.iter() {
            *right_counts.entry(row.as_ref().to_vec()).or_insert(0) += 1;
        }

        // Convert left to row format
        let left_columns: Vec<ArrayRef> = left_impl.inner.columns().to_vec();
        let left_rows = converter.convert_columns(&left_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Find rows in left that are not in right
        let mut result_indices: Vec<u64> = Vec::new();
        let mut seen_left: HashSet<Vec<u8>> = HashSet::new();
        let mut remaining_right = right_counts.clone();

        for (i, row) in left_rows.iter().enumerate() {
            let key = row.as_ref().to_vec();

            if all_rows {
                // EXCEPT ALL: for each row in left, subtract one from right count
                if let Some(count) = remaining_right.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        continue; // Skip this row
                    }
                }
                result_indices.push(i as u64);
            } else {
                // EXCEPT DISTINCT: row in left and not in right, deduplicated
                if !right_counts.contains_key(&key) && seen_left.insert(key) {
                    result_indices.push(i as u64);
                }
            }
        }

        // Take the result rows
        let indices_arr = arrow_array::UInt64Array::from(result_indices);
        let result_columns: Result<Vec<ArrayRef>, _> = left_impl.inner.columns().iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &indices_arr, None))
            .collect();
        let result_columns = result_columns.map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        let result = ArrowRecordBatch::try_new(schema.clone(), result_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn intersect_batches(
        left: record_batch::RecordBatchBorrow<'_>,
        right: record_batch::RecordBatchBorrow<'_>,
        all_rows: bool,
    ) -> Result<record_batch::RecordBatch, types::ArrowError> {
        use arrow_row::{RowConverter, SortField};
        use std::collections::{HashMap, HashSet};

        let left_impl = left.get::<RecordBatchImpl>();
        let right_impl = right.get::<RecordBatchImpl>();

        // Verify schemas match
        if left_impl.inner.schema() != right_impl.inner.schema() {
            return Err(types::ArrowError::SchemaMismatch(
                "INTERSECT requires batches with identical schemas".to_string()
            ));
        }

        let schema = left_impl.inner.schema();
        let sort_fields: Vec<SortField> = schema.fields().iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();

        let converter = RowConverter::new(sort_fields)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Convert right to row format for lookup
        let right_columns: Vec<ArrayRef> = right_impl.inner.columns().to_vec();
        let right_rows = converter.convert_columns(&right_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Build counts from right side
        let mut right_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for row in right_rows.iter() {
            *right_counts.entry(row.as_ref().to_vec()).or_insert(0) += 1;
        }

        // Convert left to row format
        let left_columns: Vec<ArrayRef> = left_impl.inner.columns().to_vec();
        let left_rows = converter.convert_columns(&left_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Find rows in both left and right
        let mut result_indices: Vec<u64> = Vec::new();
        let mut seen_left: HashSet<Vec<u8>> = HashSet::new();
        let mut remaining_right = right_counts.clone();

        for (i, row) in left_rows.iter().enumerate() {
            let key = row.as_ref().to_vec();

            if all_rows {
                // INTERSECT ALL: for each row in left, take if count in right > 0
                if let Some(count) = remaining_right.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        result_indices.push(i as u64);
                    }
                }
            } else {
                // INTERSECT DISTINCT: row in both, deduplicated
                if right_counts.contains_key(&key) && seen_left.insert(key) {
                    result_indices.push(i as u64);
                }
            }
        }

        // Take the result rows
        let indices_arr = arrow_array::UInt64Array::from(result_indices);
        let result_columns: Result<Vec<ArrayRef>, _> = left_impl.inner.columns().iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &indices_arr, None))
            .collect();
        let result_columns = result_columns.map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        let result = ArrowRecordBatch::try_new(schema.clone(), result_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn union_batches(
        batches: Vec<record_batch::RecordBatch>,
        all_rows: bool,
    ) -> Result<record_batch::RecordBatch, types::ArrowError> {
        if batches.is_empty() {
            return Err(types::ArrowError::InvalidArgument("union_batches requires at least one batch".to_string()));
        }

        // Get schema from first batch
        let first_impl = batches[0].get::<RecordBatchImpl>();
        let schema = first_impl.inner.schema();

        // Verify all schemas match
        for batch in batches.iter().skip(1) {
            let batch_impl = batch.get::<RecordBatchImpl>();
            if batch_impl.inner.schema() != schema {
                return Err(types::ArrowError::SchemaMismatch(
                    "UNION requires all batches to have identical schemas".to_string()
                ));
            }
        }

        // Collect all rows
        let inner_batches: Vec<&ArrowRecordBatch> = batches.iter()
            .map(|b| &b.get::<RecordBatchImpl>().inner)
            .collect();

        // Concatenate all batches
        let concatenated = arrow_select::concat::concat_batches(&schema, inner_batches.iter().copied())
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        if all_rows {
            // UNION ALL: just return concatenated
            return Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: concatenated }));
        }

        // UNION DISTINCT: remove duplicates
        use arrow_row::{RowConverter, SortField};
        use std::collections::HashSet;

        let sort_fields: Vec<SortField> = schema.fields().iter()
            .map(|f| SortField::new(f.data_type().clone()))
            .collect();

        let converter = RowConverter::new(sort_fields)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        let columns: Vec<ArrayRef> = concatenated.columns().to_vec();
        let rows = converter.convert_columns(&columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Find distinct rows
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut result_indices: Vec<u64> = Vec::new();

        for (i, row) in rows.iter().enumerate() {
            let key = row.as_ref().to_vec();
            if seen.insert(key) {
                result_indices.push(i as u64);
            }
        }

        // Take the distinct rows
        let indices_arr = arrow_array::UInt64Array::from(result_indices);
        let result_columns: Result<Vec<ArrayRef>, _> = concatenated.columns().iter()
            .map(|col| arrow_select::take::take(col.as_ref(), &indices_arr, None))
            .collect();
        let result_columns = result_columns.map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        let result = ArrowRecordBatch::try_new(schema.clone(), result_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    // ========== Pivot/Unpivot Operations ==========

    fn pivot(
        batch: record_batch::RecordBatchBorrow<'_>,
        index_columns: Vec<String>,
        pivot_column: String,
        value_column: String,
        agg_function: String,
    ) -> Result<record_batch::RecordBatch, types::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;
        use std::collections::{HashMap, BTreeSet};

        let batch_impl = batch.get::<RecordBatchImpl>();
        let inner = &batch_impl.inner;

        // Get pivot column
        let pivot_col = inner.column_by_name(&pivot_column)
            .ok_or_else(|| types::ArrowError::InvalidArgument(format!("Pivot column '{}' not found", pivot_column)))?;

        // Get value column
        let value_col = inner.column_by_name(&value_column)
            .ok_or_else(|| types::ArrowError::InvalidArgument(format!("Value column '{}' not found", value_column)))?;

        // Get index columns
        let index_cols: Vec<(&str, ArrayRef)> = index_columns.iter()
            .map(|name| {
                inner.column_by_name(name)
                    .map(|col| (name.as_str(), col.clone()))
                    .ok_or_else(|| types::ArrowError::InvalidArgument(format!("Index column '{}' not found", name)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Extract pivot values
        let pivot_str = pivot_col.as_any().downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| types::ArrowError::InvalidArgument("Pivot column must be String type".to_string()))?;

        // Get unique pivot values
        let mut pivot_values: BTreeSet<String> = BTreeSet::new();
        for v in pivot_str.iter().flatten() {
            pivot_values.insert(v.to_string());
        }

        // Group by index columns and pivot column, aggregate values
        // For simplicity, we'll build a map: (index_key, pivot_val) -> aggregated_value
        use arrow_row::{RowConverter, SortField};

        // Build index key converter
        let sort_fields: Vec<SortField> = index_cols.iter()
            .map(|(_, col)| SortField::new(col.data_type().clone()))
            .collect();

        if sort_fields.is_empty() {
            return Err(types::ArrowError::InvalidArgument("At least one index column required".to_string()));
        }

        let converter = RowConverter::new(sort_fields)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        let index_arrays: Vec<ArrayRef> = index_cols.iter().map(|(_, col)| col.clone()).collect();
        let rows = converter.convert_columns(&index_arrays)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Map: index_key -> (pivot_val -> values)
        let mut aggregations: HashMap<Vec<u8>, HashMap<String, Vec<f64>>> = HashMap::new();

        let value_f64 = value_col.as_any().downcast_ref::<arrow_array::Float64Array>();
        let value_i64 = value_col.as_any().downcast_ref::<arrow_array::Int64Array>();

        for i in 0..inner.num_rows() {
            let index_key = rows.row(i).as_ref().to_vec();

            if let Some(pv) = pivot_str.value(i).into() {
                let pivot_val = pv.to_string();

                let val = if let Some(f64_arr) = value_f64 {
                    if !f64_arr.is_null(i) { Some(f64_arr.value(i)) } else { None }
                } else if let Some(i64_arr) = value_i64 {
                    if !i64_arr.is_null(i) { Some(i64_arr.value(i) as f64) } else { None }
                } else {
                    None
                };

                if let Some(v) = val {
                    aggregations.entry(index_key)
                        .or_default()
                        .entry(pivot_val)
                        .or_default()
                        .push(v);
                }
            }
        }

        // Apply aggregation function
        fn aggregate(values: &[f64], func: &str) -> Option<f64> {
            if values.is_empty() { return None; }
            match func.to_lowercase().as_str() {
                "sum" => Some(values.iter().sum()),
                "avg" | "mean" => Some(values.iter().sum::<f64>() / values.len() as f64),
                "min" => values.iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()),
                "max" => values.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()),
                "count" => Some(values.len() as f64),
                "first" => values.first().cloned(),
                "last" => values.last().cloned(),
                _ => Some(values.iter().sum()), // Default to sum
            }
        }

        // Build result schema
        let mut fields: Vec<arrow_schema::Field> = index_cols.iter()
            .map(|(name, col)| arrow_schema::Field::new(*name, col.data_type().clone(), true))
            .collect();

        for pv in &pivot_values {
            fields.push(arrow_schema::Field::new(pv, arrow_schema::DataType::Float64, true));
        }

        let result_schema = Arc::new(arrow_schema::Schema::new(fields));

        // Get unique index keys
        let unique_keys: Vec<Vec<u8>> = aggregations.keys().cloned().collect();

        // Build result columns
        let mut result_columns: Vec<ArrayRef> = Vec::new();

        // Index columns: need to take first row for each unique key
        let mut key_to_row: HashMap<Vec<u8>, usize> = HashMap::new();
        for i in 0..inner.num_rows() {
            let key = rows.row(i).as_ref().to_vec();
            key_to_row.entry(key).or_insert(i);
        }

        let indices: Vec<u64> = unique_keys.iter()
            .filter_map(|k| key_to_row.get(k).map(|&i| i as u64))
            .collect();
        let indices_arr = arrow_array::UInt64Array::from(indices);

        for (_, col) in &index_cols {
            let taken = arrow_select::take::take(col.as_ref(), &indices_arr, None)
                .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;
            result_columns.push(taken);
        }

        // Pivot value columns
        for pv in &pivot_values {
            let values: Vec<Option<f64>> = unique_keys.iter()
                .map(|key| {
                    aggregations.get(key)
                        .and_then(|pivot_map| pivot_map.get(pv))
                        .and_then(|vals| aggregate(vals, &agg_function))
                })
                .collect();
            let arr: arrow_array::Float64Array = values.into_iter().collect();
            result_columns.push(Arc::new(arr));
        }

        let result = ArrowRecordBatch::try_new(result_schema, result_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn unpivot(
        batch: record_batch::RecordBatchBorrow<'_>,
        id_columns: Vec<String>,
        value_columns: Vec<String>,
        variable_name: String,
        value_name: String,
    ) -> Result<record_batch::RecordBatch, types::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;

        let batch_impl = batch.get::<RecordBatchImpl>();
        let inner = &batch_impl.inner;
        let num_rows = inner.num_rows();

        if value_columns.is_empty() {
            return Err(types::ArrowError::InvalidArgument("Value columns cannot be empty".to_string()));
        }

        // Get ID columns
        let id_cols: Vec<ArrayRef> = id_columns.iter()
            .map(|name| {
                inner.column_by_name(name)
                    .cloned()
                    .ok_or_else(|| types::ArrowError::InvalidArgument(format!("ID column '{}' not found", name)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Get value columns
        let val_cols: Vec<ArrayRef> = value_columns.iter()
            .map(|name| {
                inner.column_by_name(name)
                    .cloned()
                    .ok_or_else(|| types::ArrowError::InvalidArgument(format!("Value column '{}' not found", name)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Result will have num_rows * num_value_columns rows
        let result_rows = num_rows * value_columns.len();

        // Repeat ID columns
        let mut result_columns: Vec<ArrayRef> = Vec::new();

        for id_col in &id_cols {
            // Repeat each row value_columns.len() times
            let mut indices: Vec<u64> = Vec::with_capacity(result_rows);
            for i in 0..num_rows {
                for _ in 0..value_columns.len() {
                    indices.push(i as u64);
                }
            }
            let indices_arr = arrow_array::UInt64Array::from(indices);
            let repeated = arrow_select::take::take(id_col.as_ref(), &indices_arr, None)
                .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;
            result_columns.push(repeated);
        }

        // Variable column (column names)
        let mut variable_values: Vec<Option<String>> = Vec::with_capacity(result_rows);
        for _ in 0..num_rows {
            for col_name in &value_columns {
                variable_values.push(Some(col_name.clone()));
            }
        }
        let variable_arr: arrow_array::StringArray = variable_values.into_iter().collect();
        result_columns.push(Arc::new(variable_arr));

        // Value column - interleave values from all value columns
        // Determine common type - use Float64 for simplicity
        let mut value_values: Vec<Option<f64>> = Vec::with_capacity(result_rows);

        for row_idx in 0..num_rows {
            for val_col in &val_cols {
                if let Some(f64_arr) = val_col.as_any().downcast_ref::<arrow_array::Float64Array>() {
                    if !f64_arr.is_null(row_idx) {
                        value_values.push(Some(f64_arr.value(row_idx)));
                    } else {
                        value_values.push(None);
                    }
                } else if let Some(i64_arr) = val_col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                    if !i64_arr.is_null(row_idx) {
                        value_values.push(Some(i64_arr.value(row_idx) as f64));
                    } else {
                        value_values.push(None);
                    }
                } else if let Some(i32_arr) = val_col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                    if !i32_arr.is_null(row_idx) {
                        value_values.push(Some(i32_arr.value(row_idx) as f64));
                    } else {
                        value_values.push(None);
                    }
                } else {
                    value_values.push(None);
                }
            }
        }
        let value_arr: arrow_array::Float64Array = value_values.into_iter().collect();
        result_columns.push(Arc::new(value_arr));

        // Build schema
        let mut fields: Vec<arrow_schema::Field> = id_columns.iter()
            .zip(id_cols.iter())
            .map(|(name, col)| arrow_schema::Field::new(name, col.data_type().clone(), true))
            .collect();
        fields.push(arrow_schema::Field::new(&variable_name, arrow_schema::DataType::Utf8, false));
        fields.push(arrow_schema::Field::new(&value_name, arrow_schema::DataType::Float64, true));

        let result_schema = Arc::new(arrow_schema::Schema::new(fields));

        let result = ArrowRecordBatch::try_new(result_schema, result_columns)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn stack_arrays(
        arrays: Vec<arrays::Array>,
        labels: Vec<String>,
    ) -> Result<record_batch::RecordBatch, types::ArrowError> {
        use arrow_array::Array as ArrowArrayTrait;

        if arrays.is_empty() {
            return Err(types::ArrowError::InvalidArgument("Arrays cannot be empty".to_string()));
        }

        if arrays.len() != labels.len() {
            return Err(types::ArrowError::InvalidArgument("Arrays and labels must have same length".to_string()));
        }

        // Get total length
        let total_len: usize = arrays.iter()
            .map(|a| a.get::<ArrayImpl>().inner.len())
            .sum();

        // Build label column
        let mut label_values: Vec<Option<String>> = Vec::with_capacity(total_len);
        for (arr, label) in arrays.iter().zip(labels.iter()) {
            let arr_impl = arr.get::<ArrayImpl>();
            for _ in 0..arr_impl.inner.len() {
                label_values.push(Some(label.clone()));
            }
        }
        let label_arr: arrow_array::StringArray = label_values.into_iter().collect();

        // Concatenate arrays
        let inner_arrays: Vec<&dyn arrow_array::Array> = arrays.iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();

        let concatenated = arrow_select::concat::concat(&inner_arrays)
            .map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        // Build schema
        let data_type = arrays[0].get::<ArrayImpl>().inner.data_type().clone();
        let fields = vec![
            arrow_schema::Field::new("label", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("value", data_type, true),
        ];
        let schema = Arc::new(arrow_schema::Schema::new(fields));

        let result = ArrowRecordBatch::try_new(
            schema,
            vec![Arc::new(label_arr), concatenated],
        ).map_err(|e| types::ArrowError::ComputeError(e.to_string()))?;

        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    // ========== Pretty Print & Display Utilities ==========

    fn pretty_print_batch(batch: record_batch::RecordBatchBorrow<'_>, max_rows: Option<u32>) -> String {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let schema = batch_impl.inner.schema();
        let num_rows = batch_impl.inner.num_rows();
        let num_cols = batch_impl.inner.num_columns();
        let limit = max_rows.map(|m| m as usize).unwrap_or(num_rows).min(num_rows);

        // Calculate column widths
        let mut col_widths: Vec<usize> = schema.fields()
            .iter()
            .map(|f| f.name().len())
            .collect();

        // Update widths based on data
        for row in 0..limit {
            for (col, width) in col_widths.iter_mut().enumerate() {
                let arr = batch_impl.inner.column(col);
                let value_len = if arr.is_null(row) {
                    4 // "null"
                } else {
                    use arrow_cast::display::ArrayFormatter;
                    if let Ok(formatter) = ArrayFormatter::try_new(arr.as_ref(), &Default::default()) {
                        formatter.value(row).to_string().len()
                    } else {
                        7 // "<error>"
                    }
                };
                *width = (*width).max(value_len);
            }
        }

        let mut result = String::new();

        // Header
        result.push('|');
        for (field, width) in schema.fields().iter().zip(col_widths.iter()) {
            result.push_str(&format!(" {:^width$} |", field.name(), width = *width));
        }
        result.push('\n');

        // Separator
        result.push('|');
        for width in &col_widths {
            result.push_str(&"-".repeat(*width + 2));
            result.push('|');
        }
        result.push('\n');

        // Data rows
        for row in 0..limit {
            result.push('|');
            for (col, width) in col_widths.iter().enumerate() {
                let arr = batch_impl.inner.column(col);
                let value = if arr.is_null(row) {
                    "null".to_string()
                } else {
                    use arrow_cast::display::ArrayFormatter;
                    if let Ok(formatter) = ArrayFormatter::try_new(arr.as_ref(), &Default::default()) {
                        formatter.value(row).to_string()
                    } else {
                        "<error>".to_string()
                    }
                };
                result.push_str(&format!(" {:>width$} |", value, width = *width));
            }
            result.push('\n');
        }

        if limit < num_rows {
            result.push_str(&format!("... {} more rows\n", num_rows - limit));
        }

        result.push_str(&format!("\n{} rows x {} columns", num_rows, num_cols));
        result
    }

    fn batch_to_csv_string(batch: record_batch::RecordBatchBorrow<'_>) -> Result<String, types::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let schema = batch_impl.inner.schema();
        let num_rows = batch_impl.inner.num_rows();

        let mut result = String::new();

        // Header
        let headers: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        result.push_str(&headers.join(","));
        result.push('\n');

        // Data rows
        for row in 0..num_rows {
            let mut values: Vec<String> = Vec::with_capacity(batch_impl.inner.num_columns());
            for col in 0..batch_impl.inner.num_columns() {
                let arr = batch_impl.inner.column(col);
                let value = if arr.is_null(row) {
                    String::new()
                } else {
                    use arrow_cast::display::ArrayFormatter;
                    match ArrayFormatter::try_new(arr.as_ref(), &Default::default()) {
                        Ok(formatter) => {
                            let val = formatter.value(row).to_string();
                            // Escape CSV values with commas or quotes
                            if val.contains(',') || val.contains('"') || val.contains('\n') {
                                format!("\"{}\"", val.replace('"', "\"\""))
                            } else {
                                val
                            }
                        }
                        Err(e) => return Err(types::ArrowError::ComputeError(format!("Failed to format value: {}", e))),
                    }
                };
                values.push(value);
            }
            result.push_str(&values.join(","));
            result.push('\n');
        }

        Ok(result)
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
// Phase 16 Helper Functions
// ============================================================================

/// Linear interpolation for missing values
fn interpolate_linear(values: &[Option<f64>], limit: usize) -> Vec<Option<f64>> {
    let mut result = values.to_vec();
    let n = result.len();

    let mut i = 0;
    while i < n {
        if result[i].is_none() {
            // Find start of null run
            let start = i;
            // Find end of null run
            while i < n && result[i].is_none() {
                i += 1;
            }
            let end = i;
            let gap_len = end - start;

            // Only interpolate if within limit
            if gap_len <= limit {
                // Get boundary values
                let left_val = if start > 0 { result[start - 1] } else { None };
                let right_val = if end < n { result[end] } else { None };

                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        // Linear interpolation
                        for j in start..end {
                            let t = (j - start + 1) as f64 / (gap_len + 1) as f64;
                            result[j] = Some(l + t * (r - l));
                        }
                    }
                    _ => {} // Can't interpolate without both boundaries
                }
            }
        } else {
            i += 1;
        }
    }

    result
}

/// Nearest neighbor interpolation
fn interpolate_nearest(values: &[Option<f64>], limit: usize) -> Vec<Option<f64>> {
    let mut result = values.to_vec();
    let n = result.len();

    for i in 0..n {
        if result[i].is_none() {
            // Find nearest non-null value
            let mut left_dist = None;
            let mut left_val = None;
            let mut right_dist = None;
            let mut right_val = None;

            // Search left
            for j in (0..i).rev() {
                if let Some(v) = values[j] {
                    left_dist = Some(i - j);
                    left_val = Some(v);
                    break;
                }
            }

            // Search right
            for j in (i + 1)..n {
                if let Some(v) = values[j] {
                    right_dist = Some(j - i);
                    right_val = Some(v);
                    break;
                }
            }

            // Use nearest (prefer left if equal distance)
            let fill = match (left_dist, right_dist) {
                (Some(ld), Some(rd)) => {
                    if ld <= rd && ld <= limit { left_val }
                    else if rd <= limit { right_val }
                    else { None }
                }
                (Some(ld), None) if ld <= limit => left_val,
                (None, Some(rd)) if rd <= limit => right_val,
                _ => None,
            };

            result[i] = fill;
        }
    }

    result
}

/// Forward fill (LOCF - Last Observation Carried Forward)
fn interpolate_forward(values: &[Option<f64>], limit: usize) -> Vec<Option<f64>> {
    let mut result = values.to_vec();
    let mut last_val: Option<f64> = None;
    let mut gap_count = 0usize;

    for i in 0..result.len() {
        if result[i].is_some() {
            last_val = result[i];
            gap_count = 0;
        } else if let Some(v) = last_val {
            gap_count += 1;
            if gap_count <= limit {
                result[i] = Some(v);
            }
        }
    }

    result
}

/// Backward fill (NOCB - Next Observation Carried Backward)
fn interpolate_backward(values: &[Option<f64>], limit: usize) -> Vec<Option<f64>> {
    let mut result = values.to_vec();
    let n = result.len();
    let mut next_val: Option<f64> = None;
    let mut gap_count = 0usize;

    for i in (0..n).rev() {
        if result[i].is_some() {
            next_val = result[i];
            gap_count = 0;
        } else if let Some(v) = next_val {
            gap_count += 1;
            if gap_count <= limit {
                result[i] = Some(v);
            }
        }
    }

    result
}

/// Aggregate a slice of f64 values using specified function

/// FNV-1a 64-bit hash
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// CRC32 hash
fn crc32_hash(data: &[u8]) -> u32 {
    const CRC32_TABLE: [u32; 256] = [
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
        0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
        0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
        0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
        0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
        0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
        0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
        0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
        0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
        0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
        0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
        0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
        0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
        0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cd9,
        0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
        0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
        0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
        0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
        0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
        0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
        0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
        0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
        0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
        0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
        0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
        0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
        0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
        0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
        0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
        0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
        0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd706b3, 0x54de5729, 0x23d967bf,
        0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
    ];

    let mut crc = 0xffffffff_u32;
    for byte in data {
        let index = ((crc ^ (*byte as u32)) & 0xff) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xffffffff
}

/// Simple pseudo-random number generator (xorshift64)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next() as usize) % max
    }
}

/// Shuffle a vector in place using Fisher-Yates algorithm
fn shuffle_vec<T>(vec: &mut [T], rng: &mut SimpleRng) {
    for i in (1..vec.len()).rev() {
        let j = rng.next_usize(i + 1);
        vec.swap(i, j);
    }
}

/// Compute string similarity using specified algorithm (returns 0.0 to 1.0)
fn compute_string_similarity(s1: &str, s2: &str, algorithm: &str) -> f64 {
    match algorithm.to_lowercase().as_str() {
        "levenshtein" | "edit" => {
            let dist = levenshtein_distance(s1, s2);
            let max_len = s1.len().max(s2.len());
            if max_len == 0 { 1.0 }
            else { 1.0 - (dist as f64 / max_len as f64) }
        }
        "jaro" | "jaro_winkler" | "jaro-winkler" => {
            jaro_winkler_similarity(s1, s2)
        }
        _ => {
            // Default to Levenshtein
            let dist = levenshtein_distance(s1, s2);
            let max_len = s1.len().max(s2.len());
            if max_len == 0 { 1.0 }
            else { 1.0 - (dist as f64 / max_len as f64) }
        }
    }
}

/// Levenshtein edit distance between two strings
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Jaro-Winkler similarity between two strings
fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let m = s1_chars.len();
    let n = s2_chars.len();

    if m == 0 && n == 0 { return 1.0; }
    if m == 0 || n == 0 { return 0.0; }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; m];
    let mut s2_matches = vec![false; n];
    let mut matches = 0;
    let mut transpositions = 0;

    // Find matches
    for i in 0..m {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(n);

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
    for i in 0..m {
        if !s1_matches[i] { continue; }
        while !s2_matches[k] { k += 1; }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let jaro = (matches as f64 / m as f64
              + matches as f64 / n as f64
              + (matches - transpositions / 2) as f64 / matches as f64) / 3.0;

    // Winkler adjustment for common prefix
    let mut prefix = 0;
    for i in 0..4.min(m).min(n) {
        if s1_chars[i] == s2_chars[i] {
            prefix += 1;
        } else {
            break;
        }
    }

    jaro + (prefix as f64 * 0.1 * (1.0 - jaro))
}

// ============================================================================
// Phase 17 Helper Functions
// ============================================================================

/// Extract value from JSON string using path parts
fn extract_json_path(json_str: &str, path_parts: &[&str]) -> Option<String> {
    // Simple JSON path extraction - handles basic object access
    let mut current = json_str.trim();

    for part in path_parts {
        if part.is_empty() {
            continue;
        }

        // Find the key in the JSON
        let key_pattern = format!("\"{}\"", part);
        if let Some(pos) = current.find(&key_pattern) {
            let after_key = &current[pos + key_pattern.len()..];
            // Skip whitespace and colon
            let value_start = after_key.find(':')? + 1;
            let value_str = after_key[value_start..].trim_start();

            // Determine value type and extract
            if value_str.starts_with('"') {
                // String value
                if let Some(end) = value_str[1..].find('"') {
                    current = &value_str[1..end + 1];
                } else {
                    return None;
                }
            } else if value_str.starts_with('{') {
                // Object - find matching brace
                let mut depth = 0;
                let mut end = 0;
                for (i, c) in value_str.char_indices() {
                    match c {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                current = &value_str[..end];
            } else if value_str.starts_with('[') {
                // Array
                let mut depth = 0;
                let mut end = 0;
                for (i, c) in value_str.char_indices() {
                    match c {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                end = i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                current = &value_str[..end];
            } else {
                // Number, boolean, null
                let end = value_str.find(|c: char| c == ',' || c == '}' || c == ']')
                    .unwrap_or(value_str.len());
                current = value_str[..end].trim();
            }
        } else {
            return None;
        }
    }

    Some(current.to_string())
}

/// Convert array value at index to JSON string
fn array_value_to_json(arr: &dyn arrow_array::Array, i: usize) -> String {
    if arr.is_null(i) {
        return "null".to_string();
    }

    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        format!("\"{}\"", str_arr.value(i).replace('"', "\\\""))
    } else if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        i64_arr.value(i).to_string()
    } else if let Some(f64_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        f64_arr.value(i).to_string()
    } else if let Some(bool_arr) = arr.as_any().downcast_ref::<arrow_array::BooleanArray>() {
        bool_arr.value(i).to_string()
    } else if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        i32_arr.value(i).to_string()
    } else if let Some(f32_arr) = arr.as_any().downcast_ref::<arrow_array::Float32Array>() {
        f32_arr.value(i).to_string()
    } else {
        "null".to_string()
    }
}

/// Check if string is valid JSON
fn is_valid_json_string(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }

    // Simple validation - check balanced braces/brackets and basic structure
    let first = s.chars().next();
    let last = s.chars().last();

    match (first, last) {
        (Some('{'), Some('}')) | (Some('['), Some(']')) => {
            let mut depth_brace = 0i32;
            let mut depth_bracket = 0i32;
            let mut in_string = false;
            let mut prev_escape = false;

            for c in s.chars() {
                if prev_escape {
                    prev_escape = false;
                    continue;
                }
                match c {
                    '\\' if in_string => prev_escape = true,
                    '"' => in_string = !in_string,
                    '{' if !in_string => depth_brace += 1,
                    '}' if !in_string => depth_brace -= 1,
                    '[' if !in_string => depth_bracket += 1,
                    ']' if !in_string => depth_bracket -= 1,
                    _ => {}
                }
                if depth_brace < 0 || depth_bracket < 0 {
                    return false;
                }
            }
            depth_brace == 0 && depth_bracket == 0
        }
        (Some('"'), Some('"')) => true, // String
        _ => {
            // Check for number, boolean, null
            s == "true" || s == "false" || s == "null" || s.parse::<f64>().is_ok()
        }
    }
}

/// Hash a value at a given index
fn hash_value_at_index(arr: &Arc<dyn arrow_array::Array>, i: usize) -> u64 {
    if arr.is_null(i) {
        return 0;
    }

    if let Some(str_arr) = arr.as_any().downcast_ref::<arrow_array::StringArray>() {
        fnv1a_hash(str_arr.value(i).as_bytes())
    } else if let Some(i64_arr) = arr.as_any().downcast_ref::<arrow_array::Int64Array>() {
        fnv1a_hash(&i64_arr.value(i).to_le_bytes())
    } else if let Some(i32_arr) = arr.as_any().downcast_ref::<arrow_array::Int32Array>() {
        fnv1a_hash(&i32_arr.value(i).to_le_bytes())
    } else if let Some(f64_arr) = arr.as_any().downcast_ref::<arrow_array::Float64Array>() {
        fnv1a_hash(&f64_arr.value(i).to_bits().to_le_bytes())
    } else if let Some(bin_arr) = arr.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        fnv1a_hash(bin_arr.value(i))
    } else {
        fnv1a_hash(&(i as u64).to_le_bytes())
    }
}

/// Convert bytes to hex string
fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX_CHARS[(byte >> 4) as usize] as char);
        result.push(HEX_CHARS[(byte & 0xf) as usize] as char);
    }
    result
}

/// Convert hex string to bytes
fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    let hex = hex.trim();
    if hex.len() % 2 != 0 {
        return None;
    }

    let mut result = Vec::with_capacity(hex.len() / 2);
    let mut chars = hex.chars();

    while let (Some(h), Some(l)) = (chars.next(), chars.next()) {
        let high = h.to_digit(16)?;
        let low = l.to_digit(16)?;
        result.push((high * 16 + low) as u8);
    }

    Some(result)
}

/// Haversine distance in kilometers
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a = (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_KM * c
}

/// Compute bearing in degrees (0-360, 0=North)
fn compute_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let y = dlon.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlon.cos();

    let bearing = y.atan2(x).to_degrees();
    (bearing + 360.0) % 360.0
}

/// Compute Adler-32 checksum
fn compute_adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;

    for byte in data {
        a = (a + *byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }

    (b << 16) | a
}

/// Compute XXHash64 (simplified version)
fn compute_xxhash64(data: &[u8], seed: u64) -> u64 {
    const PRIME64_1: u64 = 0x9E3779B185EBCA87;
    const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
    const PRIME64_3: u64 = 0x165667B19E3779F9;
    const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
    const PRIME64_5: u64 = 0x27D4EB2F165667C5;

    let len = data.len();
    let mut h64: u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);

        let mut i = 0;
        while i + 32 <= len {
            let k1 = u64::from_le_bytes(data[i..i+8].try_into().unwrap_or([0; 8]));
            v1 = v1.wrapping_add(k1.wrapping_mul(PRIME64_2)).rotate_left(31).wrapping_mul(PRIME64_1);

            let k2 = u64::from_le_bytes(data[i+8..i+16].try_into().unwrap_or([0; 8]));
            v2 = v2.wrapping_add(k2.wrapping_mul(PRIME64_2)).rotate_left(31).wrapping_mul(PRIME64_1);

            let k3 = u64::from_le_bytes(data[i+16..i+24].try_into().unwrap_or([0; 8]));
            v3 = v3.wrapping_add(k3.wrapping_mul(PRIME64_2)).rotate_left(31).wrapping_mul(PRIME64_1);

            let k4 = u64::from_le_bytes(data[i+24..i+32].try_into().unwrap_or([0; 8]));
            v4 = v4.wrapping_add(k4.wrapping_mul(PRIME64_2)).rotate_left(31).wrapping_mul(PRIME64_1);

            i += 32;
        }

        h64 = v1.rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));

        h64 = (h64 ^ v1.wrapping_mul(PRIME64_2).rotate_left(31).wrapping_mul(PRIME64_1))
            .wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
        h64 = (h64 ^ v2.wrapping_mul(PRIME64_2).rotate_left(31).wrapping_mul(PRIME64_1))
            .wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
        h64 = (h64 ^ v3.wrapping_mul(PRIME64_2).rotate_left(31).wrapping_mul(PRIME64_1))
            .wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
        h64 = (h64 ^ v4.wrapping_mul(PRIME64_2).rotate_left(31).wrapping_mul(PRIME64_1))
            .wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
    } else {
        h64 = seed.wrapping_add(PRIME64_5);
    }

    h64 = h64.wrapping_add(len as u64);

    // Process remaining bytes
    let remaining = &data[len - (len % 32)..];
    let mut i = 0;
    while i + 8 <= remaining.len() {
        let k1 = u64::from_le_bytes(remaining[i..i+8].try_into().unwrap_or([0; 8]));
        h64 = (h64 ^ k1.wrapping_mul(PRIME64_2).rotate_left(31).wrapping_mul(PRIME64_1))
            .rotate_left(27).wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
        i += 8;
    }

    while i + 4 <= remaining.len() {
        let k1 = u32::from_le_bytes(remaining[i..i+4].try_into().unwrap_or([0; 4])) as u64;
        h64 = (h64 ^ k1.wrapping_mul(PRIME64_1)).rotate_left(23).wrapping_mul(PRIME64_2).wrapping_add(PRIME64_3);
        i += 4;
    }

    while i < remaining.len() {
        h64 = (h64 ^ remaining[i] as u64 * PRIME64_5).rotate_left(11).wrapping_mul(PRIME64_1);
        i += 1;
    }

    // Final mix
    h64 ^= h64 >> 33;
    h64 = h64.wrapping_mul(PRIME64_2);
    h64 ^= h64 >> 29;
    h64 = h64.wrapping_mul(PRIME64_3);
    h64 ^= h64 >> 32;

    h64
}

/// Take rows from a record batch by indices
fn take_batch(batch: &ArrowRecordBatch, indices: &arrow_array::UInt64Array) -> Result<record_batch::RecordBatch, compute::ArrowError> {
    let mut columns: Vec<Arc<dyn arrow_array::Array>> = Vec::new();

    for col in batch.columns() {
        let taken = arrow_select::take::take(col.as_ref(), indices, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        columns.push(taken);
    }

    let result = ArrowRecordBatch::try_new(batch.schema(), columns)
        .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;

    Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
}

// ============================================================================
// Phase 19 Helper Functions
// ============================================================================

/// Format timestamp from seconds and nanoseconds using strftime-like format
fn format_timestamp(secs: i64, nsecs: u32, format: &str) -> String {
    // Convert to date/time components
    let total_days = if secs >= 0 { secs / 86400 } else { (secs - 86399) / 86400 };
    let day_secs = secs - total_days * 86400;

    let hour = (day_secs / 3600) as i32;
    let minute = ((day_secs % 3600) / 60) as i32;
    let second = (day_secs % 60) as i32;

    let (year, month, day) = days_to_ymd(total_days as i32);

    // Replace format specifiers
    format.replace("%Y", &format!("{:04}", year))
        .replace("%m", &format!("{:02}", month))
        .replace("%d", &format!("{:02}", day))
        .replace("%H", &format!("{:02}", hour))
        .replace("%M", &format!("{:02}", minute))
        .replace("%S", &format!("{:02}", second))
        .replace("%f", &format!("{:06}", nsecs / 1000))  // microseconds
        .replace("%j", &format!("{:03}", day_of_year_from_days(total_days as i32)))
}

/// Format date from days since epoch
fn format_date_from_days(days: i64, format: &str) -> String {
    let (year, month, day) = days_to_ymd(days as i32);
    let dow = day_of_week_from_days(days as i32);
    let doy = day_of_year_from_days(days as i32);

    format.replace("%Y", &format!("{:04}", year))
        .replace("%m", &format!("{:02}", month))
        .replace("%d", &format!("{:02}", day))
        .replace("%j", &format!("{:03}", doy))
        .replace("%w", &format!("{}", dow))
}

/// Parse timestamp string to microseconds since epoch
fn parse_timestamp_string(s: &str, format: &str) -> Option<i64> {
    // Simple parser for common formats
    let parts: Vec<&str> = s.split(|c: char| !c.is_numeric()).filter(|p| !p.is_empty()).collect();

    if format.contains("%Y-%m-%d") || format.contains("%Y/%m/%d") {
        if parts.len() >= 3 {
            let year: i32 = parts[0].parse().ok()?;
            let month: u32 = parts[1].parse().ok()?;
            let day: u32 = parts[2].parse().ok()?;

            let days = ymd_to_days(year, month, day);

            let hour: i64 = parts.get(3).and_then(|p| p.parse().ok()).unwrap_or(0);
            let minute: i64 = parts.get(4).and_then(|p| p.parse().ok()).unwrap_or(0);
            let second: i64 = parts.get(5).and_then(|p| p.parse().ok()).unwrap_or(0);

            let total_secs = (days as i64) * 86400 + hour * 3600 + minute * 60 + second;
            return Some(total_secs * 1_000_000);
        }
    }

    // Try ISO 8601 format as default
    if parts.len() >= 6 {
        let year: i32 = parts[0].parse().ok()?;
        let month: u32 = parts[1].parse().ok()?;
        let day: u32 = parts[2].parse().ok()?;
        let hour: i64 = parts[3].parse().ok()?;
        let minute: i64 = parts[4].parse().ok()?;
        let second: i64 = parts[5].parse().ok()?;

        let days = ymd_to_days(year, month, day);
        let total_secs = (days as i64) * 86400 + hour * 3600 + minute * 60 + second;
        return Some(total_secs * 1_000_000);
    }

    None
}

/// Parse date string to days since epoch
fn parse_date_string(s: &str, _format: &str) -> Option<i32> {
    let parts: Vec<&str> = s.split(|c: char| !c.is_numeric()).filter(|p| !p.is_empty()).collect();

    if parts.len() >= 3 {
        let year: i32 = parts[0].parse().ok()?;
        let month: u32 = parts[1].parse().ok()?;
        let day: u32 = parts[2].parse().ok()?;
        return Some(ymd_to_days(year, month, day));
    }

    None
}

/// Extract timestamp component (year, month, day, hour, minute, second)
fn extract_timestamp_component(microseconds: i64, component: &str) -> i32 {
    let secs = microseconds / 1_000_000;
    let total_days = if secs >= 0 { secs / 86400 } else { (secs - 86399) / 86400 };
    let day_secs = secs - total_days * 86400;

    let (year, month, day) = days_to_ymd(total_days as i32);
    let day = day as i32;

    match component {
        "year" => year,
        "month" => month,
        "day" => day,
        "hour" => (day_secs / 3600) as i32,
        "minute" => ((day_secs % 3600) / 60) as i32,
        "second" => (day_secs % 60) as i32,
        _ => 0,
    }
}

/// Format number with thousand separator and decimal separator
fn format_number_with_separators(value: f64, decimal_places: u32, group_sep: &str, decimal_sep: &str) -> String {
    let rounded = format!("{:.prec$}", value.abs(), prec = decimal_places as usize);
    let parts: Vec<&str> = rounded.split('.').collect();

    let integer_part = parts[0];
    let decimal_part = parts.get(1).unwrap_or(&"");

    // Add thousand separators
    let chars: Vec<char> = integer_part.chars().collect();
    let mut with_seps = String::new();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            with_seps.push_str(group_sep);
        }
        with_seps.push(*c);
    }

    let sign = if value < 0.0 { "-" } else { "" };
    if decimal_places > 0 {
        format!("{}{}{}{}", sign, with_seps, decimal_sep, decimal_part)
    } else {
        format!("{}{}", sign, with_seps)
    }
}

/// Extract days from date array (Date32 or Date64)
fn extract_days_from_date_array(inner: &Arc<dyn arrow_array::Array>) -> Result<Vec<Option<i32>>, compute::ArrowError> {
    if let Some(date_arr) = inner.as_any().downcast_ref::<arrow_array::Date32Array>() {
        return Ok(date_arr.iter().collect());
    }

    if let Some(date_arr) = inner.as_any().downcast_ref::<arrow_array::Date64Array>() {
        return Ok(date_arr.iter().map(|opt| opt.map(|ms| (ms / 86_400_000) as i32)).collect());
    }

    // Also support timestamps
    if let Some(ts_arr) = inner.as_any().downcast_ref::<arrow_array::TimestampMicrosecondArray>() {
        return Ok(ts_arr.iter().map(|opt| opt.map(|us| (us / 86_400_000_000) as i32)).collect());
    }

    Err(compute::ArrowError::InvalidArgument("Expected date or timestamp array".to_string()))
}

/// Get day of week from days since epoch (0=Sunday, 6=Saturday)
fn day_of_week_from_days(days: i32) -> i32 {
    // Jan 1, 1970 was a Thursday (4)
    let dow = (days + 4) % 7;
    if dow < 0 { dow + 7 } else { dow }
}

/// Get day of year from days since epoch (1-366)
fn day_of_year_from_days(days: i32) -> i32 {
    let (year, month, day) = days_to_ymd(days);
    let jan1 = ymd_to_days(year, 1, 1);
    days - jan1 + 1
}

/// Get ISO week of year from days since epoch
fn iso_week_of_year_from_days(days: i32) -> i32 {
    let (year, _, _) = days_to_ymd(days);

    // Find the Thursday of the week containing this date
    let dow = day_of_week_from_days(days);
    let dow_mon = if dow == 0 { 6 } else { dow - 1 };  // Convert to Monday=0
    let thursday = days + (3 - dow_mon);

    // Find what year that Thursday belongs to
    let (thu_year, _, _) = days_to_ymd(thursday);

    // Find January 4th of that year (always in week 1)
    let jan4 = ymd_to_days(thu_year, 1, 4);
    let jan4_dow = day_of_week_from_days(jan4);
    let jan4_dow_mon = if jan4_dow == 0 { 6 } else { jan4_dow - 1 };

    // Find the Monday of week 1
    let week1_monday = jan4 - jan4_dow_mon;

    // Calculate week number
    ((thursday - week1_monday) / 7) + 1
}

/// Check if year is a leap year
fn is_leap_year_calc(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Add business days to a date
fn add_business_days_impl(days: i32, business_days: i32) -> i32 {
    let mut current = days;
    let mut remaining = business_days.abs();
    let direction = if business_days >= 0 { 1 } else { -1 };

    while remaining > 0 {
        current += direction;
        let dow = day_of_week_from_days(current);
        if dow != 0 && dow != 6 {  // Not weekend
            remaining -= 1;
        }
    }

    current
}

/// Count business days between two dates
fn count_business_days(start: i32, end: i32) -> i32 {
    if start == end {
        return 0;
    }

    let (from, to) = if start < end { (start, end) } else { (end, start) };
    let sign = if start < end { 1 } else { -1 };

    let mut count = 0;
    for d in (from + 1)..=to {
        let dow = day_of_week_from_days(d);
        if dow != 0 && dow != 6 {
            count += 1;
        }
    }

    count * sign
}

/// URL percent encode string
fn percent_encode_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 3);
    for c in s.bytes() {
        match c {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(c as char);
            }
            _ => {
                result.push_str(&format!("%{:02X}", c));
            }
        }
    }
    result
}

/// URL percent decode string
fn percent_decode_string(s: &str) -> String {
    let mut result = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                std::str::from_utf8(&bytes[i+1..i+3]).unwrap_or(""),
                16
            ) {
                result.push(byte);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i]);
        i += 1;
    }

    String::from_utf8_lossy(&result).to_string()
}

/// HTML entity encode
fn html_escape_string(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// HTML entity decode
fn html_unescape_string(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&apos;", "'")
}

/// Escape regex special characters
fn escape_regex_chars(s: &str) -> String {
    let special = ['\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '|', '^', '$'];
    let mut result = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if special.contains(&c) {
            result.push('\\');
        }
        result.push(c);
    }
    result
}

/// Aggregate values from a list element array
fn aggregate_list_values(values: &Arc<dyn arrow_array::Array>, op: &str) -> Option<f64> {
    let mut nums: Vec<f64> = Vec::new();

    if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Float64Array>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                nums.push(arr.value(i));
            }
        }
    } else if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                nums.push(arr.value(i) as f64);
            }
        }
    } else if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Int32Array>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                nums.push(arr.value(i) as f64);
            }
        }
    } else if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Float32Array>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                nums.push(arr.value(i) as f64);
            }
        }
    } else {
        return None;
    }

    if nums.is_empty() {
        return None;
    }

    match op {
        "sum" => Some(nums.iter().sum()),
        "mean" => Some(nums.iter().sum::<f64>() / nums.len() as f64),
        "min" => nums.iter().copied().reduce(f64::min),
        "max" => nums.iter().copied().reduce(f64::max),
        _ => None,
    }
}

/// Check if list contains a value (as string comparison)
fn list_contains_value(values: &Arc<dyn arrow_array::Array>, target: &str) -> bool {
    if let Some(arr) = values.as_any().downcast_ref::<arrow_array::StringArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) && arr.value(i) == target {
                return true;
            }
        }
    } else if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>() {
        if let Ok(target_val) = target.parse::<i64>() {
            for i in 0..arr.len() {
                if !arr.is_null(i) && arr.value(i) == target_val {
                    return true;
                }
            }
        }
    } else if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Float64Array>() {
        if let Ok(target_val) = target.parse::<f64>() {
            for i in 0..arr.len() {
                if !arr.is_null(i) && (arr.value(i) - target_val).abs() < f64::EPSILON {
                    return true;
                }
            }
        }
    }
    false
}

/// Extract first element from each list in a list array
fn extract_list_element(
    list_arr: &arrow_array::ListArray,
    index: usize,
    elem_type: Option<arrow_schema::DataType>
) -> Result<arrays::Array, compute::ArrowError> {
    use arrow_schema::DataType;

    let elem_type = elem_type.unwrap_or(DataType::Float64);

    match elem_type {
        DataType::Float64 => {
            let result: arrow_array::Float64Array = (0..list_arr.len())
                .map(|i| {
                    if list_arr.is_null(i) {
                        None
                    } else {
                        let values = list_arr.value(i);
                        if index < values.len() {
                            if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Float64Array>() {
                                if !arr.is_null(index) {
                                    return Some(arr.value(index));
                                }
                            }
                        }
                        None
                    }
                })
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
        DataType::Int64 => {
            let result: arrow_array::Int64Array = (0..list_arr.len())
                .map(|i| {
                    if list_arr.is_null(i) {
                        None
                    } else {
                        let values = list_arr.value(i);
                        if index < values.len() {
                            if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>() {
                                if !arr.is_null(index) {
                                    return Some(arr.value(index));
                                }
                            }
                        }
                        None
                    }
                })
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
        DataType::Utf8 => {
            let result: arrow_array::StringArray = (0..list_arr.len())
                .map(|i| {
                    if list_arr.is_null(i) {
                        None
                    } else {
                        let values = list_arr.value(i);
                        if index < values.len() {
                            if let Some(arr) = values.as_any().downcast_ref::<arrow_array::StringArray>() {
                                if !arr.is_null(index) {
                                    return Some(arr.value(index).to_string());
                                }
                            }
                        }
                        None
                    }
                })
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
        _ => {
            // Default to Float64 for unsupported types
            let result: arrow_array::Float64Array = std::iter::repeat(None)
                .take(list_arr.len())
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
    }
}

/// Extract last element from each list in a list array
fn extract_list_element_last(
    list_arr: &arrow_array::ListArray,
    elem_type: Option<arrow_schema::DataType>
) -> Result<arrays::Array, compute::ArrowError> {
    use arrow_schema::DataType;

    let elem_type = elem_type.unwrap_or(DataType::Float64);

    match elem_type {
        DataType::Float64 => {
            let result: arrow_array::Float64Array = (0..list_arr.len())
                .map(|i| {
                    if list_arr.is_null(i) {
                        None
                    } else {
                        let values = list_arr.value(i);
                        if values.len() > 0 {
                            let index = values.len() - 1;
                            if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Float64Array>() {
                                if !arr.is_null(index) {
                                    return Some(arr.value(index));
                                }
                            }
                        }
                        None
                    }
                })
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
        DataType::Int64 => {
            let result: arrow_array::Int64Array = (0..list_arr.len())
                .map(|i| {
                    if list_arr.is_null(i) {
                        None
                    } else {
                        let values = list_arr.value(i);
                        if values.len() > 0 {
                            let index = values.len() - 1;
                            if let Some(arr) = values.as_any().downcast_ref::<arrow_array::Int64Array>() {
                                if !arr.is_null(index) {
                                    return Some(arr.value(index));
                                }
                            }
                        }
                        None
                    }
                })
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
        DataType::Utf8 => {
            let result: arrow_array::StringArray = (0..list_arr.len())
                .map(|i| {
                    if list_arr.is_null(i) {
                        None
                    } else {
                        let values = list_arr.value(i);
                        if values.len() > 0 {
                            let index = values.len() - 1;
                            if let Some(arr) = values.as_any().downcast_ref::<arrow_array::StringArray>() {
                                if !arr.is_null(index) {
                                    return Some(arr.value(index).to_string());
                                }
                            }
                        }
                        None
                    }
                })
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
        _ => {
            let result: arrow_array::Float64Array = std::iter::repeat(None)
                .take(list_arr.len())
                .collect();
            Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
        }
    }
}

/// Convert glob pattern to regex
fn glob_to_regex(pattern: &str) -> String {
    let mut regex = String::from("^");

    for c in pattern.chars() {
        match c {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }

    regex.push('$');
    regex
}

/// Strip accents/diacritics from string
fn strip_accents(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => 'A',
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
            'È' | 'É' | 'Ê' | 'Ë' => 'E',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'Ì' | 'Í' | 'Î' | 'Ï' => 'I',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => 'O',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
            'Ù' | 'Ú' | 'Û' | 'Ü' => 'U',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            'Ý' | 'Ÿ' => 'Y',
            'ý' | 'ÿ' => 'y',
            'Ñ' => 'N',
            'ñ' => 'n',
            'Ç' => 'C',
            'ç' => 'c',
            _ => c,
        })
        .collect()
}

// ============================================================================
// Phase 21 Helper Functions (RNG)
// ============================================================================

/// Simple LCG random number generator state
fn create_rng(seed: Option<u64>) -> u64 {
    seed.unwrap_or(12345678901234567890)
}

/// LCG step function
fn lcg_next(state: &mut u64) -> u64 {
    // LCG parameters from Numerical Recipes
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;
    *state = state.wrapping_mul(A).wrapping_add(C);
    *state
}

/// Generate uniform f64 in [0, 1)
fn lcg_f64(state: &mut u64) -> f64 {
    let val = lcg_next(state);
    (val as f64) / (u64::MAX as f64)
}

// ============================================================================
// Compute implementation (stub - to be expanded)
// ============================================================================

// ============================================================================
// Compute implementation - Arrow-only operations
// ============================================================================

impl compute::Guest for Component {
    // ========== Arithmetic Operations (arrow-arith) ==========

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

    // ========== Aggregation Operations ==========

    fn sum_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            Ok(arrow_arith::aggregate::sum(int_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected Int64Array".to_string()))
        }
    }

    fn sum_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            Ok(arrow_arith::aggregate::sum(float_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected Float64Array".to_string()))
        }
    }

    fn min_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            Ok(arrow_arith::aggregate::min(int_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected Int64Array".to_string()))
        }
    }

    fn min_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            Ok(arrow_arith::aggregate::min(float_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected Float64Array".to_string()))
        }
    }

    fn max_i64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<i64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            Ok(arrow_arith::aggregate::max(int_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected Int64Array".to_string()))
        }
    }

    fn max_f64(arr: arrays::ArrayBorrow<'_>) -> Result<Option<f64>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(float_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Float64Array>() {
            Ok(arrow_arith::aggregate::max(float_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected Float64Array".to_string()))
        }
    }

    fn min_string(arr: arrays::ArrayBorrow<'_>) -> Result<Option<String>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            Ok(arrow_arith::aggregate::min_string(str_arr).map(|s| s.to_string()))
        } else {
            Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
        }
    }

    fn max_string(arr: arrays::ArrayBorrow<'_>) -> Result<Option<String>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            Ok(arrow_arith::aggregate::max_string(str_arr).map(|s| s.to_string()))
        } else {
            Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
        }
    }

    fn bool_and(arr: arrays::ArrayBorrow<'_>) -> Result<Option<bool>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(bool_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>() {
            Ok(arrow_arith::aggregate::bool_and(bool_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))
        }
    }

    fn bool_or(arr: arrays::ArrayBorrow<'_>) -> Result<Option<bool>, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(bool_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>() {
            Ok(arrow_arith::aggregate::bool_or(bool_arr))
        } else {
            Err(compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))
        }
    }

    // ========== Comparison Operations ==========

    fn compare(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>, op: compute::CompareOp) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let result = match op {
            compute::CompareOp::Equal => arrow_ord::cmp::eq(&left_impl.inner, &right_impl.inner),
            compute::CompareOp::NotEqual => arrow_ord::cmp::neq(&left_impl.inner, &right_impl.inner),
            compute::CompareOp::LessThan => arrow_ord::cmp::lt(&left_impl.inner, &right_impl.inner),
            compute::CompareOp::LessThanOrEqual => arrow_ord::cmp::lt_eq(&left_impl.inner, &right_impl.inner),
            compute::CompareOp::GreaterThan => arrow_ord::cmp::gt(&left_impl.inner, &right_impl.inner),
            compute::CompareOp::GreaterThanOrEqual => arrow_ord::cmp::gt_eq(&left_impl.inner, &right_impl.inner),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn compare_scalar_i64(arr: arrays::ArrayBorrow<'_>, value: i64, op: compute::CompareOp) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar = arrow_array::Int64Array::new_scalar(value);
        let result = match op {
            compute::CompareOp::Equal => arrow_ord::cmp::eq(&arr_impl.inner, &scalar),
            compute::CompareOp::NotEqual => arrow_ord::cmp::neq(&arr_impl.inner, &scalar),
            compute::CompareOp::LessThan => arrow_ord::cmp::lt(&arr_impl.inner, &scalar),
            compute::CompareOp::LessThanOrEqual => arrow_ord::cmp::lt_eq(&arr_impl.inner, &scalar),
            compute::CompareOp::GreaterThan => arrow_ord::cmp::gt(&arr_impl.inner, &scalar),
            compute::CompareOp::GreaterThanOrEqual => arrow_ord::cmp::gt_eq(&arr_impl.inner, &scalar),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn compare_scalar_f64(arr: arrays::ArrayBorrow<'_>, value: f64, op: compute::CompareOp) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar = arrow_array::Float64Array::new_scalar(value);
        let result = match op {
            compute::CompareOp::Equal => arrow_ord::cmp::eq(&arr_impl.inner, &scalar),
            compute::CompareOp::NotEqual => arrow_ord::cmp::neq(&arr_impl.inner, &scalar),
            compute::CompareOp::LessThan => arrow_ord::cmp::lt(&arr_impl.inner, &scalar),
            compute::CompareOp::LessThanOrEqual => arrow_ord::cmp::lt_eq(&arr_impl.inner, &scalar),
            compute::CompareOp::GreaterThan => arrow_ord::cmp::gt(&arr_impl.inner, &scalar),
            compute::CompareOp::GreaterThanOrEqual => arrow_ord::cmp::gt_eq(&arr_impl.inner, &scalar),
        }.map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn compare_scalar_string(arr: arrays::ArrayBorrow<'_>, value: String, op: compute::CompareOp) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let scalar = arrow_array::StringArray::new_scalar(&value);
        let result = match op {
            compute::CompareOp::Equal => arrow_ord::cmp::eq(&arr_impl.inner, &scalar),
            compute::CompareOp::NotEqual => arrow_ord::cmp::neq(&arr_impl.inner, &scalar),
            compute::CompareOp::LessThan => arrow_ord::cmp::lt(&arr_impl.inner, &scalar),
            compute::CompareOp::LessThanOrEqual => arrow_ord::cmp::lt_eq(&arr_impl.inner, &scalar),
            compute::CompareOp::GreaterThan => arrow_ord::cmp::gt(&arr_impl.inner, &scalar),
            compute::CompareOp::GreaterThanOrEqual => arrow_ord::cmp::gt_eq(&arr_impl.inner, &scalar),
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

    // ========== Boolean Operations ==========

    fn and(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let right_bool = right_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let result = arrow_arith::boolean::and(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn or(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let right_bool = right_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let result = arrow_arith::boolean::or(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn not(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let bool_arr = arr_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let result = arrow_arith::boolean::not(bool_arr)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn and_not(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let right_bool = right_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let result = arrow_arith::boolean::and_not(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn and_kleene(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let right_bool = right_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let result = arrow_arith::boolean::and_kleene(left_bool, right_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn or_kleene(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let left_bool = left_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
        let right_bool = right_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Expected BooleanArray".to_string()))?;
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

    // ========== Bitwise Operations ==========

    fn bitwise_and(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>()
        ) {
            let result = arrow_arith::bitwise::bitwise_and(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_and".to_string()))
    }

    fn bitwise_or(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>()
        ) {
            let result = arrow_arith::bitwise::bitwise_or(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_or".to_string()))
    }

    fn bitwise_xor(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>()
        ) {
            let result = arrow_arith::bitwise::bitwise_xor(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_xor".to_string()))
    }

    fn bitwise_not(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            let result = arrow_arith::bitwise::bitwise_not(int_arr)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_not".to_string()))
    }

    fn bitwise_and_not(left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        
        if let (Some(l), Some(r)) = (
            left_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            right_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>()
        ) {
            let result = arrow_arith::bitwise::bitwise_and_not(l, r)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_and_not".to_string()))
    }

    fn bitwise_shift_left(arr: arrays::ArrayBorrow<'_>, shift: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let shift_impl = shift.get::<ArrayImpl>();
        
        if let (Some(a), Some(s)) = (
            arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            shift_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>()
        ) {
            let result = arrow_arith::bitwise::bitwise_shift_left(a, s)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_shift_left".to_string()))
    }

    fn bitwise_shift_right(arr: arrays::ArrayBorrow<'_>, shift: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let shift_impl = shift.get::<ArrayImpl>();
        
        if let (Some(a), Some(s)) = (
            arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>(),
            shift_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>()
        ) {
            let result = arrow_arith::bitwise::bitwise_shift_right(a, s)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Unsupported type for bitwise_shift_right".to_string()))
    }

    // ========== Sorting Operations ==========

    fn sort_to_indices(arr: arrays::ArrayBorrow<'_>, options: Option<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_opts = options.map(|o| arrow_ord::sort::SortOptions {
            descending: o.descending,
            nulls_first: o.nulls_first,
        });
        let result = arrow_ord::sort::sort_to_indices(&arr_impl.inner, sort_opts, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn sort(arr: arrays::ArrayBorrow<'_>, options: Option<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_opts = options.map(|o| arrow_ord::sort::SortOptions {
            descending: o.descending,
            nulls_first: o.nulls_first,
        });
        let result = arrow_ord::sort::sort(&arr_impl.inner, sort_opts)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn sort_limit(arr: arrays::ArrayBorrow<'_>, options: Option<compute::SortOptions>, limit: u64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_opts = options.map(|o| arrow_ord::sort::SortOptions {
            descending: o.descending,
            nulls_first: o.nulls_first,
        });
        let result = arrow_ord::sort::sort_limit(&arr_impl.inner, sort_opts, Some(limit as usize))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn lexsort(arrays_list: Vec<arrays::ArrayBorrow<'_>>, options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let columns: Vec<arrow_ord::sort::SortColumn> = arrays_list.iter().zip(options.iter())
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
        let result = arrow_ord::sort::lexsort_to_indices(&columns, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }))
    }

    fn rank(arr: arrays::ArrayBorrow<'_>, options: Option<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let sort_opts = options.map(|o| arrow_ord::sort::SortOptions {
            descending: o.descending,
            nulls_first: o.nulls_first,
        });
        let result = arrow_ord::rank::rank(&arr_impl.inner, sort_opts)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        // Convert Vec<u32> to UInt32Array
        let result_arr: arrow_array::UInt32Array = result.into();
        Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result_arr) }))
    }

    // ========== Selection Operations ==========

    fn filter(arr: arrays::ArrayBorrow<'_>, predicate: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let pred_impl = predicate.get::<ArrayImpl>();
        let pred_bool = pred_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Predicate must be BooleanArray".to_string()))?;
        let result = arrow_select::filter::filter(&arr_impl.inner, pred_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn filter_record_batch(batch: record_batch::RecordBatchBorrow<'_>, predicate: arrays::ArrayBorrow<'_>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let pred_impl = predicate.get::<ArrayImpl>();
        let pred_bool = pred_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Predicate must be BooleanArray".to_string()))?;
        let result = arrow_select::filter::filter_record_batch(&batch_impl.inner, pred_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn take(arr: arrays::ArrayBorrow<'_>, indices: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let idx_impl = indices.get::<ArrayImpl>();
        let result = arrow_select::take::take(&arr_impl.inner, &idx_impl.inner, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn take_record_batch(batch: record_batch::RecordBatchBorrow<'_>, indices: arrays::ArrayBorrow<'_>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        let batch_impl = batch.get::<RecordBatchImpl>();
        let idx_impl = indices.get::<ArrayImpl>();
        let result = arrow_select::take::take_record_batch(&batch_impl.inner, &idx_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn concat(arrays_list: Vec<arrays::ArrayBorrow<'_>>) -> Result<arrays::Array, compute::ArrowError> {
        let refs: Vec<&dyn arrow_array::Array> = arrays_list.iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();
        let result = arrow_select::concat::concat(&refs)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn concat_batches(batches: Vec<record_batch::RecordBatchBorrow<'_>>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        if batches.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("Cannot concat empty list of batches".to_string()));
        }
        let first = batches[0].get::<RecordBatchImpl>();
        let schema = first.inner.schema();
        let refs: Vec<&ArrowRecordBatch> = batches.iter()
            .map(|b| &b.get::<RecordBatchImpl>().inner)
            .collect();
        let result = arrow_select::concat::concat_batches(&schema, refs.into_iter())
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(record_batch::RecordBatch::new(RecordBatchImpl { inner: result }))
    }

    fn interleave(arrays_list: Vec<arrays::ArrayBorrow<'_>>, indices: Vec<(u32, u32)>) -> Result<arrays::Array, compute::ArrowError> {
        let refs: Vec<&dyn arrow_array::Array> = arrays_list.iter()
            .map(|a| a.get::<ArrayImpl>().inner.as_ref())
            .collect();
        let interleave_indices: Vec<(usize, usize)> = indices.iter()
            .map(|(a, b)| (*a as usize, *b as usize))
            .collect();
        let result = arrow_select::interleave::interleave(&refs, &interleave_indices)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn nullif(arr: arrays::ArrayBorrow<'_>, predicate: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let pred_impl = predicate.get::<ArrayImpl>();
        let pred_bool = pred_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Predicate must be BooleanArray".to_string()))?;
        let result = arrow_select::nullif::nullif(&arr_impl.inner, pred_bool)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn zip(predicate: arrays::ArrayBorrow<'_>, left: arrays::ArrayBorrow<'_>, right: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let pred_impl = predicate.get::<ArrayImpl>();
        let left_impl = left.get::<ArrayImpl>();
        let right_impl = right.get::<ArrayImpl>();
        let pred_bool = pred_impl.inner.as_any().downcast_ref::<arrow_array::BooleanArray>()
            .ok_or_else(|| compute::ArrowError::ComputeError("Predicate must be BooleanArray".to_string()))?;
        let result = arrow_select::zip::zip(pred_bool, &left_impl.inner, &right_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn shift(arr: arrays::ArrayBorrow<'_>, offset: i64) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_select::window::shift(&arr_impl.inner, offset)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    // ========== String Operations ==========

    fn string_length(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_string::length::length(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn bit_length(arr: arrays::ArrayBorrow<'_>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_string::length::bit_length(&arr_impl.inner)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn string_like(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&pattern);
            let result = arrow_string::like::like(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn string_ilike(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&pattern);
            let result = arrow_string::like::ilike(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn string_nlike(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&pattern);
            let result = arrow_string::like::nlike(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn string_nilike(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&pattern);
            let result = arrow_string::like::nilike(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn string_starts_with(arr: arrays::ArrayBorrow<'_>, prefix: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&prefix);
            let result = arrow_string::like::starts_with(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn string_ends_with(arr: arrays::ArrayBorrow<'_>, suffix: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&suffix);
            let result = arrow_string::like::ends_with(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn string_contains(arr: arrays::ArrayBorrow<'_>, substring: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let scalar = arrow_array::StringArray::new_scalar(&substring);
            let result = arrow_string::like::contains(str_arr, &scalar)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn regexp_is_match(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let result = arrow_string::regexp::regexp_is_match_scalar(str_arr, &pattern, None)
                .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    fn regexp_match(arr: arrays::ArrayBorrow<'_>, pattern: String) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        // Create scalar pattern
        let pattern_scalar = arrow_array::StringArray::new_scalar(pattern);
        let result = arrow_string::regexp::regexp_match(&*arr_impl.inner, &pattern_scalar, None)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn substring(arr: arrays::ArrayBorrow<'_>, start: i64, length: Option<u64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let result = arrow_string::substring::substring(&arr_impl.inner, start, length.map(|l| l as u64))
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn concat_elements(arrays_list: Vec<arrays::ArrayBorrow<'_>>) -> Result<arrays::Array, compute::ArrowError> {
        if arrays_list.is_empty() {
            return Err(compute::ArrowError::InvalidArgument("Cannot concat empty list".to_string()));
        }
        
        // Get first array to check type
        let first = arrays_list[0].get::<ArrayImpl>();
        if let Some(str_arr) = first.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            let mut result = str_arr.clone();
            for arr in arrays_list.iter().skip(1) {
                let arr_impl = arr.get::<ArrayImpl>();
                let next = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| compute::ArrowError::ComputeError("All arrays must be StringArray".to_string()))?;
                result = arrow_string::concat_elements::concat_elements_utf8(&result, next)
                    .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
            }
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    // ========== Temporal Operations ==========

    fn extract_date_part(arr: arrays::ArrayBorrow<'_>, part: compute::DatePart) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let arrow_part = match part {
            compute::DatePart::Year => arrow_arith::temporal::DatePart::Year,
            compute::DatePart::Quarter => arrow_arith::temporal::DatePart::Quarter,
            compute::DatePart::Month => arrow_arith::temporal::DatePart::Month,
            compute::DatePart::Week => arrow_arith::temporal::DatePart::Week,
            compute::DatePart::Day => arrow_arith::temporal::DatePart::Day,
            compute::DatePart::DayOfWeekSunday0 => arrow_arith::temporal::DatePart::DayOfWeekSunday0,
            compute::DatePart::DayOfWeekMonday0 => arrow_arith::temporal::DatePart::DayOfWeekMonday0,
            compute::DatePart::DayOfYear => arrow_arith::temporal::DatePart::DayOfYear,
            compute::DatePart::Hour => arrow_arith::temporal::DatePart::Hour,
            compute::DatePart::Minute => arrow_arith::temporal::DatePart::Minute,
            compute::DatePart::Second => arrow_arith::temporal::DatePart::Second,
            compute::DatePart::Millisecond => arrow_arith::temporal::DatePart::Millisecond,
            compute::DatePart::Microsecond => arrow_arith::temporal::DatePart::Microsecond,
            compute::DatePart::Nanosecond => arrow_arith::temporal::DatePart::Nanosecond,
        };
        let result = arrow_arith::temporal::date_part(&arr_impl.inner, arrow_part)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    // ========== Cast Operations ==========

    fn cast(arr: arrays::ArrayBorrow<'_>, to_type: types::DataType) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        let arrow_type = convert::to_arrow_data_type(&to_type);
        let result = arrow_cast::cast(&arr_impl.inner, &arrow_type)
            .map_err(|e| compute::ArrowError::ComputeError(e.to_string()))?;
        Ok(arrays::Array::new(ArrayImpl { inner: result }))
    }

    fn can_cast(from_type: types::DataType, to_type: types::DataType) -> bool {
        let from = convert::to_arrow_data_type(&from_type);
        let to = convert::to_arrow_data_type(&to_type);
        arrow_cast::can_cast_types(&from, &to)
    }

    // ========== In-List Operations ==========

    fn in_list_i64(arr: arrays::ArrayBorrow<'_>, values: Vec<i64>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(int_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::Int64Array>() {
            // Create a HashSet for fast lookups
            let value_set: std::collections::HashSet<i64> = values.into_iter().collect();
            // Check each element
            let result: arrow_array::BooleanArray = int_arr.iter()
                .map(|opt| opt.map(|v| value_set.contains(&v)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected Int64Array".to_string()))
    }

    fn in_list_string(arr: arrays::ArrayBorrow<'_>, values: Vec<String>) -> Result<arrays::Array, compute::ArrowError> {
        let arr_impl = arr.get::<ArrayImpl>();
        if let Some(str_arr) = arr_impl.inner.as_any().downcast_ref::<arrow_array::StringArray>() {
            // Create a HashSet for fast lookups
            let value_set: std::collections::HashSet<String> = values.into_iter().collect();
            // Check each element
            let result: arrow_array::BooleanArray = str_arr.iter()
                .map(|opt| opt.map(|v| value_set.contains(v)))
                .collect();
            return Ok(arrays::Array::new(ArrayImpl { inner: Arc::new(result) }));
        }
        Err(compute::ArrowError::ComputeError("Expected StringArray".to_string()))
    }

    // ========== Utility Operations ==========

    fn count(arr: arrays::ArrayBorrow<'_>) -> u64 {
        let arr_impl = arr.get::<ArrayImpl>();
        (arr_impl.inner.len() - arr_impl.inner.null_count()) as u64
    }

    fn null_count(arr: arrays::ArrayBorrow<'_>) -> u64 {
        let arr_impl = arr.get::<ArrayImpl>();
        arr_impl.inner.null_count() as u64
    }

    fn len(arr: arrays::ArrayBorrow<'_>) -> u64 {
        let arr_impl = arr.get::<ArrayImpl>();
        arr_impl.inner.len() as u64
    }

    fn is_empty(arr: arrays::ArrayBorrow<'_>) -> bool {
        let arr_impl = arr.get::<ArrayImpl>();
        arr_impl.inner.is_empty()
    }
}

// IO imports
use arrow_ipc::reader::{FileReader as IpcFileReader, StreamReader as IpcStreamReader};
use arrow_ipc::writer::{FileWriter as IpcFileWriter, StreamWriter as IpcStreamWriter};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter as ParquetArrowWriter;
use parquet::basic::Compression as ParquetCompression;
use parquet::file::properties::WriterProperties;

/// Helper to convert errors to io::ArrowError
fn to_io_error(e: impl std::fmt::Display) -> io::ArrowError {
    io::ArrowError::IoError(e.to_string())
}

fn to_parquet_compression(comp: io::Compression) -> Result<ParquetCompression, io::ArrowError> {
    match comp {
        io::Compression::Uncompressed => Ok(ParquetCompression::UNCOMPRESSED),
        io::Compression::Snappy => Ok(ParquetCompression::SNAPPY),
        io::Compression::Lz4 => Ok(ParquetCompression::LZ4),
        io::Compression::Gzip => Ok(ParquetCompression::GZIP(Default::default())),
        io::Compression::Zstd => Err(io::ArrowError::NotImplemented(
            "ZSTD compression requires composition with compression-multiplexer component".to_string()
        )),
        io::Compression::Bzip2 => Err(io::ArrowError::NotImplemented(
            "BZIP2 compression is not supported by the Parquet format".to_string()
        )),
        io::Compression::Lzma => Err(io::ArrowError::NotImplemented(
            "LZMA compression is not supported by the Parquet format".to_string()
        )),
    }
}

fn stats_value_to_string(bytes: Option<&[u8]>, physical_type: parquet::basic::Type) -> Option<String> {
    use parquet::basic::Type;

    let bytes = bytes?;

    match physical_type {
        Type::BOOLEAN => {
            if !bytes.is_empty() {
                Some(if bytes[0] == 0 { "false" } else { "true" }.to_string())
            } else {
                None
            }
        }
        Type::INT32 => {
            if bytes.len() >= 4 {
                let value = i32::from_le_bytes(bytes[..4].try_into().ok()?);
                Some(value.to_string())
            } else {
                None
            }
        }
        Type::INT64 => {
            if bytes.len() >= 8 {
                let value = i64::from_le_bytes(bytes[..8].try_into().ok()?);
                Some(value.to_string())
            } else {
                None
            }
        }
        Type::INT96 => {
            Some(format!("INT96({} bytes)", bytes.len()))
        }
        Type::FLOAT => {
            if bytes.len() >= 4 {
                let value = f32::from_le_bytes(bytes[..4].try_into().ok()?);
                Some(value.to_string())
            } else {
                None
            }
        }
        Type::DOUBLE => {
            if bytes.len() >= 8 {
                let value = f64::from_le_bytes(bytes[..8].try_into().ok()?);
                Some(value.to_string())
            } else {
                None
            }
        }
        Type::BYTE_ARRAY | Type::FIXED_LEN_BYTE_ARRAY => {
            if let Ok(s) = std::str::from_utf8(bytes) {
                Some(s.to_string())
            } else {
                Some(format!("0x{}", bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()))
            }
        }
    }
}

fn apply_row_filters(batch: &ArrowRecordBatch, filters: &[io::ParquetRowFilter]) -> Result<ArrowRecordBatch, io::ArrowError> {
    use arrow_array::Array as ArrowArrayTrait;

    if filters.is_empty() || batch.num_rows() == 0 {
        return Ok(batch.clone());
    }

    let mut mask: Option<arrow_array::BooleanArray> = None;

    for filter in filters {
        let column = batch.column_by_name(&filter.column)
            .ok_or_else(|| io::ArrowError::InvalidArgument(format!("Column '{}' not found", filter.column)))?;

        let filter_mask = apply_single_filter(column, &filter.op, &filter.value)?;

        mask = match mask {
            None => Some(filter_mask),
            Some(existing) => {
                let combined = arrow_arith::boolean::and(&existing, &filter_mask)
                    .map_err(|e| io::ArrowError::ComputeError(e.to_string()))?;
                Some(combined)
            }
        };
    }

    if let Some(m) = mask {
        let filtered = arrow_select::filter::filter_record_batch(batch, &m)
            .map_err(|e| io::ArrowError::ComputeError(e.to_string()))?;
        Ok(filtered)
    } else {
        Ok(batch.clone())
    }
}

fn apply_single_filter(column: &ArrayRef, op: &io::ParquetFilterOp, value: &str) -> Result<arrow_array::BooleanArray, io::ArrowError> {
    use arrow_array::Array as ArrowArrayTrait;
    use arrow_ord::cmp;

    macro_rules! compare_numeric {
        ($arr_type:ty, $parse_type:ty, $column:expr, $op:expr, $value:expr) => {{
            if let Some(arr) = $column.as_any().downcast_ref::<$arr_type>() {
                let scalar: $parse_type = $value.parse()
                    .map_err(|_| io::ArrowError::InvalidArgument(format!("Cannot parse '{}' as number", $value)))?;
                let scalar_arr: $arr_type = vec![Some(scalar); arr.len()].into_iter().collect();
                let result = match $op {
                    io::ParquetFilterOp::Eq => cmp::eq(arr, &scalar_arr),
                    io::ParquetFilterOp::NotEq => cmp::neq(arr, &scalar_arr),
                    io::ParquetFilterOp::Lt => cmp::lt(arr, &scalar_arr),
                    io::ParquetFilterOp::LtEq => cmp::lt_eq(arr, &scalar_arr),
                    io::ParquetFilterOp::Gt => cmp::gt(arr, &scalar_arr),
                    io::ParquetFilterOp::GtEq => cmp::gt_eq(arr, &scalar_arr),
                };
                return result.map_err(|e| io::ArrowError::ComputeError(e.to_string()));
            }
        }};
    }

    compare_numeric!(arrow_array::Int64Array, i64, column, op, value);
    compare_numeric!(arrow_array::Int32Array, i32, column, op, value);
    compare_numeric!(arrow_array::Int16Array, i16, column, op, value);
    compare_numeric!(arrow_array::Int8Array, i8, column, op, value);
    compare_numeric!(arrow_array::UInt64Array, u64, column, op, value);
    compare_numeric!(arrow_array::UInt32Array, u32, column, op, value);
    compare_numeric!(arrow_array::UInt16Array, u16, column, op, value);
    compare_numeric!(arrow_array::UInt8Array, u8, column, op, value);
    compare_numeric!(arrow_array::Float64Array, f64, column, op, value);
    compare_numeric!(arrow_array::Float32Array, f32, column, op, value);

    if let Some(arr) = column.as_any().downcast_ref::<arrow_array::StringArray>() {
        let scalar_arr: arrow_array::StringArray = vec![Some(value); arr.len()].into_iter().collect();
        let result = match op {
            io::ParquetFilterOp::Eq => cmp::eq(arr, &scalar_arr),
            io::ParquetFilterOp::NotEq => cmp::neq(arr, &scalar_arr),
            io::ParquetFilterOp::Lt => cmp::lt(arr, &scalar_arr),
            io::ParquetFilterOp::LtEq => cmp::lt_eq(arr, &scalar_arr),
            io::ParquetFilterOp::Gt => cmp::gt(arr, &scalar_arr),
            io::ParquetFilterOp::GtEq => cmp::gt_eq(arr, &scalar_arr),
        };
        return result.map_err(|e| io::ArrowError::ComputeError(e.to_string()));
    }

    if let Some(arr) = column.as_any().downcast_ref::<arrow_array::BooleanArray>() {
        let scalar: bool = value.parse()
            .map_err(|_| io::ArrowError::InvalidArgument(format!("Cannot parse '{}' as boolean", value)))?;
        let scalar_arr: arrow_array::BooleanArray = vec![Some(scalar); arr.len()].into_iter().collect();
        let result = match op {
            io::ParquetFilterOp::Eq => cmp::eq(arr, &scalar_arr),
            io::ParquetFilterOp::NotEq => cmp::neq(arr, &scalar_arr),
            _ => return Err(io::ArrowError::InvalidArgument("Boolean only supports eq/not-eq".to_string())),
        };
        return result.map_err(|e| io::ArrowError::ComputeError(e.to_string()));
    }

    Err(io::ArrowError::InvalidArgument(format!("Unsupported column type for filtering: {:?}", column.data_type())))
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

    fn parquet_read_metadata_only(data: Vec<u8>) -> Result<io::ParquetFileMetadata, io::ArrowError> {
        let bytes = Bytes::from(data);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;
        let metadata = builder.metadata();

        Ok(io::ParquetFileMetadata {
            num_rows: metadata.file_metadata().num_rows() as u64,
            num_row_groups: metadata.num_row_groups() as u32,
            created_by: metadata.file_metadata().created_by().map(|s| s.to_string()),
            key_value_metadata: metadata
                .file_metadata()
                .key_value_metadata()
                .map(|kv| {
                    kv.iter()
                        .filter_map(|e| {
                            e.value.as_ref().map(|v| (e.key.clone(), v.clone()))
                        })
                        .collect()
                })
                .unwrap_or_default(),
        })
    }

    fn parquet_get_column_chunk_stats(data: Vec<u8>, row_group: u32, column: String) -> Result<io::ParquetColumnChunkStats, io::ArrowError> {
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
        let schema_descr = metadata.file_metadata().schema_descr();

        // Find column by name
        let mut col_idx = None;
        for i in 0..rg_metadata.num_columns() {
            let col_descr = schema_descr.column(i);
            if col_descr.name() == column {
                col_idx = Some(i);
                break;
            }
        }

        let col_idx = col_idx.ok_or_else(|| {
            io::ArrowError::InvalidArgument(format!(
                "Column '{}' not found in row group {}",
                column, row_group
            ))
        })?;

        let col_metadata = rg_metadata.column(col_idx);
        let col_descr = schema_descr.column(col_idx);
        let statistics = col_metadata.statistics();

        // Convert statistics min/max to string representation
        let (min_str, max_str) = if let Some(stats) = statistics {
            (
                stats_value_to_string(stats.min_bytes_opt(), col_descr.physical_type()),
                stats_value_to_string(stats.max_bytes_opt(), col_descr.physical_type()),
            )
        } else {
            (None, None)
        };

        Ok(io::ParquetColumnChunkStats {
            column: column.clone(),
            physical_type: format!("{:?}", col_descr.physical_type()),
            num_values: col_metadata.num_values(),
            null_count: statistics.and_then(|s| s.null_count_opt()).map(|c| c as i64),
            distinct_count: statistics.and_then(|s| s.distinct_count_opt()).map(|c| c as i64),
            min_value: min_str,
            max_value: max_str,
            compressed_size: col_metadata.compressed_size(),
            uncompressed_size: col_metadata.uncompressed_size(),
        })
    }

    fn parquet_row_group_stats(data: Vec<u8>, row_group: u32) -> Result<Vec<io::ParquetColumnChunkStats>, io::ArrowError> {
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
        let schema_descr = metadata.file_metadata().schema_descr();
        let mut results = Vec::new();

        for i in 0..rg_metadata.num_columns() {
            let col_metadata = rg_metadata.column(i);
            let col_descr = schema_descr.column(i);
            let statistics = col_metadata.statistics();

            // Convert statistics min/max to string representation
            let (min_str, max_str) = if let Some(stats) = statistics {
                (
                    stats_value_to_string(stats.min_bytes_opt(), col_descr.physical_type()),
                    stats_value_to_string(stats.max_bytes_opt(), col_descr.physical_type()),
                )
            } else {
                (None, None)
            };

            results.push(io::ParquetColumnChunkStats {
                column: col_descr.name().to_string(),
                physical_type: format!("{:?}", col_descr.physical_type()),
                num_values: col_metadata.num_values(),
                null_count: statistics.and_then(|s| s.null_count_opt()).map(|c| c as i64),
                distinct_count: statistics.and_then(|s| s.distinct_count_opt()).map(|c| c as i64),
                min_value: min_str,
                max_value: max_str,
                compressed_size: col_metadata.compressed_size(),
                uncompressed_size: col_metadata.uncompressed_size(),
            });
        }

        Ok(results)
    }

    fn parquet_has_dictionary(data: Vec<u8>, row_group: u32, column: String) -> Result<bool, io::ArrowError> {
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
        let schema_descr = metadata.file_metadata().schema_descr();

        // Find column by name
        let mut col_idx = None;
        for i in 0..rg_metadata.num_columns() {
            let col_descr = schema_descr.column(i);
            if col_descr.name() == column {
                col_idx = Some(i);
                break;
            }
        }

        let col_idx = col_idx.ok_or_else(|| {
            io::ArrowError::InvalidArgument(format!(
                "Column '{}' not found in row group {}",
                column, row_group
            ))
        })?;

        let col_metadata = rg_metadata.column(col_idx);

        // Check if dictionary page offset is set (indicates dictionary encoding)
        Ok(col_metadata.dictionary_page_offset().is_some())
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

    fn parquet_read_advanced(data: Vec<u8>, options: io::ParquetReadOptions) -> Result<Vec<record_batch::RecordBatch>, io::ArrowError> {
        let bytes = Bytes::from(data);
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(to_io_error)?;

        let schema = builder.schema().clone();
        let parquet_schema = builder.parquet_schema().clone();

        // Apply column projection
        if !options.projection.is_empty() {
            let indices: Vec<usize> = options.projection
                .iter()
                .filter_map(|name| schema.index_of(name).ok())
                .collect();
            builder = builder.with_projection(parquet::arrow::ProjectionMask::leaves(
                &parquet_schema,
                indices,
            ));
        }

        // Apply row group selection
        if !options.row_groups.is_empty() {
            builder = builder.with_row_groups(
                options.row_groups.into_iter().map(|i| i as usize).collect()
            );
        }

        // Apply batch size
        if let Some(batch_size) = options.batch_size {
            builder = builder.with_batch_size(batch_size as usize);
        }

        let reader = builder.build().map_err(to_io_error)?;
        let mut batches: Vec<ArrowRecordBatch> = reader.collect::<Result<Vec<_>, _>>().map_err(to_io_error)?;

        // Apply row filters post-read (predicate pushdown at row level)
        // This filters rows after reading but before returning
        if !options.filters.is_empty() {
            batches = batches.into_iter()
                .map(|batch| apply_row_filters(&batch, &options.filters))
                .collect::<Result<Vec<_>, _>>()?;
        }

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

    fn parquet_write_extended(
        batches: Vec<record_batch::RecordBatchBorrow<'_>>,
        options: io::ParquetWriteOptionsExtended,
    ) -> Result<Vec<u8>, io::ArrowError> {
        if batches.is_empty() {
            return Err(io::ArrowError::InvalidArgument("No batches to write".to_string()));
        }

        let first_batch = batches[0].get::<RecordBatchImpl>();
        let schema = first_batch.inner.schema();

        let mut props_builder = WriterProperties::builder();

        // Apply compression
        props_builder = props_builder.set_compression(to_parquet_compression(options.compression)?);

        // Apply global options
        if let Some(page_size) = options.data_page_size {
            props_builder = props_builder.set_data_page_size_limit(page_size as usize);
        }
        if let Some(dict_page_size) = options.dictionary_page_size {
            props_builder = props_builder.set_dictionary_page_size_limit(dict_page_size as usize);
        }
        if let Some(row_group_size) = options.row_group_size {
            props_builder = props_builder.set_max_row_group_size(row_group_size as usize);
        }
        if let Some(max_rows) = options.max_row_group_rows {
            props_builder = props_builder.set_max_row_group_size(max_rows as usize);
        }
        if let Some(write_stats) = options.write_statistics {
            if write_stats {
                props_builder = props_builder.set_statistics_enabled(
                    parquet::file::properties::EnabledStatistics::Chunk,
                );
            }
        }
        if let Some(created_by) = options.created_by {
            props_builder = props_builder.set_created_by(created_by);
        }

        // Apply key-value metadata
        if !options.key_value_metadata.is_empty() {
            let kv_metadata: Vec<parquet::file::metadata::KeyValue> = options.key_value_metadata
                .into_iter()
                .map(|(key, value)| parquet::file::metadata::KeyValue::new(key, value))
                .collect();
            props_builder = props_builder.set_key_value_metadata(Some(kv_metadata));
        }

        // Apply per-column options
        for col_opts in options.column_options {
            let col_path = parquet::schema::types::ColumnPath::from(col_opts.column.clone());

            if let Some(encoding) = col_opts.encoding {
                let parquet_encoding = match encoding {
                    io::ParquetEncoding::Plain => parquet::basic::Encoding::PLAIN,
                    io::ParquetEncoding::PlainDictionary => parquet::basic::Encoding::PLAIN_DICTIONARY,
                    io::ParquetEncoding::Rle => parquet::basic::Encoding::RLE,
                    io::ParquetEncoding::RleDictionary => parquet::basic::Encoding::RLE_DICTIONARY,
                    io::ParquetEncoding::DeltaBinaryPacked => parquet::basic::Encoding::DELTA_BINARY_PACKED,
                    io::ParquetEncoding::DeltaLengthByteArray => parquet::basic::Encoding::DELTA_LENGTH_BYTE_ARRAY,
                    io::ParquetEncoding::DeltaByteArray => parquet::basic::Encoding::DELTA_BYTE_ARRAY,
                    io::ParquetEncoding::ByteStreamSplit => parquet::basic::Encoding::BYTE_STREAM_SPLIT,
                };
                props_builder = props_builder.set_column_encoding(col_path.clone(), parquet_encoding);
            }

            if let Some(dict_enabled) = col_opts.dictionary_enabled {
                props_builder = props_builder.set_column_dictionary_enabled(col_path.clone(), dict_enabled);
            }

            if let Some(bloom_enabled) = col_opts.bloom_filter_enabled {
                if bloom_enabled {
                    let mut bloom_props = parquet::file::properties::BloomFilterProperties::default();
                    if let Some(fpp) = col_opts.bloom_filter_fpp {
                        bloom_props.fpp = fpp;
                    }
                    if let Some(ndv) = col_opts.bloom_filter_ndv {
                        bloom_props.ndv = ndv;
                    }
                    props_builder = props_builder.set_column_bloom_filter_enabled(col_path.clone(), true);
                    props_builder = props_builder.set_column_bloom_filter_fpp(col_path.clone(), bloom_props.fpp);
                    props_builder = props_builder.set_column_bloom_filter_ndv(col_path, bloom_props.ndv);
                }
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

    // ========== Parquet Bloom Filter Operations ==========

    fn parquet_has_bloom_filter(data: Vec<u8>, row_group: u32, column: String) -> Result<bool, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;
        let metadata = reader.metadata();

        if row_group as usize >= metadata.num_row_groups() {
            return Err(io::ArrowError::InvalidArgument(format!(
                "Row group {} out of range (file has {} row groups)",
                row_group, metadata.num_row_groups()
            )));
        }

        let rg_metadata = metadata.row_group(row_group as usize);

        // Find column index
        let schema_descr = metadata.file_metadata().schema_descr();
        let col_idx = (0..rg_metadata.num_columns())
            .find(|&i| {
                let col_descr = rg_metadata.column(i);
                col_descr.column_path().string() == column
            });

        match col_idx {
            Some(idx) => {
                let col_chunk = rg_metadata.column(idx);
                Ok(col_chunk.bloom_filter_offset().is_some())
            }
            None => Err(io::ArrowError::InvalidArgument(format!("Column '{}' not found", column))),
        }
    }

    fn parquet_bloom_check_i64(data: Vec<u8>, row_group: u32, column: String, value: i64) -> Result<bool, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;
        use parquet::bloom_filter::Sbbf;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;

        let rg_reader = reader.get_row_group(row_group as usize).map_err(to_io_error)?;

        // Find column index
        let metadata = reader.metadata();
        let rg_metadata = metadata.row_group(row_group as usize);

        let col_idx = (0..rg_metadata.num_columns())
            .find(|&i| rg_metadata.column(i).column_path().string() == column);

        match col_idx {
            Some(idx) => {
                match rg_reader.get_column_bloom_filter(idx) {
                    Some(bloom) => Ok(bloom.check(&value)),
                    None => Err(io::ArrowError::InvalidArgument(format!(
                        "No bloom filter for column '{}' in row group {}", column, row_group
                    ))),
                }
            }
            None => Err(io::ArrowError::InvalidArgument(format!("Column '{}' not found", column))),
        }
    }

    fn parquet_bloom_check_string(data: Vec<u8>, row_group: u32, column: String, value: String) -> Result<bool, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;

        let rg_reader = reader.get_row_group(row_group as usize).map_err(to_io_error)?;

        // Find column index
        let metadata = reader.metadata();
        let rg_metadata = metadata.row_group(row_group as usize);

        let col_idx = (0..rg_metadata.num_columns())
            .find(|&i| rg_metadata.column(i).column_path().string() == column);

        match col_idx {
            Some(idx) => {
                match rg_reader.get_column_bloom_filter(idx) {
                    Some(bloom) => Ok(bloom.check(&parquet::data_type::ByteArray::from(value.as_bytes()))),
                    None => Err(io::ArrowError::InvalidArgument(format!(
                        "No bloom filter for column '{}' in row group {}", column, row_group
                    ))),
                }
            }
            None => Err(io::ArrowError::InvalidArgument(format!("Column '{}' not found", column))),
        }
    }

    fn parquet_bloom_check_binary(data: Vec<u8>, row_group: u32, column: String, value: Vec<u8>) -> Result<bool, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;

        let rg_reader = reader.get_row_group(row_group as usize).map_err(to_io_error)?;

        // Find column index
        let metadata = reader.metadata();
        let rg_metadata = metadata.row_group(row_group as usize);

        let col_idx = (0..rg_metadata.num_columns())
            .find(|&i| rg_metadata.column(i).column_path().string() == column);

        match col_idx {
            Some(idx) => {
                match rg_reader.get_column_bloom_filter(idx) {
                    Some(bloom) => Ok(bloom.check(&parquet::data_type::ByteArray::from(value))),
                    None => Err(io::ArrowError::InvalidArgument(format!(
                        "No bloom filter for column '{}' in row group {}", column, row_group
                    ))),
                }
            }
            None => Err(io::ArrowError::InvalidArgument(format!("Column '{}' not found", column))),
        }
    }

    // ========== Parquet Page Index Access ==========

    fn parquet_has_page_index(data: Vec<u8>) -> Result<bool, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;
        let metadata = reader.metadata();

        // Check if any row group has offset index
        for rg_idx in 0..metadata.num_row_groups() {
            if let Some(offset_index) = metadata.offset_index() {
                if rg_idx < offset_index.len() && !offset_index[rg_idx].is_empty() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn parquet_page_locations(
        data: Vec<u8>,
        row_group: u32,
        column: String,
    ) -> Result<Vec<io::ParquetPageLocation>, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;
        let metadata = reader.metadata();

        let rg_idx = row_group as usize;
        if rg_idx >= metadata.num_row_groups() {
            return Err(io::ArrowError::InvalidArgument(format!(
                "Row group {} out of range (file has {} row groups)",
                row_group, metadata.num_row_groups()
            )));
        }

        // Find column index
        let rg_metadata = metadata.row_group(rg_idx);
        let col_idx = (0..rg_metadata.num_columns())
            .find(|&i| rg_metadata.column(i).column_path().string() == column);

        let col_idx = col_idx.ok_or_else(|| {
            io::ArrowError::InvalidArgument(format!("Column '{}' not found", column))
        })?;

        // Get offset index (page locations)
        let offset_index = metadata.offset_index().ok_or_else(|| {
            io::ArrowError::InvalidArgument("File does not have page index".to_string())
        })?;

        if rg_idx >= offset_index.len() {
            return Err(io::ArrowError::InvalidArgument("Offset index not available for row group".to_string()));
        }

        if col_idx >= offset_index[rg_idx].len() {
            return Err(io::ArrowError::InvalidArgument("Offset index not available for column".to_string()));
        }

        let page_locations: Vec<io::ParquetPageLocation> = offset_index[rg_idx][col_idx]
            .page_locations
            .iter()
            .map(|loc| io::ParquetPageLocation {
                offset: loc.offset,
                compressed_size: loc.compressed_page_size as u32,
                first_row_index: loc.first_row_index,
            })
            .collect();

        Ok(page_locations)
    }

    fn parquet_get_page_stats(
        data: Vec<u8>,
        row_group: u32,
        column: String,
    ) -> Result<Option<io::ParquetPageStats>, io::ArrowError> {
        use parquet::file::reader::FileReader;
        use parquet::file::serialized_reader::SerializedFileReader;

        let bytes = Bytes::from(data);
        let reader = SerializedFileReader::new(bytes).map_err(to_io_error)?;
        let metadata = reader.metadata();

        let rg_idx = row_group as usize;
        if rg_idx >= metadata.num_row_groups() {
            return Err(io::ArrowError::InvalidArgument(format!(
                "Row group {} out of range (file has {} row groups)",
                row_group, metadata.num_row_groups()
            )));
        }

        // Find column index
        let rg_metadata = metadata.row_group(rg_idx);
        let col_idx = (0..rg_metadata.num_columns())
            .find(|&i| rg_metadata.column(i).column_path().string() == column);

        let col_idx = col_idx.ok_or_else(|| {
            io::ArrowError::InvalidArgument(format!("Column '{}' not found", column))
        })?;

        // Get column index (page statistics)
        let column_index = match metadata.column_index() {
            Some(ci) => ci,
            None => return Ok(None),
        };

        if rg_idx >= column_index.len() {
            return Ok(None);
        }

        if col_idx >= column_index[rg_idx].len() {
            return Ok(None);
        }

        let col_index = &column_index[rg_idx][col_idx];

        // Extract null counts which are available directly
        let null_counts: Vec<Option<i64>> = match col_index.null_counts() {
            Some(counts) => counts.iter().map(|&c| Some(c)).collect(),
            None => return Ok(None),
        };

        if null_counts.is_empty() {
            return Ok(None);
        }

        // Min/max values require more complex type handling
        // For now, we return empty vectors for min/max but provide null counts
        let num_pages = null_counts.len();
        let min_values: Vec<Option<String>> = vec![None; num_pages];
        let max_values: Vec<Option<String>> = vec![None; num_pages];

        Ok(Some(io::ParquetPageStats {
            min_values,
            max_values,
            null_counts,
        }))
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
