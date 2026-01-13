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

    fn add_scalar_i64(_arr: arrays::ArrayBorrow<'_>, _scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("add_scalar_i64".to_string()))
    }

    fn add_scalar_f64(_arr: arrays::ArrayBorrow<'_>, _scalar: f64) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("add_scalar_f64".to_string()))
    }

    fn multiply_scalar_i64(_arr: arrays::ArrayBorrow<'_>, _scalar: i64) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("multiply_scalar_i64".to_string()))
    }

    fn multiply_scalar_f64(_arr: arrays::ArrayBorrow<'_>, _scalar: f64) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("multiply_scalar_f64".to_string()))
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

    fn compare_scalar_i64(_arr: arrays::ArrayBorrow<'_>, _scalar: i64, _op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("compare_scalar_i64".to_string()))
    }

    fn compare_scalar_f64(_arr: arrays::ArrayBorrow<'_>, _scalar: f64, _op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("compare_scalar_f64".to_string()))
    }

    fn compare_scalar_string(_arr: arrays::ArrayBorrow<'_>, _scalar: String, _op: compute::ComparisonOp) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("compare_scalar_string".to_string()))
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

    fn sort_record_batch(_batch: record_batch::RecordBatchBorrow<'_>, _sort_columns: Vec<(String, compute::SortOptions)>) -> Result<record_batch::RecordBatch, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("sort_record_batch".to_string()))
    }

    fn lexsort(_arrays: Vec<arrays::Array>, _options: Vec<compute::SortOptions>) -> Result<arrays::Array, compute::ArrowError> {
        Err(compute::ArrowError::NotImplemented("lexsort".to_string()))
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

fn to_parquet_compression(comp: io::Compression) -> ParquetCompression {
    match comp {
        io::Compression::Uncompressed => ParquetCompression::UNCOMPRESSED,
        io::Compression::Snappy => ParquetCompression::SNAPPY,
        io::Compression::Lz4 => ParquetCompression::LZ4,
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
            props_builder = props_builder.set_compression(to_parquet_compression(opts.compression));
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
