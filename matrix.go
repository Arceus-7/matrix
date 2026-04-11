package matrix

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand"
)

// Epsilon is the tolerance used for floating-point comparisons.
// Operations like singularity checks compare against this value
// rather than testing exact equality to zero.
var Epsilon = 1e-9

// Numeric is the type constraint for all matrix element types.
// It supports integers, floats, and complex numbers.
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
	~float32 | ~float64 |
	~complex64 | ~complex128
}

// RealNumeric is a stricter constraint for operations that require
// ordered comparisons or real-valued math functions (Abs, Sqrt, etc.).
// Complex types are excluded because they lack a natural ordering.
type RealNumeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
	~float32 | ~float64
}

// Float is a constraint for floating-point types only.
// Used for operations like Random that only make sense for floats.
type Float interface {
	~float32 | ~float64
}

// Matrix is the core generic matrix type.
// It stores elements in row-major order as a 2D slice.
// All operations return new matrices — the original is never mutated
// (except Set, which operates on a pointer receiver).
type Matrix[T Numeric] struct {
	data [][]T
	rows int
	cols int
}

// New creates a Matrix from a 2D slice. It validates that the input is
// rectangular (all rows have the same length) and performs a deep copy
// so the caller's original slice is not aliased.
//
// Returns nil and an error if the input is empty or jagged.
func New[T Numeric](data [][]T) (*Matrix[T], error) {
	rows := len(data)
	if rows == 0 {
		return nil, ErrEmptyMatrix
	}
	cols := len(data[0])
	if cols == 0 {
		return nil, ErrEmptyMatrix
	}

	// Validate rectangular and deep-copy
	copied := make([][]T, rows)
	for i := 0; i < rows; i++ {
		if len(data[i]) != cols {
			return nil, errors.New("matrix: jagged input — all rows must have the same number of columns")
		}
		copied[i] = make([]T, cols)
		copy(copied[i], data[i])
	}

	return &Matrix[T]{
		data: copied,
		rows: rows,
		cols: cols,
	}, nil
}

// MustNew is like New but panics on error. Useful for test code and
// literals where you know the input is valid.
func MustNew[T Numeric](data [][]T) *Matrix[T] {
	m, err := New(data)
	if err != nil {
		panic(err)
	}
	return m
}

// Identity returns an n×n identity matrix (1s on the diagonal, 0s elsewhere).
// Panics if n <= 0.
func Identity[T Numeric](n int) *Matrix[T] {
	if n <= 0 {
		panic("matrix: Identity requires n > 0")
	}
	data := make([][]T, n)
	for i := 0; i < n; i++ {
		data[i] = make([]T, n)
		data[i][i] = T(1)
	}
	return &Matrix[T]{data: data, rows: n, cols: n}
}

// Zeros returns a rows×cols matrix filled with zeros.
// Panics if rows or cols <= 0.
func Zeros[T Numeric](rows, cols int) *Matrix[T] {
	if rows <= 0 || cols <= 0 {
		panic("matrix: Zeros requires rows > 0 and cols > 0")
	}
	data := make([][]T, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]T, cols)
	}
	return &Matrix[T]{data: data, rows: rows, cols: cols}
}

// Ones returns a rows×cols matrix filled with ones.
// Panics if rows or cols <= 0.
func Ones[T Numeric](rows, cols int) *Matrix[T] {
	if rows <= 0 || cols <= 0 {
		panic("matrix: Ones requires rows > 0 and cols > 0")
	}
	data := make([][]T, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]T, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = T(1)
		}
	}
	return &Matrix[T]{data: data, rows: rows, cols: cols}
}

// Random returns a rows×cols matrix filled with random float values in [0, 1).
// Only available for float32 and float64 types.
// Panics if rows or cols <= 0.
func Random[T Float](rows, cols int) *Matrix[T] {
	if rows <= 0 || cols <= 0 {
		panic("matrix: Random requires rows > 0 and cols > 0")
	}
	data := make([][]T, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]T, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = T(rand.Float64())
		}
	}
	return &Matrix[T]{data: data, rows: rows, cols: cols}
}

// Shape returns the number of rows and columns in the matrix.
func (m *Matrix[T]) Shape() (int, int) {
	return m.rows, m.cols
}

// Rows returns the number of rows.
func (m *Matrix[T]) Rows() int {
	return m.rows
}

// Cols returns the number of columns.
func (m *Matrix[T]) Cols() int {
	return m.cols
}

// At returns the element at row i, column j (0-indexed).
// Returns an error if the indices are out of bounds.
func (m *Matrix[T]) At(i, j int) (T, error) {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		var zero T
		return zero, ErrOutOfBounds
	}
	return m.data[i][j], nil
}

// Set sets the element at row i, column j (0-indexed).
// This is the only mutating operation — it uses a pointer receiver.
// Returns an error if the indices are out of bounds.
func (m *Matrix[T]) Set(i, j int, val T) error {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		return ErrOutOfBounds
	}
	m.data[i][j] = val
	return nil
}

// Copy returns a deep copy of the matrix.
func (m *Matrix[T]) Copy() *Matrix[T] {
	data := make([][]T, m.rows)
	for i := 0; i < m.rows; i++ {
		data[i] = make([]T, m.cols)
		copy(data[i], m.data[i])
	}
	return &Matrix[T]{data: data, rows: m.rows, cols: m.cols}
}

// Data returns a deep copy of the underlying 2D slice.
// The returned slice is safe to modify without affecting the matrix.
func (m *Matrix[T]) Data() [][]T {
	data := make([][]T, m.rows)
	for i := 0; i < m.rows; i++ {
		data[i] = make([]T, m.cols)
		copy(data[i], m.data[i])
	}
	return data
}

// Equals returns true if two matrices have the same shape and identical elements.
// For floating-point types, exact equality is used — see ApproxEquals for
// epsilon-based comparison.
func (m *Matrix[T]) Equals(other *Matrix[T]) bool {
	if m.rows != other.rows || m.cols != other.cols {
		return false
	}
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if m.data[i][j] != other.data[i][j] {
				return false
			}
		}
	}
	return true
}

// toFloat64 converts any Numeric value to float64 for internal computation.
func toFloat64[T Numeric](v T) float64 {
	// Use type switch on any to handle all numeric types
	switch val := any(v).(type) {
	case int:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case float32:
		return float64(val)
	case float64:
		return val
	case complex64:
		return float64(real(val))
	case complex128:
		return real(val)
	default:
		return 0
	}
}

// toFloat64Matrix converts any Matrix[T] to Matrix[float64] for operations
// that require floating-point arithmetic (RREF, Inverse, LU, etc.).
func toFloat64Matrix[T Numeric](m *Matrix[T]) *Matrix[float64] {
	data := make([][]float64, m.rows)
	for i := 0; i < m.rows; i++ {
		data[i] = make([]float64, m.cols)
		for j := 0; j < m.cols; j++ {
			data[i][j] = toFloat64(m.data[i][j])
		}
	}
	return &Matrix[float64]{data: data, rows: m.rows, cols: m.cols}
}

// absFloat64 returns the absolute value of a Numeric value as float64.
// For complex types, it returns the complex modulus.
func absFloat64[T Numeric](v T) float64 {
	switch val := any(v).(type) {
	case complex64:
		return cmplx.Abs(complex128(val))
	case complex128:
		return cmplx.Abs(val)
	default:
		f := toFloat64(v)
		return math.Abs(f)
	}
}
