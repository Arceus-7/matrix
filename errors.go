// Package matrix provides a clean, generic matrix type with support for
// common linear algebra operations — creation, arithmetic, transformations,
// and decompositions.
//
// Built with Go generics, it works with int, float32, float64, complex128,
// and other numeric types. Zero external dependencies.
package matrix

import "errors"

// Sentinel errors for matrix operations.
// All operations that can fail return (result, error) — never panic.
var (
	// ErrDimensionMismatch is returned when matrix dimensions are incompatible
	// for the requested operation (e.g., adding a 2x3 to a 3x2).
	ErrDimensionMismatch = errors.New("matrix: dimension mismatch")

	// ErrNotSquare is returned when an operation requires a square matrix
	// but receives a non-square one (e.g., determinant of a 2x3).
	ErrNotSquare = errors.New("matrix: matrix must be square")

	// ErrSingular is returned when a matrix is singular (determinant ≈ 0)
	// and the operation requires a non-singular matrix.
	ErrSingular = errors.New("matrix: matrix is singular")

	// ErrNotInvertible is returned when a matrix cannot be inverted.
	// This is typically because the matrix is singular.
	ErrNotInvertible = errors.New("matrix: matrix is not invertible")

	// ErrOutOfBounds is returned when an index is outside the valid range
	// for the matrix dimensions.
	ErrOutOfBounds = errors.New("matrix: index out of bounds")

	// ErrEmptyMatrix is returned when attempting to create or operate on
	// a matrix with zero rows or zero columns.
	ErrEmptyMatrix = errors.New("matrix: empty matrix")

	// ErrNotVector is returned when an operation expects a vector (n×1 or 1×n)
	// but receives a general matrix.
	ErrNotVector = errors.New("matrix: expected a vector (n×1)")

	// ErrNotImplemented is returned for operations that are planned but not
	// yet implemented (e.g., SVD in v1).
	ErrNotImplemented = errors.New("matrix: operation not yet implemented")
)
