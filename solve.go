package matrix

import "math"

// Solve solves the linear system Ax = b using LU decomposition with
// forward and backward substitution.
//
// Given:
//   - A is an n×n coefficient matrix
//   - b is an n×1 column vector (right-hand side)
//
// Returns x such that Ax = b.
//
// Algorithm:
//  1. Decompose PA = LU (with partial pivoting)
//  2. Permute b according to P → Pb
//  3. Solve Ly = Pb (forward substitution)
//  4. Solve Ux = y (backward substitution)
//
// Returns ErrNotSquare if A isn't square.
// Returns ErrDimensionMismatch if b doesn't have the right dimensions.
// Returns ErrNotVector if b isn't a column vector.
// Returns ErrSingular if A is singular.
func Solve[T Numeric](A *Matrix[T], b *Matrix[T]) (*Matrix[float64], error) {
	if A.rows != A.cols {
		return nil, ErrNotSquare
	}

	n := A.rows

	// Validate b dimensions
	if b.rows != n {
		return nil, ErrDimensionMismatch
	}
	if b.cols != 1 {
		return nil, ErrNotVector
	}

	// LU decompose A with permutation tracking
	L, U, perm, err := luWithPerm(A)
	if err != nil {
		return nil, err
	}

	// Convert b to float64 and apply the permutation
	bf := toFloat64Matrix(b)
	pb := make([]float64, n)
	for i := 0; i < n; i++ {
		pb[i] = bf.data[perm[i]][0]
	}

	// Step 1: Forward substitution — solve Ly = Pb
	// L is lower triangular with ones on diagonal
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := pb[i]
		for j := 0; j < i; j++ {
			sum -= L.data[i][j] * y[j]
		}
		y[i] = sum // L[i][i] = 1, so no division needed
	}

	// Step 2: Backward substitution — solve Ux = y
	// U is upper triangular
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < n; j++ {
			sum -= U.data[i][j] * x[j]
		}
		if math.Abs(U.data[i][i]) < Epsilon {
			return nil, ErrSingular
		}
		x[i] = sum / U.data[i][i]
	}

	// Build result as n×1 matrix
	result := Zeros[float64](n, 1)
	for i := 0; i < n; i++ {
		result.data[i][0] = x[i]
	}

	return result, nil
}
