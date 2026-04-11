package matrix

import "math"

// REF returns the Row Echelon Form of the matrix using Gaussian elimination
// with partial pivoting.
//
// In REF:
//   - All zero rows are at the bottom
//   - The leading entry (pivot) of each non-zero row is to the right of
//     the pivot in the row above
//   - All entries below a pivot are zero
//
// The result is always float64 regardless of the input type, since
// row reduction inherently produces fractions.
func (m *Matrix[T]) REF() (*Matrix[float64], error) {
	// Convert to float64 for computation
	a := toFloat64Matrix(m)
	ref(a)
	return a, nil
}

// ref performs in-place Gaussian elimination with partial pivoting.
// This is the internal workhorse used by REF, RREF, Det, and Rank.
func ref(a *Matrix[float64]) int {
	rows, cols := a.rows, a.cols
	pivotRow := 0
	pivotCount := 0

	for col := 0; col < cols && pivotRow < rows; col++ {
		// Partial pivoting: find the row with the largest absolute value
		// in the current column at or below the pivot row.
		maxVal := math.Abs(a.data[pivotRow][col])
		maxRow := pivotRow
		for i := pivotRow + 1; i < rows; i++ {
			if v := math.Abs(a.data[i][col]); v > maxVal {
				maxVal = v
				maxRow = i
			}
		}

		// If the max value is effectively zero, skip this column
		if maxVal < Epsilon {
			continue
		}

		// Swap rows
		if maxRow != pivotRow {
			a.data[pivotRow], a.data[maxRow] = a.data[maxRow], a.data[pivotRow]
		}

		// Eliminate entries below the pivot
		pivot := a.data[pivotRow][col]
		for i := pivotRow + 1; i < rows; i++ {
			if math.Abs(a.data[i][col]) < Epsilon {
				continue
			}
			factor := a.data[i][col] / pivot
			a.data[i][col] = 0 // Explicitly set to avoid floating-point drift
			for j := col + 1; j < cols; j++ {
				a.data[i][j] -= factor * a.data[pivotRow][j]
			}
		}

		pivotRow++
		pivotCount++
	}

	return pivotCount
}

// RREF returns the Reduced Row Echelon Form of the matrix using
// Gauss-Jordan elimination with partial pivoting.
//
// In RREF (extends REF with):
//   - Each pivot is exactly 1
//   - Each pivot is the only non-zero entry in its column
//
// RREF is unique for any given matrix — unlike REF, which depends on
// the elimination order.
func (m *Matrix[T]) RREF() (*Matrix[float64], error) {
	a := toFloat64Matrix(m)
	rref(a)
	return a, nil
}

// rref performs in-place Gauss-Jordan elimination.
func rref(a *Matrix[float64]) int {
	rows, cols := a.rows, a.cols
	pivotRow := 0
	pivotCount := 0

	for col := 0; col < cols && pivotRow < rows; col++ {
		// Partial pivoting
		maxVal := math.Abs(a.data[pivotRow][col])
		maxRow := pivotRow
		for i := pivotRow + 1; i < rows; i++ {
			if v := math.Abs(a.data[i][col]); v > maxVal {
				maxVal = v
				maxRow = i
			}
		}

		if maxVal < Epsilon {
			continue
		}

		// Swap rows
		if maxRow != pivotRow {
			a.data[pivotRow], a.data[maxRow] = a.data[maxRow], a.data[pivotRow]
		}

		// Scale pivot row so the pivot becomes 1
		pivot := a.data[pivotRow][col]
		for j := col; j < cols; j++ {
			a.data[pivotRow][j] /= pivot
		}

		// Eliminate all other entries in this column (above AND below)
		for i := 0; i < rows; i++ {
			if i == pivotRow || math.Abs(a.data[i][col]) < Epsilon {
				continue
			}
			factor := a.data[i][col]
			a.data[i][col] = 0
			for j := col + 1; j < cols; j++ {
				a.data[i][j] -= factor * a.data[pivotRow][j]
			}
		}

		pivotRow++
		pivotCount++
	}

	return pivotCount
}

// Inverse returns the multiplicative inverse of a square matrix using
// Gauss-Jordan elimination on the augmented matrix [A | I].
//
// The inverse A⁻¹ satisfies: A * A⁻¹ = A⁻¹ * A = I
//
// Returns ErrNotSquare if the matrix isn't square.
// Returns ErrNotInvertible if the matrix is singular (det ≈ 0).
func (m *Matrix[T]) Inverse() (*Matrix[float64], error) {
	if m.rows != m.cols {
		return nil, ErrNotSquare
	}

	n := m.rows

	// Build augmented matrix [A | I] of size n × 2n
	aug := Zeros[float64](n, 2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug.data[i][j] = toFloat64(m.data[i][j])
		}
		aug.data[i][n+i] = 1.0
	}

	// Apply Gauss-Jordan elimination
	for col := 0; col < n; col++ {
		// Partial pivoting
		maxVal := math.Abs(aug.data[col][col])
		maxRow := col
		for i := col + 1; i < n; i++ {
			if v := math.Abs(aug.data[i][col]); v > maxVal {
				maxVal = v
				maxRow = i
			}
		}

		if maxVal < Epsilon {
			return nil, ErrNotInvertible
		}

		// Swap rows
		if maxRow != col {
			aug.data[col], aug.data[maxRow] = aug.data[maxRow], aug.data[col]
		}

		// Scale pivot row
		pivot := aug.data[col][col]
		for j := col; j < 2*n; j++ {
			aug.data[col][j] /= pivot
		}

		// Eliminate all other rows
		for i := 0; i < n; i++ {
			if i == col {
				continue
			}
			factor := aug.data[i][col]
			for j := col; j < 2*n; j++ {
				aug.data[i][j] -= factor * aug.data[col][j]
			}
		}
	}

	// Extract the right half (the inverse)
	result := Zeros[float64](n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result.data[i][j] = aug.data[i][n+j]
		}
	}

	return result, nil
}
