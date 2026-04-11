package matrix

import "math"

// IsSquare returns true if the matrix has the same number of rows and columns.
func (m *Matrix[T]) IsSquare() bool {
	return m.rows == m.cols
}

// IsSymmetric returns true if the matrix equals its own transpose: A = Aᵀ.
// For floating-point types, comparisons use the package-level Epsilon tolerance.
// A non-square matrix is never symmetric.
func (m *Matrix[T]) IsSymmetric() bool {
	if m.rows != m.cols {
		return false
	}
	for i := 0; i < m.rows; i++ {
		for j := i + 1; j < m.cols; j++ {
			diff := absFloat64(m.data[i][j] - m.data[j][i])
			if diff > Epsilon {
				return false
			}
		}
	}
	return true
}

// Trace returns the sum of the diagonal elements. Also known as the "spur."
//
// Mathematically: tr(A) = Σ a[i][i] for i = 0..n-1
//
// Properties:
//   - tr(A + B) = tr(A) + tr(B)
//   - tr(cA) = c * tr(A)
//   - tr(AB) = tr(BA)
//
// Returns ErrNotSquare if the matrix isn't square.
func (m *Matrix[T]) Trace() (T, error) {
	if m.rows != m.cols {
		var zero T
		return zero, ErrNotSquare
	}
	var sum T
	for i := 0; i < m.rows; i++ {
		sum += m.data[i][i]
	}
	return sum, nil
}

// Norm returns the Frobenius norm of the matrix.
//
// Mathematically: ‖A‖_F = √(Σ |a[i][j]|²)
//
// The Frobenius norm is the matrix analogue of the Euclidean vector norm.
// It's always non-negative and equals zero only for the zero matrix.
func (m *Matrix[T]) Norm() float64 {
	var sum float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			v := absFloat64(m.data[i][j])
			sum += v * v
		}
	}
	return math.Sqrt(sum)
}

// Det returns the determinant of a square matrix.
//
// The determinant is computed via LU-style Gaussian elimination with
// partial pivoting. The determinant equals the product of the diagonal
// elements of the upper triangular form, adjusted for row swaps.
//
// Properties:
//   - det(I) = 1
//   - det(AB) = det(A) * det(B)
//   - det(Aᵀ) = det(A)
//   - det(cA) = c^n * det(A) for n×n matrix
//   - A is invertible ⟺ det(A) ≠ 0
//
// Returns ErrNotSquare if the matrix isn't square.
func (m *Matrix[T]) Det() (float64, error) {
	if m.rows != m.cols {
		return 0, ErrNotSquare
	}

	n := m.rows

	// Special case: 1×1
	if n == 1 {
		return toFloat64(m.data[0][0]), nil
	}

	// Special case: 2×2 — avoid full elimination overhead
	if n == 2 {
		a := toFloat64(m.data[0][0])
		b := toFloat64(m.data[0][1])
		c := toFloat64(m.data[1][0])
		d := toFloat64(m.data[1][1])
		return a*d - b*c, nil
	}

	// General case: Gaussian elimination, track sign from row swaps
	a := toFloat64Matrix(m)
	sign := 1.0

	for col := 0; col < n; col++ {
		// Partial pivoting
		maxVal := math.Abs(a.data[col][col])
		maxRow := col
		for i := col + 1; i < n; i++ {
			if v := math.Abs(a.data[i][col]); v > maxVal {
				maxVal = v
				maxRow = i
			}
		}

		if maxVal < Epsilon {
			return 0, nil // Singular matrix
		}

		if maxRow != col {
			a.data[col], a.data[maxRow] = a.data[maxRow], a.data[col]
			sign *= -1
		}

		// Eliminate below pivot
		pivot := a.data[col][col]
		for i := col + 1; i < n; i++ {
			factor := a.data[i][col] / pivot
			for j := col + 1; j < n; j++ {
				a.data[i][j] -= factor * a.data[col][j]
			}
		}
	}

	// Determinant = sign * product of diagonal
	det := sign
	for i := 0; i < n; i++ {
		det *= a.data[i][i]
	}
	return det, nil
}

// Rank returns the rank of the matrix — the number of linearly independent
// rows (equivalently, columns).
//
// Computed by reducing to RREF and counting non-zero rows (pivots).
//
// Properties:
//   - 0 ≤ rank(A) ≤ min(rows, cols)
//   - rank(A) = rank(Aᵀ)
//   - A is full rank if rank(A) = min(rows, cols)
func (m *Matrix[T]) Rank() int {
	a := toFloat64Matrix(m)
	return rref(a)
}

// IsIdentity returns true if the matrix is the identity matrix
// (ones on diagonal, zeros elsewhere), within Epsilon tolerance.
func (m *Matrix[T]) IsIdentity() bool {
	if m.rows != m.cols {
		return false
	}
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if math.Abs(toFloat64(m.data[i][j])-expected) > Epsilon {
				return false
			}
		}
	}
	return true
}

// IsZero returns true if all elements are zero (within Epsilon tolerance).
func (m *Matrix[T]) IsZero() bool {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			if absFloat64(m.data[i][j]) > Epsilon {
				return false
			}
		}
	}
	return true
}
