package matrix

// Add performs element-wise addition of two matrices.
// Both matrices must have the same dimensions.
//
// Mathematically: C[i][j] = A[i][j] + B[i][j]
func Add[T Numeric](a, b *Matrix[T]) (*Matrix[T], error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, ErrDimensionMismatch
	}

	data := make([][]T, a.rows)
	for i := 0; i < a.rows; i++ {
		data[i] = make([]T, a.cols)
		for j := 0; j < a.cols; j++ {
			data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}

	return &Matrix[T]{data: data, rows: a.rows, cols: a.cols}, nil
}

// Sub performs element-wise subtraction of two matrices.
// Both matrices must have the same dimensions.
//
// Mathematically: C[i][j] = A[i][j] - B[i][j]
func Sub[T Numeric](a, b *Matrix[T]) (*Matrix[T], error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, ErrDimensionMismatch
	}

	data := make([][]T, a.rows)
	for i := 0; i < a.rows; i++ {
		data[i] = make([]T, a.cols)
		for j := 0; j < a.cols; j++ {
			data[i][j] = a.data[i][j] - b.data[i][j]
		}
	}

	return &Matrix[T]{data: data, rows: a.rows, cols: a.cols}, nil
}

// Mul performs matrix multiplication (dot product) of two matrices.
// Requires a.cols == b.rows. The result has dimensions (a.rows × b.cols).
//
// Mathematically: C[i][j] = Σ(k=0..n-1) A[i][k] * B[k][j]
//
// Uses the naive O(n³) algorithm. Fine for small-to-medium matrices.
// For large matrices, consider Strassen (planned for v2).
func Mul[T Numeric](a, b *Matrix[T]) (*Matrix[T], error) {
	if a.cols != b.rows {
		return nil, ErrDimensionMismatch
	}

	data := make([][]T, a.rows)
	for i := 0; i < a.rows; i++ {
		data[i] = make([]T, b.cols)
		for j := 0; j < b.cols; j++ {
			var sum T
			for k := 0; k < a.cols; k++ {
				sum += a.data[i][k] * b.data[k][j]
			}
			data[i][j] = sum
		}
	}

	return &Matrix[T]{data: data, rows: a.rows, cols: b.cols}, nil
}

// Scale multiplies every element of the matrix by a scalar value.
// Always succeeds — returns a new matrix.
//
// Mathematically: C[i][j] = scalar * A[i][j]
func Scale[T Numeric](m *Matrix[T], scalar T) *Matrix[T] {
	data := make([][]T, m.rows)
	for i := 0; i < m.rows; i++ {
		data[i] = make([]T, m.cols)
		for j := 0; j < m.cols; j++ {
			data[i][j] = scalar * m.data[i][j]
		}
	}

	return &Matrix[T]{data: data, rows: m.rows, cols: m.cols}
}

// Transpose returns the transpose of the matrix.
// Rows become columns and columns become rows.
//
// Mathematically: B[j][i] = A[i][j]
//
// Properties:
//   - (Aᵀ)ᵀ = A
//   - (A + B)ᵀ = Aᵀ + Bᵀ
//   - (AB)ᵀ = BᵀAᵀ
func Transpose[T Numeric](m *Matrix[T]) *Matrix[T] {
	data := make([][]T, m.cols)
	for j := 0; j < m.cols; j++ {
		data[j] = make([]T, m.rows)
		for i := 0; i < m.rows; i++ {
			data[j][i] = m.data[i][j]
		}
	}

	return &Matrix[T]{data: data, rows: m.cols, cols: m.rows}
}

// HadamardProduct performs element-wise multiplication of two matrices.
// Both matrices must have the same dimensions. Also known as the
// Schur product.
//
// Mathematically: C[i][j] = A[i][j] * B[i][j]
func HadamardProduct[T Numeric](a, b *Matrix[T]) (*Matrix[T], error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, ErrDimensionMismatch
	}

	data := make([][]T, a.rows)
	for i := 0; i < a.rows; i++ {
		data[i] = make([]T, a.cols)
		for j := 0; j < a.cols; j++ {
			data[i][j] = a.data[i][j] * b.data[i][j]
		}
	}

	return &Matrix[T]{data: data, rows: a.rows, cols: a.cols}, nil
}
