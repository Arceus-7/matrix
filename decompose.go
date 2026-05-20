package matrix

import "math"

// LU performs LU decomposition with partial pivoting.
//
// Decomposes the matrix A into a lower triangular matrix L and an upper
// triangular matrix U such that PA = LU, where P is an implicit
// permutation matrix from row swaps.
//
// The Doolittle algorithm is used: L has ones on its diagonal.
//
// Uses:
//   - Solving linear systems (Ax = b → LUx = b)
//   - Computing determinants efficiently
//   - Matrix inversion
//
// Returns (L, U, error). Returns ErrNotSquare if the matrix isn't square.
// Returns ErrSingular if a zero pivot is encountered.
func (m *Matrix[T]) LU() (*Matrix[float64], *Matrix[float64], error) {
	L, U, _, err := luWithPerm(m)
	return L, U, err
}

// LUP performs LU decomposition with partial pivoting and also returns
// the permutation matrix P.
//
// Returns (L, U, P, error) such that P*A = L*U.
//
// P is an n×n permutation matrix built from the row-swap history.
// Use this when you need to verify or reconstruct the decomposition.
func (m *Matrix[T]) LUP() (*Matrix[float64], *Matrix[float64], *Matrix[float64], error) {
	L, U, perm, err := luWithPerm(m)
	if err != nil {
		return nil, nil, nil, err
	}

	n := m.rows
	P := Zeros[float64](n, n)
	for i := 0; i < n; i++ {
		P.data[i][perm[i]] = 1.0
	}

	return L, U, P, nil
}

// luWithPerm performs LU decomposition with partial pivoting and returns
// the permutation vector. perm[i] = j means row i of the original matrix
// ended up at row j. This is used internally by Solve to correctly
// permute the right-hand side vector.
func luWithPerm[T Numeric](m *Matrix[T]) (*Matrix[float64], *Matrix[float64], []int, error) {
	if m.rows != m.cols {
		return nil, nil, nil, ErrNotSquare
	}

	n := m.rows
	// Working copy
	u := toFloat64Matrix(m)
	l := Zeros[float64](n, n)

	// Permutation vector: perm[i] tracks which original row is now at position i
	perm := make([]int, n)
	for i := 0; i < n; i++ {
		perm[i] = i
	}

	// Initialize L diagonal to 1
	for i := 0; i < n; i++ {
		l.data[i][i] = 1.0
	}

	for col := 0; col < n; col++ {
		// Partial pivoting
		maxVal := math.Abs(u.data[col][col])
		maxRow := col
		for i := col + 1; i < n; i++ {
			if v := math.Abs(u.data[i][col]); v > maxVal {
				maxVal = v
				maxRow = i
			}
		}

		if maxVal < Epsilon {
			return nil, nil, nil, ErrSingular
		}

		// Swap rows in U, L (already-computed part), and permutation
		if maxRow != col {
			u.data[col], u.data[maxRow] = u.data[maxRow], u.data[col]
			perm[col], perm[maxRow] = perm[maxRow], perm[col]
			// Swap the L entries we've already computed (columns 0..col-1)
			for j := 0; j < col; j++ {
				l.data[col][j], l.data[maxRow][j] = l.data[maxRow][j], l.data[col][j]
			}
		}

		// Eliminate below pivot and record multipliers in L
		pivot := u.data[col][col]
		for i := col + 1; i < n; i++ {
			factor := u.data[i][col] / pivot
			l.data[i][col] = factor
			u.data[i][col] = 0
			for j := col + 1; j < n; j++ {
				u.data[i][j] -= factor * u.data[col][j]
			}
		}
	}

	return l, u, perm, nil
}

// QR performs QR decomposition using the Modified Gram-Schmidt process.
//
// Decomposes the matrix A into an orthogonal matrix Q and an upper
// triangular matrix R such that A = QR.
//
// Q has orthonormal columns: QᵀQ = I
// R is upper triangular
//
// Uses:
//   - Solving least-squares problems
//   - Eigenvalue computation (QR algorithm)
//   - Numerical stability improvements
//
// The modified Gram-Schmidt process is more numerically stable than the
// classical version.
//
// Returns (Q, R, error). Works on any m×n matrix where m ≥ n.
func (m *Matrix[T]) QR() (*Matrix[float64], *Matrix[float64], error) {
	rows := m.rows
	cols := m.cols

	if rows < cols {
		return nil, nil, ErrDimensionMismatch
	}

	a := toFloat64Matrix(m)
	q := Zeros[float64](rows, cols)
	r := Zeros[float64](cols, cols)

	// Copy columns of A into Q for in-place orthogonalization
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			q.data[i][j] = a.data[i][j]
		}
	}

	// Modified Gram-Schmidt
	for j := 0; j < cols; j++ {
		// Compute the norm of column j
		norm := 0.0
		for i := 0; i < rows; i++ {
			norm += q.data[i][j] * q.data[i][j]
		}
		norm = math.Sqrt(norm)

		if norm < Epsilon {
			// Column is linearly dependent — set R[j][j] = 0, column stays zero
			r.data[j][j] = 0
			continue
		}

		r.data[j][j] = norm

		// Normalize column j
		for i := 0; i < rows; i++ {
			q.data[i][j] /= norm
		}

		// Orthogonalize remaining columns against column j
		for k := j + 1; k < cols; k++ {
			// Compute dot product of q_j and q_k
			dot := 0.0
			for i := 0; i < rows; i++ {
				dot += q.data[i][j] * q.data[i][k]
			}
			r.data[j][k] = dot

			// Subtract projection
			for i := 0; i < rows; i++ {
				q.data[i][k] -= dot * q.data[i][j]
			}
		}
	}

	return q, r, nil
}

// Eigen computes the eigenvalues of a square matrix using the QR algorithm
// with implicit shifts.
//
// Eigenvalues λ satisfy: Av = λv for some non-zero vector v.
//
// The QR algorithm iteratively decomposes A = QR, then forms A' = RQ.
// This converges to an upper triangular (Schur) form where the diagonal
// entries are the eigenvalues.
//
// For real matrices, complex eigenvalues appear as 2×2 blocks on the diagonal.
//
// Returns eigenvalues as complex128 (real eigenvalues have zero imaginary part).
// Returns ErrNotSquare if the matrix isn't square.
// Returns ErrNotConverged if the algorithm fails to converge.
//
// Note: eigenvector computation is planned for v1.5.
func (m *Matrix[T]) Eigen() ([]complex128, error) {
	if m.rows != m.cols {
		return nil, ErrNotSquare
	}

	n := m.rows
	if n == 0 {
		return nil, ErrEmptyMatrix
	}

	// Special case: 1×1
	if n == 1 {
		return []complex128{complex(toFloat64(m.data[0][0]), 0)}, nil
	}

	// Special case: 2×2 — use the quadratic formula directly
	if n == 2 {
		return eigen2x2(m), nil
	}

	// General case: QR algorithm
	a := toFloat64Matrix(m)
	maxIter := 1000

	converged := false
	for iter := 0; iter < maxIter; iter++ {
		// Check for convergence: sub-diagonal elements near zero
		allSmall := true
		for i := 0; i < n-1; i++ {
			if math.Abs(a.data[i+1][i]) > Epsilon {
				allSmall = false
				break
			}
		}
		if allSmall {
			converged = true
			break
		}

		// Wilkinson shift: use eigenvalue of bottom-right 2×2 block
		// closest to a[n-1][n-1] as shift
		shift := wilkinsonShift(a)

		// Shift: A - σI
		for i := 0; i < n; i++ {
			a.data[i][i] -= shift
		}

		// QR decomposition
		q, r, err := a.QR()
		if err != nil {
			// Fallback: no shift
			for i := 0; i < n; i++ {
				a.data[i][i] += shift
			}
			continue
		}

		// A = RQ + σI
		rq, err := Mul(r, q)
		if err != nil {
			for i := 0; i < n; i++ {
				a.data[i][i] += shift
			}
			continue
		}

		// Restore shift
		for i := 0; i < n; i++ {
			rq.data[i][i] += shift
		}
		a = rq
	}

	if !converged {
		return nil, ErrNotConverged
	}

	// Extract eigenvalues from the (quasi-)upper triangular form
	eigenvalues := make([]complex128, 0, n)
	i := 0
	for i < n {
		if i == n-1 || math.Abs(a.data[i+1][i]) < Epsilon {
			// Real eigenvalue on the diagonal
			eigenvalues = append(eigenvalues, complex(a.data[i][i], 0))
			i++
		} else {
			// 2×2 block — extract complex conjugate pair
			aa := a.data[i][i]
			bb := a.data[i][i+1]
			cc := a.data[i+1][i]
			dd := a.data[i+1][i+1]
			tr := aa + dd
			det := aa*dd - bb*cc
			disc := tr*tr - 4*det
			if disc < 0 {
				realPart := tr / 2
				imagPart := math.Sqrt(-disc) / 2
				eigenvalues = append(eigenvalues,
					complex(realPart, imagPart),
					complex(realPart, -imagPart),
				)
			} else {
				sqrtDisc := math.Sqrt(disc)
				eigenvalues = append(eigenvalues,
					complex((tr+sqrtDisc)/2, 0),
					complex((tr-sqrtDisc)/2, 0),
				)
			}
			i += 2
		}
	}

	return eigenvalues, nil
}

// eigen2x2 computes eigenvalues of a 2×2 matrix using the quadratic formula.
//
// For a 2×2 matrix [[a, b], [c, d]], the characteristic equation is:
//
//	λ² - (a+d)λ + (ad-bc) = 0
//
// Solutions: λ = ((a+d) ± √((a+d)² - 4(ad-bc))) / 2
func eigen2x2[T Numeric](m *Matrix[T]) []complex128 {
	a := toFloat64(m.data[0][0])
	b := toFloat64(m.data[0][1])
	c := toFloat64(m.data[1][0])
	d := toFloat64(m.data[1][1])

	trace := a + d
	det := a*d - b*c
	disc := trace*trace - 4*det

	if disc >= 0 {
		sqrtDisc := math.Sqrt(disc)
		return []complex128{
			complex((trace+sqrtDisc)/2, 0),
			complex((trace-sqrtDisc)/2, 0),
		}
	}

	// Complex eigenvalues
	realPart := trace / 2
	imagPart := math.Sqrt(-disc) / 2
	return []complex128{
		complex(realPart, imagPart),
		complex(realPart, -imagPart),
	}
}

// wilkinsonShift computes the Wilkinson shift for the QR algorithm.
// Uses the eigenvalue of the bottom-right 2×2 block closest to a[n-1][n-1].
func wilkinsonShift(a *Matrix[float64]) float64 {
	n := a.rows
	if n < 2 {
		return 0
	}

	// Bottom-right 2×2 block
	am := a.data[n-2][n-2]
	bm := a.data[n-2][n-1]
	cm := a.data[n-1][n-2]
	dm := a.data[n-1][n-1]

	tr := am + dm
	det := am*dm - bm*cm
	disc := tr*tr - 4*det

	if disc < 0 {
		// Complex eigenvalues — use dm as shift
		return dm
	}

	sqrtDisc := math.Sqrt(disc)
	lambda1 := (tr + sqrtDisc) / 2
	lambda2 := (tr - sqrtDisc) / 2

	// Pick the eigenvalue closest to dm
	if math.Abs(lambda1-dm) < math.Abs(lambda2-dm) {
		return lambda1
	}
	return lambda2
}

// SVD performs Singular Value Decomposition (thin SVD).
//
// Decomposes A = U * Σ * Vᵀ where:
//   - U is an m×k orthogonal matrix (left singular vectors)
//   - Σ is a k×k diagonal matrix (singular values, non-negative, descending)
//   - V is an n×k orthogonal matrix (right singular vectors)
//   - k = min(m, n)
//
// Algorithm: One-sided Jacobi SVD. Iteratively applies Jacobi rotations
// to orthogonalize the columns of A, producing V and the singular values
// as column norms, then normalizes to get U.
//
// Uses:
//   - Principal Component Analysis (PCA)
//   - Low-rank approximation
//   - Pseudoinverse computation
//   - Condition number estimation
//
// Returns (U, Σ, V, error). Returns ErrEmptyMatrix for empty inputs.
func (m *Matrix[T]) SVD() (*Matrix[float64], *Matrix[float64], *Matrix[float64], error) {
	if m.rows == 0 || m.cols == 0 {
		return nil, nil, nil, ErrEmptyMatrix
	}

	a := toFloat64Matrix(m)
	rows, cols := a.rows, a.cols

	// For m < n: compute SVD of Aᵀ, then swap U and V
	if rows < cols {
		at := Transpose(a)
		uT, sT, vT, err := (&Matrix[float64]{data: at.data, rows: at.rows, cols: at.cols}).SVD()
		if err != nil {
			return nil, nil, nil, err
		}
		return vT, sT, uT, nil
	}

	// One-sided Jacobi SVD for m >= n
	// W = A (working copy, columns will become orthogonal)
	// V accumulates the right rotations
	k := cols
	W := a.Copy()
	V := Identity[float64](k)

	maxSweeps := 100
	for sweep := 0; sweep < maxSweeps; sweep++ {
		// Check convergence: off-diagonal of WᵀW ≈ 0
		converged := true
		for p := 0; p < k-1; p++ {
			for q := p + 1; q < k; q++ {
				// Compute WᵀW[p][q] = dot(col_p, col_q)
				dot := 0.0
				for i := 0; i < rows; i++ {
					dot += W.data[i][p] * W.data[i][q]
				}
				// Compute norms² of columns p and q
				normp2 := 0.0
				normq2 := 0.0
				for i := 0; i < rows; i++ {
					normp2 += W.data[i][p] * W.data[i][p]
					normq2 += W.data[i][q] * W.data[i][q]
				}
				// Check if off-diagonal is negligible
				if math.Abs(dot) <= Epsilon*math.Sqrt(normp2*normq2) {
					continue
				}
				converged = false

				// Compute Jacobi rotation angle to zero WᵀW[p][q]
				// The 2x2 symmetric subproblem: [[normp2, dot], [dot, normq2]]
				tau := (normq2 - normp2) / (2 * dot)
				var t float64
				if tau >= 0 {
					t = 1.0 / (tau + math.Sqrt(1+tau*tau))
				} else {
					t = -1.0 / (-tau + math.Sqrt(1+tau*tau))
				}
				c := 1.0 / math.Sqrt(1+t*t)
				s := t * c

				// Apply rotation to columns p, q of W
				for i := 0; i < rows; i++ {
					wp := W.data[i][p]
					wq := W.data[i][q]
					W.data[i][p] = c*wp - s*wq
					W.data[i][q] = s*wp + c*wq
				}

				// Accumulate V: apply rotation to columns p, q of V
				for i := 0; i < k; i++ {
					vp := V.data[i][p]
					vq := V.data[i][q]
					V.data[i][p] = c*vp - s*vq
					V.data[i][q] = s*vp + c*vq
				}
			}
		}
		if converged {
			break
		}
	}

	// Extract singular values (column norms of W) and normalize to get U
	sigma := Zeros[float64](k, k)
	U := Zeros[float64](rows, k)
	for j := 0; j < k; j++ {
		norm := 0.0
		for i := 0; i < rows; i++ {
			norm += W.data[i][j] * W.data[i][j]
		}
		norm = math.Sqrt(norm)
		sigma.data[j][j] = norm

		if norm > Epsilon {
			for i := 0; i < rows; i++ {
				U.data[i][j] = W.data[i][j] / norm
			}
		}
	}

	// Sort singular values in descending order
	for i := 0; i < k-1; i++ {
		maxIdx := i
		maxVal := sigma.data[i][i]
		for j := i + 1; j < k; j++ {
			if sigma.data[j][j] > maxVal {
				maxVal = sigma.data[j][j]
				maxIdx = j
			}
		}
		if maxIdx != i {
			sigma.data[i][i], sigma.data[maxIdx][maxIdx] = sigma.data[maxIdx][maxIdx], sigma.data[i][i]
			for r := 0; r < rows; r++ {
				U.data[r][i], U.data[r][maxIdx] = U.data[r][maxIdx], U.data[r][i]
			}
			for r := 0; r < k; r++ {
				V.data[r][i], V.data[r][maxIdx] = V.data[r][maxIdx], V.data[r][i]
			}
		}
	}

	return U, sigma, V, nil
}
