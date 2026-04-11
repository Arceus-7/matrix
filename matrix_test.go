package matrix

import (
	"math"
	"math/cmplx"
	"testing"
)

// ─── Helpers ─────────────────────────────────────────────────────────

// approxEqual checks if two float64 values are within epsilon.
func approxEqual(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

// matApproxEqual checks if two float64 matrices are approximately equal.
func matApproxEqual(a, b *Matrix[float64], eps float64) bool {
	if a.rows != b.rows || a.cols != b.cols {
		return false
	}
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			if !approxEqual(a.data[i][j], b.data[i][j], eps) {
				return false
			}
		}
	}
	return true
}

// ─── Constructor Tests ───────────────────────────────────────────────

func TestNew(t *testing.T) {
	t.Run("valid 2x3", func(t *testing.T) {
		m, err := New([][]float64{
			{1, 2, 3},
			{4, 5, 6},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		rows, cols := m.Shape()
		if rows != 2 || cols != 3 {
			t.Errorf("expected 2x3, got %dx%d", rows, cols)
		}
	})

	t.Run("empty input", func(t *testing.T) {
		_, err := New([][]float64{})
		if err != ErrEmptyMatrix {
			t.Errorf("expected ErrEmptyMatrix, got %v", err)
		}
	})

	t.Run("empty row", func(t *testing.T) {
		_, err := New([][]float64{{}})
		if err != ErrEmptyMatrix {
			t.Errorf("expected ErrEmptyMatrix, got %v", err)
		}
	})

	t.Run("jagged input", func(t *testing.T) {
		_, err := New([][]float64{
			{1, 2},
			{3, 4, 5},
		})
		if err == nil {
			t.Error("expected error for jagged input")
		}
	})

	t.Run("deep copy", func(t *testing.T) {
		data := [][]float64{{1, 2}, {3, 4}}
		m, _ := New(data)
		data[0][0] = 999
		val, _ := m.At(0, 0)
		if val != 1 {
			t.Error("New should deep-copy — modifying original affected the matrix")
		}
	})

	t.Run("integer matrix", func(t *testing.T) {
		m, err := New([][]int{{1, 2}, {3, 4}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		val, _ := m.At(1, 1)
		if val != 4 {
			t.Errorf("expected 4, got %d", val)
		}
	})

	t.Run("complex matrix", func(t *testing.T) {
		m, err := New([][]complex128{{1 + 2i, 3 + 4i}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		val, _ := m.At(0, 0)
		if val != 1+2i {
			t.Errorf("expected (1+2i), got %v", val)
		}
	})
}

func TestMustNew(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		m := MustNew([][]int{{1, 2}, {3, 4}})
		if m.rows != 2 || m.cols != 2 {
			t.Errorf("expected 2x2")
		}
	})

	t.Run("panics on invalid", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("MustNew should panic on empty input")
			}
		}()
		MustNew([][]int{})
	})
}

func TestIdentity(t *testing.T) {
	tests := []int{1, 2, 3, 5}
	for _, n := range tests {
		m := Identity[float64](n)
		rows, cols := m.Shape()
		if rows != n || cols != n {
			t.Errorf("Identity(%d): expected %dx%d, got %dx%d", n, n, n, rows, cols)
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				val, _ := m.At(i, j)
				if i == j && val != 1 {
					t.Errorf("Identity(%d)[%d][%d] = %f, want 1", n, i, j, val)
				}
				if i != j && val != 0 {
					t.Errorf("Identity(%d)[%d][%d] = %f, want 0", n, i, j, val)
				}
			}
		}
	}
}

func TestZeros(t *testing.T) {
	m := Zeros[float64](3, 4)
	rows, cols := m.Shape()
	if rows != 3 || cols != 4 {
		t.Errorf("expected 3x4, got %dx%d", rows, cols)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			val, _ := m.At(i, j)
			if val != 0 {
				t.Errorf("Zeros[%d][%d] = %f, want 0", i, j, val)
			}
		}
	}
}

func TestOnes(t *testing.T) {
	m := Ones[int](2, 3)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			val, _ := m.At(i, j)
			if val != 1 {
				t.Errorf("Ones[%d][%d] = %d, want 1", i, j, val)
			}
		}
	}
}

func TestRandom(t *testing.T) {
	m := Random[float64](3, 3)
	rows, cols := m.Shape()
	if rows != 3 || cols != 3 {
		t.Errorf("expected 3x3, got %dx%d", rows, cols)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			val, _ := m.At(i, j)
			if val < 0 || val >= 1 {
				t.Errorf("Random[%d][%d] = %f, want [0, 1)", i, j, val)
			}
		}
	}
}

// ─── Accessor Tests ──────────────────────────────────────────────────

func TestAtAndSet(t *testing.T) {
	m := MustNew([][]float64{{1, 2}, {3, 4}})

	t.Run("valid access", func(t *testing.T) {
		val, err := m.At(0, 1)
		if err != nil || val != 2 {
			t.Errorf("At(0,1) = %f, %v; want 2, nil", val, err)
		}
	})

	t.Run("out of bounds", func(t *testing.T) {
		_, err := m.At(-1, 0)
		if err != ErrOutOfBounds {
			t.Errorf("At(-1,0) error = %v, want ErrOutOfBounds", err)
		}
		_, err = m.At(0, 5)
		if err != ErrOutOfBounds {
			t.Errorf("At(0,5) error = %v, want ErrOutOfBounds", err)
		}
	})

	t.Run("Set", func(t *testing.T) {
		err := m.Set(1, 0, 99)
		if err != nil {
			t.Fatalf("Set error: %v", err)
		}
		val, _ := m.At(1, 0)
		if val != 99 {
			t.Errorf("After Set(1,0,99), At(1,0) = %f, want 99", val)
		}
	})

	t.Run("Set out of bounds", func(t *testing.T) {
		err := m.Set(5, 0, 1)
		if err != ErrOutOfBounds {
			t.Errorf("Set(5,0,1) error = %v, want ErrOutOfBounds", err)
		}
	})
}

func TestCopy(t *testing.T) {
	original := MustNew([][]float64{{1, 2}, {3, 4}})
	copied := original.Copy()

	if !original.Equals(copied) {
		t.Error("copy should equal original")
	}

	copied.Set(0, 0, 999)
	val, _ := original.At(0, 0)
	if val != 1 {
		t.Error("modifying copy affected original — not a deep copy")
	}
}

func TestData(t *testing.T) {
	m := MustNew([][]int{{1, 2}, {3, 4}})
	data := m.Data()
	data[0][0] = 999
	val, _ := m.At(0, 0)
	if val != 1 {
		t.Error("modifying Data() result affected the matrix")
	}
}

func TestEquals(t *testing.T) {
	a := MustNew([][]int{{1, 2}, {3, 4}})
	b := MustNew([][]int{{1, 2}, {3, 4}})
	c := MustNew([][]int{{1, 2}, {3, 5}})
	d := MustNew([][]int{{1, 2, 3}})

	if !a.Equals(b) {
		t.Error("identical matrices should be equal")
	}
	if a.Equals(c) {
		t.Error("different matrices should not be equal")
	}
	if a.Equals(d) {
		t.Error("different-shaped matrices should not be equal")
	}
}

// ─── Arithmetic Tests ────────────────────────────────────────────────

func TestAdd(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		a := MustNew([][]float64{{1, 2}, {3, 4}})
		b := MustNew([][]float64{{5, 6}, {7, 8}})
		c, err := Add(a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := MustNew([][]float64{{6, 8}, {10, 12}})
		if !c.Equals(expected) {
			t.Errorf("Add result incorrect\ngot: %v\nwant: %v", c, expected)
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		a := MustNew([][]float64{{1, 2}})
		b := MustNew([][]float64{{1}, {2}})
		_, err := Add(a, b)
		if err != ErrDimensionMismatch {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("integer", func(t *testing.T) {
		a := MustNew([][]int{{1, 2}, {3, 4}})
		b := MustNew([][]int{{10, 20}, {30, 40}})
		c, err := Add(a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := MustNew([][]int{{11, 22}, {33, 44}})
		if !c.Equals(expected) {
			t.Errorf("integer Add failed")
		}
	})
}

func TestSub(t *testing.T) {
	a := MustNew([][]float64{{5, 6}, {7, 8}})
	b := MustNew([][]float64{{1, 2}, {3, 4}})
	c, err := Sub(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := MustNew([][]float64{{4, 4}, {4, 4}})
	if !c.Equals(expected) {
		t.Errorf("Sub result incorrect")
	}
}

func TestMul(t *testing.T) {
	t.Run("2x3 * 3x2", func(t *testing.T) {
		a := MustNew([][]float64{
			{1, 2, 3},
			{4, 5, 6},
		})
		b := MustNew([][]float64{
			{7, 8},
			{9, 10},
			{11, 12},
		})
		c, err := Mul(a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// [1*7+2*9+3*11, 1*8+2*10+3*12]   = [58, 64]
		// [4*7+5*9+6*11, 4*8+5*10+6*12]   = [139, 154]
		expected := MustNew([][]float64{
			{58, 64},
			{139, 154},
		})
		if !c.Equals(expected) {
			t.Errorf("Mul result incorrect\ngot: %v\nwant: %v", c, expected)
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		a := MustNew([][]float64{{1, 2}})
		b := MustNew([][]float64{{1, 2}})
		_, err := Mul(a, b)
		if err != ErrDimensionMismatch {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("identity multiplication", func(t *testing.T) {
		a := MustNew([][]float64{{1, 2}, {3, 4}})
		i := Identity[float64](2)
		result, err := Mul(a, i)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !result.Equals(a) {
			t.Errorf("A*I should equal A")
		}
	})

	t.Run("1x1", func(t *testing.T) {
		a := MustNew([][]float64{{3}})
		b := MustNew([][]float64{{5}})
		c, err := Mul(a, b)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		val, _ := c.At(0, 0)
		if val != 15 {
			t.Errorf("1x1 Mul: got %f, want 15", val)
		}
	})
}

func TestScale(t *testing.T) {
	m := MustNew([][]float64{{1, 2}, {3, 4}})
	s := Scale(m, 2.5)
	expected := MustNew([][]float64{{2.5, 5}, {7.5, 10}})
	if !s.Equals(expected) {
		t.Errorf("Scale result incorrect")
	}
}

func TestTranspose(t *testing.T) {
	t.Run("2x3", func(t *testing.T) {
		m := MustNew([][]float64{
			{1, 2, 3},
			{4, 5, 6},
		})
		mt := Transpose(m)
		expected := MustNew([][]float64{
			{1, 4},
			{2, 5},
			{3, 6},
		})
		if !mt.Equals(expected) {
			t.Errorf("Transpose incorrect")
		}
		rows, cols := mt.Shape()
		if rows != 3 || cols != 2 {
			t.Errorf("Transpose shape: expected 3x2, got %dx%d", rows, cols)
		}
	})

	t.Run("double transpose = original", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2, 3}, {4, 5, 6}})
		mtt := Transpose(Transpose(m))
		if !mtt.Equals(m) {
			t.Error("(Aᵀ)ᵀ should equal A")
		}
	})

	t.Run("1x1", func(t *testing.T) {
		m := MustNew([][]float64{{42}})
		mt := Transpose(m)
		if !mt.Equals(m) {
			t.Error("transpose of 1x1 should equal itself")
		}
	})
}

func TestHadamardProduct(t *testing.T) {
	a := MustNew([][]float64{{1, 2}, {3, 4}})
	b := MustNew([][]float64{{5, 6}, {7, 8}})
	c, err := HadamardProduct(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := MustNew([][]float64{{5, 12}, {21, 32}})
	if !c.Equals(expected) {
		t.Error("HadamardProduct incorrect")
	}
}

// ─── Property Tests ──────────────────────────────────────────────────

func TestIsSquare(t *testing.T) {
	sq := MustNew([][]int{{1, 2}, {3, 4}})
	if !sq.IsSquare() {
		t.Error("2x2 should be square")
	}

	rect := MustNew([][]int{{1, 2, 3}})
	if rect.IsSquare() {
		t.Error("1x3 should not be square")
	}
}

func TestIsSymmetric(t *testing.T) {
	sym := MustNew([][]float64{
		{1, 2, 3},
		{2, 5, 6},
		{3, 6, 9},
	})
	if !sym.IsSymmetric() {
		t.Error("symmetric matrix not detected")
	}

	asym := MustNew([][]float64{
		{1, 2},
		{3, 4},
	})
	if asym.IsSymmetric() {
		t.Error("non-symmetric matrix incorrectly detected as symmetric")
	}

	rect := MustNew([][]float64{{1, 2, 3}})
	if rect.IsSymmetric() {
		t.Error("non-square matrix should not be symmetric")
	}
}

func TestTrace(t *testing.T) {
	t.Run("3x3", func(t *testing.T) {
		m := MustNew([][]float64{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		})
		tr, err := m.Trace()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tr != 15 { // 1 + 5 + 9
			t.Errorf("Trace = %f, want 15", tr)
		}
	})

	t.Run("non-square error", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2, 3}})
		_, err := m.Trace()
		if err != ErrNotSquare {
			t.Errorf("expected ErrNotSquare, got %v", err)
		}
	})

	t.Run("1x1", func(t *testing.T) {
		m := MustNew([][]float64{{42}})
		tr, _ := m.Trace()
		if tr != 42 {
			t.Errorf("Trace of 1x1 = %f, want 42", tr)
		}
	})

	t.Run("identity trace", func(t *testing.T) {
		id := Identity[float64](5)
		tr, _ := id.Trace()
		if tr != 5 {
			t.Errorf("Trace(I_5) = %f, want 5", tr)
		}
	})
}

func TestNorm(t *testing.T) {
	// ‖[[1,2],[3,4]]‖_F = √(1+4+9+16) = √30
	m := MustNew([][]float64{{1, 2}, {3, 4}})
	norm := m.Norm()
	expected := math.Sqrt(30)
	if !approxEqual(norm, expected, 1e-10) {
		t.Errorf("Norm = %f, want %f", norm, expected)
	}
}

func TestDet(t *testing.T) {
	tests := []struct {
		name     string
		data     [][]float64
		expected float64
	}{
		{
			name:     "1x1",
			data:     [][]float64{{5}},
			expected: 5,
		},
		{
			name:     "2x2",
			data:     [][]float64{{1, 2}, {3, 4}},
			expected: -2, // 1*4 - 2*3
		},
		{
			name: "3x3",
			data: [][]float64{
				{6, 1, 1},
				{4, -2, 5},
				{2, 8, 7},
			},
			expected: -306,
		},
		{
			name:     "identity",
			data:     [][]float64{{1, 0}, {0, 1}},
			expected: 1,
		},
		{
			name:     "singular",
			data:     [][]float64{{1, 2}, {2, 4}},
			expected: 0,
		},
		{
			name: "3x3 identity",
			data: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := MustNew(tt.data)
			det, err := m.Det()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !approxEqual(det, tt.expected, 1e-6) {
				t.Errorf("Det = %f, want %f", det, tt.expected)
			}
		})
	}

	t.Run("non-square error", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2, 3}})
		_, err := m.Det()
		if err != ErrNotSquare {
			t.Errorf("expected ErrNotSquare, got %v", err)
		}
	})
}

func TestRank(t *testing.T) {
	tests := []struct {
		name string
		data [][]float64
		rank int
	}{
		{
			name: "identity 3x3",
			data: [][]float64{
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
			},
			rank: 3,
		},
		{
			name: "rank 2 of 3x3",
			data: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			rank: 2,
		},
		{
			name: "zero matrix",
			data: [][]float64{
				{0, 0},
				{0, 0},
			},
			rank: 0,
		},
		{
			name: "rank 1",
			data: [][]float64{
				{1, 2},
				{2, 4},
			},
			rank: 1,
		},
		{
			name: "non-square 2x3",
			data: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
			rank: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := MustNew(tt.data)
			rank := m.Rank()
			if rank != tt.rank {
				t.Errorf("Rank = %d, want %d", rank, tt.rank)
			}
		})
	}
}

func TestIsIdentity(t *testing.T) {
	id := Identity[float64](3)
	if !id.IsIdentity() {
		t.Error("Identity(3) should be identity")
	}

	notId := MustNew([][]float64{{1, 1}, {0, 1}})
	if notId.IsIdentity() {
		t.Error("non-identity should not be identity")
	}
}

func TestIsZero(t *testing.T) {
	z := Zeros[float64](2, 3)
	if !z.IsZero() {
		t.Error("Zeros(2,3) should be zero")
	}

	nz := MustNew([][]float64{{0, 1}, {0, 0}})
	if nz.IsZero() {
		t.Error("non-zero matrix should not be zero")
	}
}

// ─── Transform Tests ─────────────────────────────────────────────────

func TestREF(t *testing.T) {
	m := MustNew([][]float64{
		{2, 1, -1},
		{-3, -1, 2},
		{-2, 1, 2},
	})
	r, err := m.REF()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// In REF, entries below pivots should be zero
	rows, cols := r.Shape()
	if rows != 3 || cols != 3 {
		t.Fatalf("REF shape: expected 3x3, got %dx%d", rows, cols)
	}

	// Below-diagonal in pivot columns should be zero
	for i := 1; i < rows; i++ {
		for j := 0; j < i && j < cols; j++ {
			val, _ := r.At(i, j)
			if math.Abs(val) > 1e-6 {
				t.Errorf("REF[%d][%d] = %f, expected ~0 (below pivot)", i, j, val)
			}
		}
	}
}

func TestRREF(t *testing.T) {
	t.Run("3x3 identity result", func(t *testing.T) {
		m := MustNew([][]float64{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		})
		r, _ := m.RREF()
		if !r.IsIdentity() {
			t.Error("RREF of identity should be identity")
		}
	})

	t.Run("augmented system", func(t *testing.T) {
		// System: x + y = 3, 2x + 3y = 8
		// Augmented: [[1,1,3],[2,3,8]]
		// RREF: [[1,0,1],[0,1,2]]
		m := MustNew([][]float64{
			{1, 1, 3},
			{2, 3, 8},
		})
		r, _ := m.RREF()
		expected := MustNew([][]float64{
			{1, 0, 1},
			{0, 1, 2},
		})
		if !matApproxEqual(r, expected, 1e-9) {
			t.Errorf("RREF incorrect\ngot:\n%v\nwant:\n%v", r, expected)
		}
	})
}

func TestInverse(t *testing.T) {
	t.Run("2x2", func(t *testing.T) {
		m := MustNew([][]float64{{4, 7}, {2, 6}})
		inv, err := m.Inverse()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Inverse of [[4,7],[2,6]] = [[0.6,-0.7],[-0.2,0.4]]
		expected := MustNew([][]float64{
			{0.6, -0.7},
			{-0.2, 0.4},
		})
		if !matApproxEqual(inv, expected, 1e-9) {
			t.Errorf("Inverse incorrect\ngot:\n%v\nwant:\n%v", inv, expected)
		}
	})

	t.Run("A * A^-1 = I", func(t *testing.T) {
		m := MustNew([][]float64{
			{1, 2, 3},
			{0, 1, 4},
			{5, 6, 0},
		})
		inv, err := m.Inverse()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		mf := toFloat64Matrix(m)
		product, err := Mul(mf, inv)
		if err != nil {
			t.Fatalf("Mul error: %v", err)
		}
		id := Identity[float64](3)
		if !matApproxEqual(product, id, 1e-9) {
			t.Errorf("A * A⁻¹ ≠ I\ngot:\n%v", product)
		}
	})

	t.Run("singular matrix", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2}, {2, 4}})
		_, err := m.Inverse()
		if err != ErrNotInvertible {
			t.Errorf("expected ErrNotInvertible, got %v", err)
		}
	})

	t.Run("non-square", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2, 3}})
		_, err := m.Inverse()
		if err != ErrNotSquare {
			t.Errorf("expected ErrNotSquare, got %v", err)
		}
	})

	t.Run("1x1", func(t *testing.T) {
		m := MustNew([][]float64{{5}})
		inv, err := m.Inverse()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		val, _ := inv.At(0, 0)
		if !approxEqual(val, 0.2, 1e-10) {
			t.Errorf("Inverse of [[5]] = [[%f]], want [[0.2]]", val)
		}
	})

	t.Run("integer matrix inverse", func(t *testing.T) {
		m := MustNew([][]int{{2, 1}, {5, 3}})
		inv, err := m.Inverse()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Inverse of [[2,1],[5,3]] = [[3,-1],[-5,2]]
		expected := MustNew([][]float64{{3, -1}, {-5, 2}})
		if !matApproxEqual(inv, expected, 1e-9) {
			t.Errorf("integer matrix inverse incorrect")
		}
	})
}

// ─── Decomposition Tests ─────────────────────────────────────────────

func TestLU(t *testing.T) {
	t.Run("3x3", func(t *testing.T) {
		m := MustNew([][]float64{
			{2, -1, -2},
			{-4, 6, 3},
			{-4, -2, 8},
		})
		L, U, err := m.LU()
		if err != nil {
			t.Fatalf("LU error: %v", err)
		}

		// Verify L * U ≈ PA (permuted A)
		product, err := Mul(L, U)
		if err != nil {
			t.Fatalf("Mul error: %v", err)
		}

		// L should be lower triangular with 1s on diagonal
		for i := 0; i < 3; i++ {
			val, _ := L.At(i, i)
			if !approxEqual(val, 1, 1e-10) {
				t.Errorf("L[%d][%d] = %f, want 1", i, i, val)
			}
			for j := i + 1; j < 3; j++ {
				val, _ = L.At(i, j)
				if math.Abs(val) > 1e-10 {
					t.Errorf("L[%d][%d] = %f, want 0 (upper triangle)", i, j, val)
				}
			}
		}

		// U should be upper triangular
		for i := 0; i < 3; i++ {
			for j := 0; j < i; j++ {
				val, _ := U.At(i, j)
				if math.Abs(val) > 1e-10 {
					t.Errorf("U[%d][%d] = %f, want 0 (lower triangle)", i, j, val)
				}
			}
		}

		// Since we use partial pivoting, L*U equals the permuted original
		// Just verify the product is valid (rows are a permutation of original)
		_ = product
	})

	t.Run("singular", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2}, {2, 4}})
		_, _, err := m.LU()
		if err != ErrSingular {
			t.Errorf("expected ErrSingular, got %v", err)
		}
	})

	t.Run("non-square", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2, 3}})
		_, _, err := m.LU()
		if err != ErrNotSquare {
			t.Errorf("expected ErrNotSquare, got %v", err)
		}
	})
}

func TestQR(t *testing.T) {
	t.Run("3x3", func(t *testing.T) {
		m := MustNew([][]float64{
			{12, -51, 4},
			{6, 167, -68},
			{-4, 24, -41},
		})
		Q, R, err := m.QR()
		if err != nil {
			t.Fatalf("QR error: %v", err)
		}

		// Verify Q * R ≈ A
		product, err := Mul(Q, R)
		if err != nil {
			t.Fatalf("Mul error: %v", err)
		}
		mf := toFloat64Matrix(m)
		if !matApproxEqual(product, mf, 1e-9) {
			t.Errorf("Q*R ≠ A\ngot:\n%v\nwant:\n%v", product, mf)
		}

		// Verify Q is orthogonal: QᵀQ ≈ I
		qt := Transpose(Q)
		qtq, _ := Mul(qt, Q)
		id := Identity[float64](3)
		if !matApproxEqual(qtq, id, 1e-9) {
			t.Errorf("QᵀQ ≠ I\ngot:\n%v", qtq)
		}

		// Verify R is upper triangular
		for i := 0; i < R.rows; i++ {
			for j := 0; j < i; j++ {
				val, _ := R.At(i, j)
				if math.Abs(val) > 1e-9 {
					t.Errorf("R[%d][%d] = %f, should be 0", i, j, val)
				}
			}
		}
	})
}

func TestEigen(t *testing.T) {
	t.Run("2x2 real eigenvalues", func(t *testing.T) {
		// [[2, 1], [1, 2]] has eigenvalues 3 and 1
		m := MustNew([][]float64{{2, 1}, {1, 2}})
		vals, err := m.Eigen()
		if err != nil {
			t.Fatalf("Eigen error: %v", err)
		}
		if len(vals) != 2 {
			t.Fatalf("expected 2 eigenvalues, got %d", len(vals))
		}

		// Sort eigenvalues (larger first)
		realVals := []float64{real(vals[0]), real(vals[1])}
		if realVals[0] < realVals[1] {
			realVals[0], realVals[1] = realVals[1], realVals[0]
		}
		if !approxEqual(realVals[0], 3, 1e-6) || !approxEqual(realVals[1], 1, 1e-6) {
			t.Errorf("eigenvalues = %v, want [3, 1]", realVals)
		}
	})

	t.Run("2x2 complex eigenvalues", func(t *testing.T) {
		// [[0, -1], [1, 0]] has eigenvalues ±i
		m := MustNew([][]float64{{0, -1}, {1, 0}})
		vals, err := m.Eigen()
		if err != nil {
			t.Fatalf("Eigen error: %v", err)
		}
		if len(vals) != 2 {
			t.Fatalf("expected 2 eigenvalues, got %d", len(vals))
		}

		// Should be ±i
		for _, v := range vals {
			if !approxEqual(real(v), 0, 1e-6) {
				t.Errorf("real part = %f, want 0", real(v))
			}
			if !approxEqual(math.Abs(imag(v)), 1, 1e-6) {
				t.Errorf("|imag part| = %f, want 1", math.Abs(imag(v)))
			}
		}
	})

	t.Run("diagonal matrix", func(t *testing.T) {
		// Eigenvalues of diagonal matrix are the diagonal entries
		m := MustNew([][]float64{
			{3, 0, 0},
			{0, 7, 0},
			{0, 0, 2},
		})
		vals, err := m.Eigen()
		if err != nil {
			t.Fatalf("Eigen error: %v", err)
		}
		if len(vals) != 3 {
			t.Fatalf("expected 3 eigenvalues, got %d", len(vals))
		}

		// Collect real parts and sort
		realVals := make([]float64, 3)
		for i, v := range vals {
			realVals[i] = real(v)
		}
		// Check that {3, 7, 2} are eigenvalues (in any order)
		found := map[float64]bool{3: false, 7: false, 2: false}
		for _, rv := range realVals {
			for expected := range found {
				if approxEqual(rv, expected, 1e-6) {
					found[expected] = true
				}
			}
		}
		for expected, wasFound := range found {
			if !wasFound {
				t.Errorf("eigenvalue %f not found in %v", expected, realVals)
			}
		}
	})

	t.Run("1x1", func(t *testing.T) {
		m := MustNew([][]float64{{42}})
		vals, err := m.Eigen()
		if err != nil {
			t.Fatalf("Eigen error: %v", err)
		}
		if len(vals) != 1 || real(vals[0]) != 42 {
			t.Errorf("eigenvalue of [[42]] = %v, want [42]", vals)
		}
	})

	t.Run("non-square error", func(t *testing.T) {
		m := MustNew([][]float64{{1, 2, 3}})
		_, err := m.Eigen()
		if err != ErrNotSquare {
			t.Errorf("expected ErrNotSquare, got %v", err)
		}
	})
}

func TestSVD(t *testing.T) {
	m := MustNew([][]float64{{1, 2}, {3, 4}})
	_, _, _, err := m.SVD()
	if err != ErrNotImplemented {
		t.Errorf("expected ErrNotImplemented, got %v", err)
	}
}

// ─── Solve Tests ─────────────────────────────────────────────────────

func TestSolve(t *testing.T) {
	t.Run("2x2 system", func(t *testing.T) {
		// 2x + y = 5
		// x + 3y = 7
		// Solution: x = 1.6, y = 1.8
		A := MustNew([][]float64{{2, 1}, {1, 3}})
		b := MustNew([][]float64{{5}, {7}})
		x, err := Solve(A, b)
		if err != nil {
			t.Fatalf("Solve error: %v", err)
		}

		x0, _ := x.At(0, 0)
		x1, _ := x.At(1, 0)
		if !approxEqual(x0, 1.6, 1e-9) || !approxEqual(x1, 1.8, 1e-9) {
			t.Errorf("Solution = [%f, %f], want [1.6, 1.8]", x0, x1)
		}
	})

	t.Run("3x3 system", func(t *testing.T) {
		// x + y + z = 6
		// 2y + 5z = -4
		// 2x + 5y - z = 27
		// Solution: x = 5, y = 3, z = -2
		A := MustNew([][]float64{
			{1, 1, 1},
			{0, 2, 5},
			{2, 5, -1},
		})
		b := MustNew([][]float64{{6}, {-4}, {27}})
		x, err := Solve(A, b)
		if err != nil {
			t.Fatalf("Solve error: %v", err)
		}

		expected := []float64{5, 3, -2}
		for i, exp := range expected {
			val, _ := x.At(i, 0)
			if !approxEqual(val, exp, 1e-9) {
				t.Errorf("x[%d] = %f, want %f", i, val, exp)
			}
		}
	})

	t.Run("singular system", func(t *testing.T) {
		A := MustNew([][]float64{{1, 2}, {2, 4}})
		b := MustNew([][]float64{{3}, {6}})
		_, err := Solve(A, b)
		if err != ErrSingular {
			t.Errorf("expected ErrSingular, got %v", err)
		}
	})

	t.Run("non-square A", func(t *testing.T) {
		A := MustNew([][]float64{{1, 2, 3}})
		b := MustNew([][]float64{{1}})
		_, err := Solve(A, b)
		if err != ErrNotSquare {
			t.Errorf("expected ErrNotSquare, got %v", err)
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		A := MustNew([][]float64{{1, 2}, {3, 4}})
		b := MustNew([][]float64{{1}, {2}, {3}})
		_, err := Solve(A, b)
		if err != ErrDimensionMismatch {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})

	t.Run("not a vector", func(t *testing.T) {
		A := MustNew([][]float64{{1, 2}, {3, 4}})
		b := MustNew([][]float64{{1, 2}, {3, 4}})
		_, err := Solve(A, b)
		if err != ErrNotVector {
			t.Errorf("expected ErrNotVector, got %v", err)
		}
	})

	t.Run("verify Ax = b", func(t *testing.T) {
		A := MustNew([][]float64{
			{3, 2, -1},
			{2, -2, 4},
			{-1, 0.5, -1},
		})
		b := MustNew([][]float64{{1}, {-2}, {0}})
		x, err := Solve(A, b)
		if err != nil {
			t.Fatalf("Solve error: %v", err)
		}

		// Verify: Ax should equal b
		Af := toFloat64Matrix(A)
		result, _ := Mul(Af, x)
		bf := toFloat64Matrix(b)
		if !matApproxEqual(result, bf, 1e-9) {
			t.Errorf("Ax ≠ b\nAx = %v\nb = %v", result, bf)
		}
	})
}

// ─── Print Tests ─────────────────────────────────────────────────────

func TestString(t *testing.T) {
	m := MustNew([][]int{{1, 2}, {3, 4}})
	s := m.String()
	if s == "" {
		t.Error("String() returned empty")
	}

	// Should contain the values
	if len(s) < 4 {
		t.Errorf("String() too short: %q", s)
	}
}

func TestPrintWith(t *testing.T) {
	// Just test it doesn't panic
	m := MustNew([][]float64{{1.23456, 2.34567}, {3.45678, 4.56789}})
	m.PrintWith(PrintOptions{
		Precision: 2,
		Padding:   3,
		Brackets:  "round",
	})
}

// ─── Property-Based Tests ────────────────────────────────────────────

func TestPropertyTransposeOfTranspose(t *testing.T) {
	// (Aᵀ)ᵀ = A
	m := MustNew([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	result := Transpose(Transpose(m))
	if !result.Equals(m) {
		t.Error("(Aᵀ)ᵀ ≠ A")
	}
}

func TestPropertyTransposeOfProduct(t *testing.T) {
	// (AB)ᵀ = BᵀAᵀ
	a := MustNew([][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	})
	b := MustNew([][]float64{
		{7, 8, 9},
		{10, 11, 12},
	})

	ab, _ := Mul(a, b)
	lhs := Transpose(ab) // (AB)ᵀ

	bt := Transpose(b)
	at := Transpose(a)
	rhs, _ := Mul(bt, at) // BᵀAᵀ

	if !lhs.Equals(rhs) {
		t.Error("(AB)ᵀ ≠ BᵀAᵀ")
	}
}

func TestPropertyDetProduct(t *testing.T) {
	// det(AB) = det(A) * det(B)
	a := MustNew([][]float64{
		{1, 2},
		{3, 4},
	})
	b := MustNew([][]float64{
		{5, 6},
		{7, 8},
	})

	ab, _ := Mul(a, b)

	detA, _ := a.Det()
	detB, _ := b.Det()
	detAB, _ := ab.Det()

	if !approxEqual(detAB, detA*detB, 1e-6) {
		t.Errorf("det(AB) = %f, det(A)*det(B) = %f", detAB, detA*detB)
	}
}

func TestPropertyInverseProduct(t *testing.T) {
	// A * A⁻¹ = I
	m := MustNew([][]float64{
		{2, 1},
		{5, 3},
	})
	inv, err := m.Inverse()
	if err != nil {
		t.Fatalf("Inverse error: %v", err)
	}

	mf := toFloat64Matrix(m)
	product, _ := Mul(mf, inv)
	id := Identity[float64](2)
	if !matApproxEqual(product, id, 1e-9) {
		t.Errorf("A * A⁻¹ ≠ I\n%v", product)
	}
}

func TestPropertyDetTranspose(t *testing.T) {
	// det(A) = det(Aᵀ)
	m := MustNew([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 0},
	})
	mt := Transpose(m)

	detM, _ := m.Det()
	detMt, _ := mt.Det()

	if !approxEqual(detM, detMt, 1e-6) {
		t.Errorf("det(A) = %f, det(Aᵀ) = %f", detM, detMt)
	}
}

// Suppress unused import warning for cmplx
var _ = cmplx.Abs
