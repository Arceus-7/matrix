package matrix

import "testing"

// ─── Benchmark Helpers ───────────────────────────────────────────────

// benchMul benchmarks matrix multiplication at size n.
func benchMul(b *testing.B, n int) {
	a := Random[float64](n, n)
	m := Random[float64](n, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Mul(a, m)
	}
}

func BenchmarkMul10(b *testing.B)  { benchMul(b, 10) }
func BenchmarkMul100(b *testing.B) { benchMul(b, 100) }

// benchLU benchmarks LU decomposition at size n.
func benchLU(b *testing.B, n int) {
	m := Random[float64](n, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.LU()
	}
}

func BenchmarkLU10(b *testing.B)  { benchLU(b, 10) }
func BenchmarkLU100(b *testing.B) { benchLU(b, 100) }

// benchQR benchmarks QR decomposition at size n.
func benchQR(b *testing.B, n int) {
	m := Random[float64](n, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.QR()
	}
}

func BenchmarkQR10(b *testing.B)  { benchQR(b, 10) }
func BenchmarkQR100(b *testing.B) { benchQR(b, 100) }

// benchSolve benchmarks solving Ax = b at size n.
func benchSolve(b *testing.B, n int) {
	A := Random[float64](n, n)
	// Make it non-singular by adding n*I (diagonally dominant)
	for i := 0; i < n; i++ {
		v, _ := A.At(i, i)
		A.Set(i, i, v+float64(n))
	}
	bVec := Random[float64](n, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Solve(A, bVec)
	}
}

func BenchmarkSolve10(b *testing.B)  { benchSolve(b, 10) }
func BenchmarkSolve100(b *testing.B) { benchSolve(b, 100) }

// benchDet benchmarks determinant computation at size n.
func benchDet(b *testing.B, n int) {
	m := Random[float64](n, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Det()
	}
}

func BenchmarkDet10(b *testing.B)  { benchDet(b, 10) }
func BenchmarkDet100(b *testing.B) { benchDet(b, 100) }
