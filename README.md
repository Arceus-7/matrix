# matrix

A clean, generic, zero-dependency matrix math package for Go.

[![Go Reference](https://pkg.go.dev/badge/github.com/Arceus-7/matrix.svg)](https://pkg.go.dev/github.com/Arceus-7/matrix)
[![Go Report Card](https://goreportcard.com/badge/github.com/Arceus-7/matrix)](https://goreportcard.com/report/github.com/Arceus-7/matrix)
[![codecov](https://codecov.io/gh/Arceus-7/MatrixPackage/graph/badge.svg)](https://codecov.io/gh/Arceus-7/MatrixPackage)
## Features

- **Generic** — works with `int`, `float32`, `float64`, `complex128`, and more
- **Zero dependencies** — pure Go stdlib only
- **Immutable by default** — operations return new matrices, never mutate
- **Numerically stable** — partial pivoting throughout, epsilon-based comparisons
- **Well documented** — every function explains the math, not just the code

## Installation

```bash
go get github.com/Arceus-7/matrix
```

Requires **Go 1.21+** (for generics stability).

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/Arceus-7/matrix"
)

func main() {
    // Create from a 2D slice
    m := matrix.MustNew([][]float64{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    })
    m.Print()
    // [ 1.0000  2.0000  3.0000 ]
    // [ 4.0000  5.0000  6.0000 ]
    // [ 7.0000  8.0000  9.0000 ]

    // Special constructors
    identity := matrix.Identity[float64](3)     // 3×3 identity
    zeros    := matrix.Zeros[float64](3, 4)     // 3×4 zero matrix
    ones     := matrix.Ones[float64](2, 2)      // 2×2 ones matrix
    random   := matrix.Random[float64](3, 3)    // 3×3 random [0,1)

    _ = identity; _ = zeros; _ = ones; _ = random
}
```

## Operations

### Arithmetic

```go
a := matrix.MustNew([][]float64{{1, 2}, {3, 4}})
b := matrix.MustNew([][]float64{{5, 6}, {7, 8}})

sum, _     := matrix.Add(a, b)        // element-wise addition
diff, _    := matrix.Sub(a, b)        // element-wise subtraction
product, _ := matrix.Mul(a, b)        // matrix multiplication
scaled     := matrix.Scale(a, 2.5)    // scalar multiplication
transposed := matrix.Transpose(a)     // transpose
hadamard,_ := matrix.HadamardProduct(a, b) // element-wise multiply

_, _, _, _ = sum, diff, product, scaled
_, _ = transposed, hadamard
```

### Properties

```go
m := matrix.MustNew([][]float64{
    {6, 1, 1},
    {4, -2, 5},
    {2, 8, 7},
})

rows, cols := m.Shape()           // (3, 3)
val, _     := m.At(0, 1)          // 1.0
det, _     := m.Det()             // -306.0
rank       := m.Rank()            // 3
trace, _   := m.Trace()           // 11.0
norm       := m.Norm()            // Frobenius norm
isSquare   := m.IsSquare()        // true
isSym      := m.IsSymmetric()     // false

_, _, _, _, _, _, _, _ = rows, cols, val, det, rank, trace, norm, isSquare
_ = isSym
```

### Transformations

```go
m := matrix.MustNew([][]float64{
    {2, 1, -1},
    {-3, -1, 2},
    {-2, 1, 2},
})

inv, _  := m.Inverse()     // matrix inverse (A⁻¹)
ref, _  := m.REF()         // Row Echelon Form
rref, _ := m.RREF()        // Reduced Row Echelon Form

_, _, _ = inv, ref, rref
```

### Decompositions

```go
m := matrix.MustNew([][]float64{
    {12, -51, 4},
    {6, 167, -68},
    {-4, 24, -41},
})

L, U, _   := m.LU()       // LU decomposition (PA = LU)
Q, R, _   := m.QR()        // QR decomposition (A = QR)
eigs, _   := m.Eigen()     // eigenvalues

_, _, _, _ = L, U, Q, R
_ = eigs
```

### Solving Linear Systems

```go
// Solve Ax = b
// System: 2x + y = 5, x + 3y = 7
// Solution: x = 1.6, y = 1.8
A := matrix.MustNew([][]float64{{2, 1}, {1, 3}})
b := matrix.MustNew([][]float64{{5}, {7}})

x, err := matrix.Solve(A, b)
if err != nil {
    panic(err)
}
x.Print()
// [ 1.6000 ]
// [ 1.8000 ]
```

### Pretty Printing

```go
m := matrix.MustNew([][]float64{{1.5, 2.7}, {3.14, 4.0}})

// Default formatting
fmt.Println(m)

// Custom formatting
m.PrintWith(matrix.PrintOptions{
    Precision: 2,      // 2 decimal places
    Padding:   3,      // 3 spaces between columns
    Brackets:  "round", // ( ) instead of [ ]
})
```

## Generic Type Support

```go
// Integer matrices
intM := matrix.MustNew([][]int{{1, 2}, {3, 4}})

// Float32
f32M := matrix.MustNew([][]float32{{1.5, 2.5}, {3.5, 4.5}})

// Complex numbers
cplxM := matrix.MustNew([][]complex128{{1+2i, 3+4i}, {5+6i, 7+8i}})

// Operations that produce fractional results (RREF, Inverse, LU, etc.)
// always return *Matrix[float64], even for integer inputs
inv, _ := intM.Inverse() // returns *Matrix[float64]

_, _, _ = f32M, cplxM, inv
```

## Error Handling

All operations that can fail return `(result, error)` — never panic:

```go
var (
    matrix.ErrDimensionMismatch  // incompatible matrix dimensions
    matrix.ErrNotSquare          // operation needs a square matrix
    matrix.ErrSingular           // matrix is singular (det ≈ 0)
    matrix.ErrNotInvertible      // matrix cannot be inverted
    matrix.ErrOutOfBounds        // index out of range
    matrix.ErrEmptyMatrix        // zero rows or columns
    matrix.ErrNotVector          // expected n×1 column vector
    matrix.ErrNotImplemented     // planned for future version
)
```

## Numerical Stability

Floating-point comparisons use an epsilon tolerance (default `1e-9`):

```go
// Adjust the global epsilon if needed
matrix.Epsilon = 1e-12
```

All elimination-based operations (REF, RREF, Inverse, LU) use **partial pivoting** — they swap rows to place the largest absolute value on the diagonal, reducing numerical error from catastrophic cancellation.

## API Reference

| Category | Functions / Methods |
|----------|-------------------|
| **Constructors** | `New`, `MustNew`, `Identity`, `Zeros`, `Ones`, `Random` |
| **Accessors** | `Shape`, `Rows`, `Cols`, `At`, `Set`, `Copy`, `Data`, `Equals` |
| **Arithmetic** | `Add`, `Sub`, `Mul`, `Scale`, `Transpose`, `HadamardProduct` |
| **Properties** | `IsSquare`, `IsSymmetric`, `IsIdentity`, `IsZero`, `Trace`, `Norm`, `Det`, `Rank` |
| **Transforms** | `REF`, `RREF`, `Inverse` |
| **Decompositions** | `LU`, `QR`, `Eigen`, `SVD` (planned) |
| **Solve** | `Solve` |
| **Printing** | `String`, `Print`, `PrintWith` |

## Roadmap

### v1.0 
- [x] Core matrix type with generics
- [x] Arithmetic operations
- [x] Determinant, rank, trace, norm
- [x] REF, RREF, inverse
- [x] LU and QR decomposition
- [x] Eigenvalue computation
- [x] Linear system solver
- [x] Pretty printing

## License

MIT — see [LICENSE](LICENSE) for details.
