# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.1.0] — 2026-05-20

### Added

- **SVD** — Singular Value Decomposition via one-sided Jacobi rotations (`SVD()`)
- **LUP** — LU decomposition with permutation matrix (`LUP()` → `P*A = L*U`)
- **ApproxEquals** — epsilon-based matrix comparison for floating-point results
- **Row / Col / SubMatrix** — slice extraction accessors with bounds checking
- **ErrNotConverged** — `Eigen()` now returns an error if the QR algorithm fails to converge instead of silently returning incorrect values
- **Benchmark suite** — `benchmark_test.go` covering `Mul`, `LU`, `QR`, `Solve`, `Det` at 10×10 and 100×100
- `.gitignore` for Go projects

### Changed

- `Equals()` docstring updated to reference the new `ApproxEquals` method
- README expanded with new sections: Accessors, Comparison, Benchmarks, and updated API reference table
- Roadmap updated with v1.1 milestone and v1.2 planned items

### Removed

- `ErrNotImplemented` — no longer needed now that SVD is implemented

## [v0.1.0] — 2026-04-11

### Added

- Core generic `Matrix[T]` type with `Numeric`, `RealNumeric`, `Float` constraints
- Constructors: `New`, `MustNew`, `Identity`, `Zeros`, `Ones`, `Random`
- Accessors: `Shape`, `Rows`, `Cols`, `At`, `Set`, `Copy`, `Data`, `Equals`
- Arithmetic: `Add`, `Sub`, `Mul`, `Scale`, `Transpose`, `HadamardProduct`
- Properties: `IsSquare`, `IsSymmetric`, `IsIdentity`, `IsZero`, `Trace`, `Norm`, `Det`, `Rank`
- Transforms: `REF`, `RREF`, `Inverse`
- Decompositions: `LU`, `QR`, `Eigen`
- Linear solver: `Solve` (LU-based with forward/backward substitution)
- Pretty printing: `String`, `Print`, `PrintWith` with configurable precision, padding, and bracket styles
- Configurable global `Epsilon` for floating-point comparisons
- Partial pivoting in all elimination-based operations
- Comprehensive test suite (~84% coverage)
- CI via GitHub Actions (tests + Codecov)
- Community files: LICENSE (MIT), README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Example programs in `examples/`

[v1.1.0]: https://github.com/Arceus-7/matrix/compare/v0.1.0...v1.1.0
[v0.1.0]: https://github.com/Arceus-7/matrix/releases/tag/v0.1.0
