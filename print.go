package matrix

import (
	"fmt"
	"strings"
)

// PrintOptions controls how a matrix is formatted when using PrintWith.
type PrintOptions struct {
	// Precision is the number of decimal places for floating-point values.
	// Default: 4. Ignored for integer types.
	Precision int

	// Padding is the minimum number of spaces between columns.
	// Default: 2.
	Padding int

	// Brackets controls the bracket style for the matrix.
	// Options: "square" (default), "round", "pipe", "none".
	Brackets string
}

// defaultPrintOptions returns sensible defaults for printing.
func defaultPrintOptions() PrintOptions {
	return PrintOptions{
		Precision: 4,
		Padding:   2,
		Brackets:  "square",
	}
}

// bracketChars returns the left and right bracket characters for a style.
func bracketChars(style string) (string, string) {
	switch style {
	case "round":
		return "(", ")"
	case "pipe":
		return "|", "|"
	case "none":
		return " ", " "
	default: // "square"
		return "[", "]"
	}
}

// String implements the fmt.Stringer interface, providing a clean
// default representation of the matrix.
//
// Example output for a 2×3 matrix:
//
//	[ 1.0000   2.0000   3.0000 ]
//	[ 4.0000   5.0000   6.0000 ]
func (m *Matrix[T]) String() string {
	opts := defaultPrintOptions()
	return formatMatrix(m, opts)
}

// Print prints the matrix to stdout with default formatting.
func (m *Matrix[T]) Print() {
	fmt.Println(m.String())
}

// PrintWith prints the matrix to stdout with custom formatting options.
func (m *Matrix[T]) PrintWith(opts PrintOptions) {
	if opts.Precision <= 0 {
		opts.Precision = 4
	}
	if opts.Padding <= 0 {
		opts.Padding = 2
	}
	if opts.Brackets == "" {
		opts.Brackets = "square"
	}
	fmt.Println(formatMatrix(m, opts))
}

// formatMatrix builds the string representation of a matrix.
func formatMatrix[T Numeric](m *Matrix[T], opts PrintOptions) string {
	if m.rows == 0 || m.cols == 0 {
		return "(empty matrix)"
	}

	// Format all cells first to determine column widths
	cells := make([][]string, m.rows)
	colWidths := make([]int, m.cols)

	for i := 0; i < m.rows; i++ {
		cells[i] = make([]string, m.cols)
		for j := 0; j < m.cols; j++ {
			cells[i][j] = formatElement(m.data[i][j], opts.Precision)
			if len(cells[i][j]) > colWidths[j] {
				colWidths[j] = len(cells[i][j])
			}
		}
	}

	// Build the output with alignment
	left, right := bracketChars(opts.Brackets)
	padding := strings.Repeat(" ", opts.Padding)
	var sb strings.Builder

	for i := 0; i < m.rows; i++ {
		sb.WriteString(left)
		sb.WriteString(" ")
		for j := 0; j < m.cols; j++ {
			// Right-align each cell within its column width
			cell := cells[i][j]
			spaces := colWidths[j] - len(cell)
			sb.WriteString(strings.Repeat(" ", spaces))
			sb.WriteString(cell)
			if j < m.cols-1 {
				sb.WriteString(padding)
			}
		}
		sb.WriteString(" ")
		sb.WriteString(right)
		if i < m.rows-1 {
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

// formatElement formats a single matrix element based on its type.
func formatElement[T Numeric](val T, precision int) string {
	switch v := any(val).(type) {
	case float32:
		return fmt.Sprintf("%.*f", precision, v)
	case float64:
		return fmt.Sprintf("%.*f", precision, v)
	case complex64:
		return fmt.Sprintf("%.*f", precision, v)
	case complex128:
		return fmt.Sprintf("%.*f", precision, v)
	default:
		// Integer types — no decimal places needed
		return fmt.Sprintf("%v", v)
	}
}
