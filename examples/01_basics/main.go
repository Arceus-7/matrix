package main

import (
	"fmt"
	"github.com/Arceus-7/matrix"
)

func main() {
	fmt.Println("--- Matrix Basics ---")
	
	// Create two 2x2 matrices using floats
	a := matrix.MustNew([][]float64{
		{1, 2},
		{3, 4},
	})
	
	b := matrix.MustNew([][]float64{
		{5, 6},
		{7, 8},
	})

	fmt.Println("Matrix A:")
	a.Print()
	
	fmt.Println("\nMatrix B:")
	b.Print()

	// Addition
	c, _ := matrix.Add(a, b)
	fmt.Println("\nA + B:")
	c.Print()

	// Multiplication
	d, _ := matrix.Mul(a, b)
	fmt.Println("\nA * B:")
	d.Print()

	// Transpose
	e := matrix.Transpose(a)
	fmt.Println("\nTranspose of A:")
	e.Print()
	
	// Error handling demonstration
	// Operations that can fail return an error, never panic
	_, err := matrix.Add(a, matrix.Zeros[float64](3, 3))
	if err != nil {
		fmt.Printf("\nError expected when adding matrices of different shapes:\n%v\n", err)
	}
}
