package main

import (
	"fmt"
	"github.com/Arceus-7/matrix"
)

func main() {
	fmt.Println("--- Solving a Linear System ---")
	
	// Solve the system of equations:
	//  2x +  y -  z =  8
	// -3x -  y + 2z = -11
	// -2x +  y + 2z = -3

	// Coefficient matrix A
	A := matrix.MustNew([][]float64{
		{ 2,  1, -1},
		{-3, -1,  2},
		{-2,  1,  2},
	})

	// Constant vector b
	b := matrix.MustNew([][]float64{
		{8},
		{-11},
		{-3},
	})

	fmt.Println("Coefficient Matrix A:")
	A.Print()

	fmt.Println("\nConstant Vector b:")
	b.Print()
	
	// Check the determinant to ensure the system has a unique solution
	det, err := A.Det()
	if err != nil {
		fmt.Println("Error calculating determinant:", err)
		return
	}
	
	fmt.Printf("\nDeterminant of A: %.2f\n", det)

	if det != 0 {
		// Solve Ax = b
		x, err := matrix.Solve(A, b)
		if err != nil {
			fmt.Println("Error solving system:", err)
			return
		}

		fmt.Println("\nSolution Vector x:")
		x.Print()
	} else {
		fmt.Println("\nThe system does not have a unique solution (det = 0).")
	}
}
