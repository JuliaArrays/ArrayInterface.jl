# Julia's Extended Linear Algebra Interface

The following ArrayInterface functions extend Julia's Base LinearAlgebra interface
to improve the ability to do generic linear algebra.

## Generic Matrix Constructors

These functions allow for the construction of matrices from array information in a generic
way. It handles cases like how if `x` is a vector on the GPU, its associated matrix type
should also be GPU-based, and thus appropriately placed with zeros/undef values.

```@docs
ArrayInterface.zeromatrix
ArrayInterface.undefmatrix
```

## Generic Matrix Functions

These query allow for easily learning properties of a general matrix.

```@docs
ArrayInterface.issingular
```

## Factorization Instance Functions 

These functions allow for generating the instance of a factorization's result without
running the full factorization. This thus allows for building types to hold the factorization
without having to perform expensive extra computations.

```@docs
ArrayInterface.bunchkaufman_instance
ArrayInterface.cholesky_instance
ArrayInterface.ldlt_instance
ArrayInterface.lu_instance
ArrayInterface.qr_instance
ArrayInterface.svd_instance
```

## Addtional Linear Algebra Interface Tools

If dealing with general linear algebra, consider:

- [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl): An extended linear solving library with support for generic arrays.