# Julia's Extended Sparse Array Interface

The following ArrayInterface functions extend Julia's Base LinearAlgebra interface
to improve the ability to do sparse linear algebra.

## Sparse Indexing

These routines allow for improving sparse iteration and indexing.

```@docs
ArrayInterface.isstructured
ArrayInterface.findstructralnz
ArrayInterface.has_sparsestruct
```

## Matrix Coloring

Many routines require calculating the coloring of a matrix, such as for sparse
differentation. The `matrix_colors` function is the high level function which
returns a color vector `Vector{Int}` with the column colors. This function
is overloaded for many special matrix types with analytical solutions for the
matrix colors.

```@docs
ArrayInterface.fast_matrix_colors
ArrayInterface.matrix_colors
```

### General Matrix Colors

For the general dispatch of `matrix_colors`, see [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl)
for a full graph-based coloring algorithm which extends ArrayInterface.

## Addtional Sparse Array Interface Tools

If dealing with general sparse arrays, consider:

- [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): A general set of tools for extending calculus libraries for sparse optimizations.
- [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl): An extended linear solving library with support for generic sparse arrays.