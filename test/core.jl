using ArrayInterface
using ArrayInterface: zeromatrix, undefmatrix
import ArrayInterface: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance,
        parent_type, zeromatrix
using LinearAlgebra
using Random
using SparseArrays
using Test

# ensure we are correctly parsing these
ArrayInterface.@assume_effects :total foo(x::Bool) = x
ArrayInterface.@assume_effects bar(x::Bool) = x
@test foo(true)
@test bar(true)

@testset "zeromatrix and unsafematrix" begin
    for T in (Int, Float32, Float64)
        for (vectype, mattype) in ((Vector{T}, Matrix{T}), (SparseVector{T}, SparseMatrixCSC{T, Int}))
            v = vectype(rand(T, 4))
            um = undefmatrix(v)
            @test size(um) == (length(v),length(v))
            @test typeof(um) == mattype
            @test zeromatrix(v) == zeros(T,length(v),length(v))
        end
        v = rand(T,4,4,4)
        um = undefmatrix(v)
        @test size(um) == (length(v),length(v))
        @test typeof(um) == Matrix{T}
        @test zeromatrix(v) == zeros(T,4*4*4,4*4*4)
        @test zeromatrix(rand(T)) == zero(T)
        @test undefmatrix(rand(T)) isa T
    end
end

@testset "matrix colors" begin
    @test ArrayInterface.fast_matrix_colors(1) == false
    @test ArrayInterface.fast_matrix_colors(Diagonal{Int,Vector{Int}})

    @test ArrayInterface.matrix_colors(Diagonal([1,2,3,4])) == [1, 1, 1, 1]
    @test ArrayInterface.matrix_colors(Bidiagonal([1,2,3,4], [7,8,9], :U)) == [1, 2, 1, 2]
    @test ArrayInterface.matrix_colors(Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])) == [1, 2, 3, 1]
    @test ArrayInterface.matrix_colors(SymTridiagonal([1,2,3,4],[5,6,7])) == [1, 2, 3, 1]
    @test ArrayInterface.matrix_colors(rand(4,4)) == Base.OneTo(4)
end

@testset "parent_type" begin
    x = ones(4, 4)
    @test parent_type(view(x, 1:2, 1:2)) <: typeof(x)
    @test parent_type(reshape(x, 2, :)) <: typeof(x)
    @test parent_type(transpose(x)) <: typeof(x)
    @test parent_type(Symmetric(x)) <: typeof(x)
    @test parent_type(UpperTriangular(x)) <: typeof(x)
    @test parent_type(PermutedDimsArray(x, (2,1))) <: typeof(x)
    @test parent_type(Base.Slice(1:10)) <: UnitRange{Int}
    @test parent_type(Diagonal{Int,Vector{Int}}) <: Vector{Int}
    @test parent_type(UpperTriangular{Int,Matrix{Int}}) <: Matrix{Int}
    @test parent_type(LowerTriangular{Int,Matrix{Int}}) <: Matrix{Int}
end

@testset "buffer" begin
    @test ArrayInterface.buffer(sparse([1,2,3],[1,2,3],[1,2,3])) == [1, 2, 3]
    @test ArrayInterface.buffer(sparsevec([1, 2, 0, 0, 3, 0])) == [1, 2, 3]
    @test ArrayInterface.buffer(Diagonal([1,2,3])) == [1, 2, 3]
end

@test ArrayInterface.can_avx(ArrayInterface.can_avx) == false

@testset "lu_instance" begin
    A = randn(5, 5)
    @test lu_instance(A) isa typeof(lu(A))
    A = sprand(50, 50, 0.5)
    @test lu_instance(A) isa typeof(lu(A))
    @test lu_instance(1) === 1
end

@testset "ismutable" begin
    @test ArrayInterface.ismutable(rand(3))
    @test ArrayInterface.ismutable((0.1,1.0)) == false
    @test ArrayInterface.ismutable(Base.ImmutableDict{Symbol,Int64}) == false
    @test ArrayInterface.ismutable((;x=1)) == false
    @test ArrayInterface.ismutable(UnitRange{Int}) == false
    @test ArrayInterface.ismutable(Dict{Any,Any})
    @test ArrayInterface.ismutable(spzeros(1, 1))
    @test ArrayInterface.ismutable(spzeros(1))
end

@testset "can_change_size" begin
    @test ArrayInterface.can_change_size([1])
    @test ArrayInterface.can_change_size(Vector{Int})
    @test ArrayInterface.can_change_size(Dict{Symbol,Any})
    @test !ArrayInterface.can_change_size(Base.ImmutableDict{Symbol,Int64})
    @test !ArrayInterface.can_change_size(Tuple{})
end

@testset "can_setindex" begin
    @test !@inferred(ArrayInterface.can_setindex(1:2))
    @test @inferred(ArrayInterface.can_setindex(Vector{Int}))
    @test !@inferred(ArrayInterface.can_setindex(UnitRange{Int}))
    @test !@inferred(ArrayInterface.can_setindex(Base.ImmutableDict{Int,Int}))
    @test !@inferred(ArrayInterface.can_setindex(Tuple{}))
    @test !@inferred(ArrayInterface.can_setindex(NamedTuple{(),Tuple{}}))
    @test @inferred(ArrayInterface.can_setindex(Dict{Int,Int}))
end

@testset "fast_scalar_indexing" begin
    @test !fast_scalar_indexing(qr(rand(10, 10)).Q)
    if VERSION >= v"1.7"
        @test !fast_scalar_indexing(qr(rand(10, 10), ColumnNorm()).Q)
    else
        @test !fast_scalar_indexing(qr(rand(10, 10), Val(true)).Q)
    end
    @test !fast_scalar_indexing(lq(rand(10, 10)).Q)
    @test fast_scalar_indexing(Nothing)  # test default
end

@testset "isstructured" begin
    Sp=sparse([1,2,3],[1,2,3],[1,2,3])
    @test has_sparsestruct(Sp)
    rowind,colind=findstructralnz(Sp)
    @test [Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3]
end

@testset "restructure" begin
    x = rand(Float32, 2, 2)
    y = rand(4)
    yr = ArrayInterface.restructure(x, y)
    @test yr isa Matrix{Float64}
    @test size(yr) == (2,2)
    @test vec(yr) == vec(y)

    @testset "views" begin
        x = @view rand(4)[1:2]
        y = rand(2)
        yr = ArrayInterface.restructure(x, y)
        @test yr isa Vector{Float64}
        @test size(yr) == (2,)
        @test yr == y

        x = @view rand(4,4)[1:2,1:2]
        y = rand(2,2)
        yr = ArrayInterface.restructure(x, y)
        @test yr isa Matrix{Float64}
        @test size(yr) == (2,2)
        @test yr == y


        x = @view rand(4,4)[1]
        y = @view rand(2,2)[1]
        yr = ArrayInterface.restructure(x, y)
        @test yr isa Array{Float64,0}
        @test size(yr) == ()
        @test yr == y
    end
end

@testset "isstructured" begin
    @test !@inferred(ArrayInterface.isstructured(Matrix{Int}))  # default
    @test @inferred(ArrayInterface.isstructured(Hermitian{Complex{Int64}, Matrix{Complex{Int64}}}))
    @test @inferred(ArrayInterface.isstructured(Symmetric{Int,Matrix{Int}}))
    @test @inferred(ArrayInterface.isstructured(LowerTriangular{Int,Matrix{Int}}))
    @test @inferred(ArrayInterface.isstructured(UpperTriangular{Int,Matrix{Int}}))
    @test @inferred(ArrayInterface.isstructured(Diagonal{Int64, Vector{Int64}}))
    @test @inferred(ArrayInterface.isstructured(Bidiagonal{Int64, Vector{Int64}}))
    @test @inferred(ArrayInterface.isstructured(Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])))
    @test @inferred(ArrayInterface.isstructured(SymTridiagonal{Int64, Vector{Int64}}))
end

@testset "ArrayInterface.issingular" begin
    for T in [Float64, ComplexF64]
        R = randn(MersenneTwister(2), T, 5, 5)
        S = Symmetric(R)
        L = UpperTriangular(R)
        U = LowerTriangular(R)
        @test all(!ArrayInterface.issingular, [R, S, L, U, U'])
        R[:, 2] .= 0
        @test all(ArrayInterface.issingular, [R, L, U, U'])
        @test !ArrayInterface.issingular(S)
        R[2, :] .= 0
        @test ArrayInterface.issingular(S)
        @test all(!ArrayInterface.issingular, [UnitLowerTriangular(R), UnitUpperTriangular(R), UnitUpperTriangular(R)'])
    end
    @test !@inferred(ArrayInterface.issingular(Diagonal([1,2,3,4])))
    @test @inferred(ArrayInterface.issingular(UniformScaling(0)))
    @test !@inferred(ArrayInterface.issingular(Bidiagonal([1,2,3,4], [7,8,9], :U)))
    @test !@inferred(ArrayInterface.issingular(SymTridiagonal([1,2,3,4],[5,6,7])))
    @test !@inferred(ArrayInterface.issingular(Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])))
end

@testset "Sparsity Structure" begin
    D=Diagonal([1,2,3,4])
    @test has_sparsestruct(D)
    rowind,colind=findstructralnz(D)
    @test [D[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4]
    @test length(rowind)==4
    @test length(rowind)==length(colind)

    Bu = Bidiagonal([1,2,3,4], [7,8,9], :U)
    @test has_sparsestruct(Bu)
    rowind,colind=findstructralnz(Bu)
    @test [Bu[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,7,2,8,3,9,4]
    Bl = Bidiagonal([1,2,3,4], [7,8,9], :L)
    @test has_sparsestruct(Bl)
    rowind,colind=findstructralnz(Bl)
    @test [Bl[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,7,2,8,3,9,4]

    Tri=Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])
    @test has_sparsestruct(Tri)
    rowind,colind=findstructralnz(Tri)
    @test [Tri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4,4,5,6,1,2,3]

    STri=SymTridiagonal([1,2,3,4],[5,6,7])
    @test has_sparsestruct(STri)
    rowind,colind=findstructralnz(STri)
    @test [STri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3,4,5,6,7,5,6,7]
end

@testset "ndims_index" begin
    @test @inferred(ArrayInterface.ndims_index(CartesianIndices(()))) == 1
    @test @inferred(ArrayInterface.ndims_index(trues(2, 2))) == 2
    @test @inferred(ArrayInterface.ndims_index(CartesianIndex(2,2))) == 2
    @test @inferred(ArrayInterface.ndims_index(1)) == 1
end

@testset "ndims_shape" begin
    @test @inferred(ArrayInterface.ndims_shape(1)) === 0
    @test @inferred(ArrayInterface.ndims_shape(:)) === 1
    @test @inferred(ArrayInterface.ndims_shape(CartesianIndex(1, 2))) === 0
    @test @inferred(ArrayInterface.ndims_shape(CartesianIndices((2,2)))) === 2
    @test @inferred(ArrayInterface.ndims_shape([1 1])) === 2
end

@testset "indices_do_not_alias" begin
  @test ArrayInterface.instances_do_not_alias(Float64)
  @test !ArrayInterface.instances_do_not_alias(Matrix{Float64})
  @test ArrayInterface.indices_do_not_alias(Matrix{Float64})
  @test !ArrayInterface.indices_do_not_alias(BitMatrix)
  @test !ArrayInterface.indices_do_not_alias(Matrix{Matrix{Float64}})
  @test ArrayInterface.indices_do_not_alias(Adjoint{Float64,Matrix{Float64}})
  @test ArrayInterface.indices_do_not_alias(Transpose{Float64,Matrix{Float64}})
  @test ArrayInterface.indices_do_not_alias(typeof(view(rand(4,4)', 2:3, 1:2)))
  @test ArrayInterface.indices_do_not_alias(typeof(view(rand(4,4,4), CartesianIndex(1,2), 2:3)))
  @test ArrayInterface.indices_do_not_alias(typeof(view(rand(4,4)', 1:2, 2)))
  @test !ArrayInterface.indices_do_not_alias(typeof(view(rand(7),ones(Int,7))))
  @test !ArrayInterface.indices_do_not_alias(Adjoint{Matrix{Float64},Matrix{Matrix{Float64}}})
  @test !ArrayInterface.indices_do_not_alias(Transpose{Matrix{Float64},Matrix{Matrix{Float64}}})
  @test !ArrayInterface.indices_do_not_alias(typeof(view(fill(rand(4,4),4,4)', 2:3, 1:2)))
  @test !ArrayInterface.indices_do_not_alias(typeof(view(rand(4,4)', StepRangeLen(1,0,5), 1:2)))
end

@testset "ensures_all_unique" begin
    @test ArrayInterface.ensures_all_unique(BitSet())
    @test !ArrayInterface.ensures_all_unique([])
    @test ArrayInterface.ensures_all_unique(1:10)
    @test !ArrayInterface.ensures_all_unique(LinRange(1, 1, 10))
end

@testset "ensures_sorted" begin
    @test ArrayInterface.ensures_sorted(BitSet())
    @test !ArrayInterface.ensures_sorted([])
    @test ArrayInterface.ensures_sorted(1:10)
end

@testset "linearalgebra instances" begin
    for A in [rand(2,2), rand(Float32,2,2), rand(BigFloat,2,2)]
        
        @test ArrayInterface.lu_instance(A) isa typeof(lu(A))
        @test ArrayInterface.qr_instance(A) isa typeof(qr(A))

        if !(eltype(A) <: BigFloat)
            @test ArrayInterface.bunchkaufman_instance(A) isa typeof(bunchkaufman(A' * A))
            @test ArrayInterface.cholesky_instance(A) isa typeof(cholesky(A' * A))
            @test ArrayInterface.ldlt_instance(A) isa typeof(ldlt(SymTridiagonal(A' * A)))
            @test ArrayInterface.svd_instance(A) isa typeof(svd(A))
        end
    end
end