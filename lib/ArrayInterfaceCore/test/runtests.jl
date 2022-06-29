using ArrayInterfaceCore
using ArrayInterfaceCore: zeromatrix
import ArrayInterfaceCore: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance,
        parent_type, zeromatrix, IndicesInfo, indices_to_dims
using Base: setindex
using LinearAlgebra
using Random
using SparseArrays
using Test

using Aqua
Aqua.test_all(ArrayInterfaceCore)

@test zeromatrix(rand(4,4,4)) == zeros(4*4*4,4*4*4)

@testset "matrix colors" begin
    @test ArrayInterfaceCore.fast_matrix_colors(1) == false
    @test ArrayInterfaceCore.fast_matrix_colors(Diagonal{Int,Vector{Int}})

    @test ArrayInterfaceCore.matrix_colors(Diagonal([1,2,3,4])) == [1, 1, 1, 1]
    @test ArrayInterfaceCore.matrix_colors(Bidiagonal([1,2,3,4], [7,8,9], :U)) == [1, 2, 1, 2]
    @test ArrayInterfaceCore.matrix_colors(Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])) == [1, 2, 3, 1]
    @test ArrayInterfaceCore.matrix_colors(SymTridiagonal([1,2,3,4],[5,6,7])) == [1, 2, 3, 1]
    @test ArrayInterfaceCore.matrix_colors(rand(4,4)) == Base.OneTo(4)
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
    @test ArrayInterfaceCore.buffer(sparse([1,2,3],[1,2,3],[1,2,3])) == [1, 2, 3]
    @test ArrayInterfaceCore.buffer(sparsevec([1, 2, 0, 0, 3, 0])) == [1, 2, 3]
    @test ArrayInterfaceCore.buffer(Diagonal([1,2,3])) == [1, 2, 3]
end

@test ArrayInterfaceCore.can_avx(ArrayInterfaceCore.can_avx) == false

@testset "lu_instance" begin
    A = randn(5, 5)
    @test lu_instance(A) isa typeof(lu(A))
    A = sprand(50, 50, 0.5)
    @test lu_instance(A) isa typeof(lu(A))
    @test lu_instance(1) === 1
end

@testset "ismutable" begin
    @test ArrayInterfaceCore.ismutable(rand(3))
    @test ArrayInterfaceCore.ismutable((0.1,1.0)) == false
    @test ArrayInterfaceCore.ismutable(Base.ImmutableDict{Symbol,Int64}) == false
    @test ArrayInterfaceCore.ismutable((;x=1)) == false
    @test ArrayInterfaceCore.ismutable(UnitRange{Int}) == false
    @test ArrayInterfaceCore.ismutable(Dict{Any,Any})
    @test ArrayInterfaceCore.ismutable(spzeros(1, 1))
    @test ArrayInterfaceCore.ismutable(spzeros(1))
end

@testset "can_change_size" begin
    @test ArrayInterfaceCore.can_change_size([1])
    @test ArrayInterfaceCore.can_change_size(Vector{Int})
    @test ArrayInterfaceCore.can_change_size(Dict{Symbol,Any})
    @test !ArrayInterfaceCore.can_change_size(Base.ImmutableDict{Symbol,Int64})
    @test !ArrayInterfaceCore.can_change_size(Tuple{})
end

@testset "can_setindex" begin
    @test !@inferred(ArrayInterfaceCore.can_setindex(1:2))
    @test @inferred(ArrayInterfaceCore.can_setindex(Vector{Int}))
    @test !@inferred(ArrayInterfaceCore.can_setindex(UnitRange{Int}))
    @test !@inferred(ArrayInterfaceCore.can_setindex(Base.ImmutableDict{Int,Int}))
    @test !@inferred(ArrayInterfaceCore.can_setindex(Tuple{}))
    @test !@inferred(ArrayInterfaceCore.can_setindex(NamedTuple{(),Tuple{}}))
    @test @inferred(ArrayInterfaceCore.can_setindex(Dict{Int,Int}))
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
    yr = ArrayInterfaceCore.restructure(x, y)
    @test yr isa Matrix{Float64}
    @test size(yr) == (2,2)
    @test vec(yr) == vec(y)

    @testset "views" begin
        x = @view rand(4)[1:2]
        y = rand(2)
        yr = ArrayInterfaceCore.restructure(x, y)
        @test yr isa Vector{Float64}
        @test size(yr) == (2,)
        @test yr == y

        x = @view rand(4,4)[1:2,1:2]
        y = rand(2,2)
        yr = ArrayInterfaceCore.restructure(x, y)
        @test yr isa Matrix{Float64}
        @test size(yr) == (2,2)
        @test yr == y


        x = @view rand(4,4)[1]
        y = @view rand(2,2)[1]
        yr = ArrayInterfaceCore.restructure(x, y)
        @test yr isa Array{Float64,0}
        @test size(yr) == ()
        @test yr == y
    end
end

@testset "isstructured" begin
    @test !@inferred(ArrayInterfaceCore.isstructured(Matrix{Int}))  # default
    @test @inferred(ArrayInterfaceCore.isstructured(Hermitian{Complex{Int64}, Matrix{Complex{Int64}}}))
    @test @inferred(ArrayInterfaceCore.isstructured(Symmetric{Int,Matrix{Int}}))
    @test @inferred(ArrayInterfaceCore.isstructured(LowerTriangular{Int,Matrix{Int}}))
    @test @inferred(ArrayInterfaceCore.isstructured(UpperTriangular{Int,Matrix{Int}}))
    @test @inferred(ArrayInterfaceCore.isstructured(Diagonal{Int64, Vector{Int64}}))
    @test @inferred(ArrayInterfaceCore.isstructured(Bidiagonal{Int64, Vector{Int64}}))
    @test @inferred(ArrayInterfaceCore.isstructured(Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])))
    @test @inferred(ArrayInterfaceCore.isstructured(SymTridiagonal{Int64, Vector{Int64}}))
end

@testset "ArrayInterfaceCore.issingular" begin
    for T in [Float64, ComplexF64]
        R = randn(MersenneTwister(2), T, 5, 5)
        S = Symmetric(R)
        L = UpperTriangular(R)
        U = LowerTriangular(R)
        @test all(!ArrayInterfaceCore.issingular, [R, S, L, U, U'])
        R[:, 2] .= 0
        @test all(ArrayInterfaceCore.issingular, [R, L, U, U'])
        @test !ArrayInterfaceCore.issingular(S)
        R[2, :] .= 0
        @test ArrayInterfaceCore.issingular(S)
        @test all(!ArrayInterfaceCore.issingular, [UnitLowerTriangular(R), UnitUpperTriangular(R), UnitUpperTriangular(R)'])
    end
    @test !@inferred(ArrayInterfaceCore.issingular(Diagonal([1,2,3,4])))
    @test @inferred(ArrayInterfaceCore.issingular(UniformScaling(0)))
    @test !@inferred(ArrayInterfaceCore.issingular(Bidiagonal([1,2,3,4], [7,8,9], :U)))
    @test !@inferred(ArrayInterfaceCore.issingular(SymTridiagonal([1,2,3,4],[5,6,7])))
    @test !@inferred(ArrayInterfaceCore.issingular(Tridiagonal([1,2,3],[1,2,3,4],[4,5,6])))
end

@testset "setindex" begin
    @testset "$(typeof(x))" for x in [
        zeros(3),
        falses(3),
        spzeros(3),
    ]
        y = setindex(x, true, 1)
        @test iszero(x)  # x is not mutated
        @test y[1] == true
        @test iszero(x[CartesianIndices(size(x)) .== [CartesianIndex(1)]])

        y2 = setindex(x, one.(x), :)
        @test iszero(x)
        @test all(isone, y2)
    end

    @testset "$(typeof(x))" for x in [
        zeros(3, 3),
        falses(3, 3),
        spzeros(3, 3),
    ]
        y = setindex(x, true, 1, 1)
        @test iszero(x)  # x is not mutated
        @test y[1, 1] == true
        @test iszero(x[CartesianIndices(size(x)) .== [CartesianIndex(1, 1)]])

        y2 = setindex(x, one.(x), :, :)
        @test iszero(x)
        @test all(isone, y2)
    end

    @testset "$(typeof(x))" for x in [
        zeros(3, 3, 3),
        falses(3, 3, 3),
    ]
        y = setindex(x, true, 1, 1, 1)
        @test iszero(x)  # x is not mutated
        @test y[1, 1, 1] == true
        @test iszero(x[CartesianIndices(size(x)) .== [CartesianIndex(1, 1, 1)]])

        y2 = setindex(x, one.(x), :, :, :)
        @test iszero(x)
        @test all(isone, y2)
    end
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

@testset "known values" begin
    CI = CartesianIndices((2, 2))

    @test isnothing(@inferred(ArrayInterfaceCore.known_first(typeof(1:4))))
    @test isone(@inferred(ArrayInterfaceCore.known_first(Base.OneTo(4))))
    @test isone(@inferred(ArrayInterfaceCore.known_first(typeof(Base.OneTo(4)))))
    @test @inferred(ArrayInterfaceCore.known_first(typeof(CI))) == CartesianIndex(1, 1)
    @test @inferred(ArrayInterfaceCore.known_first(typeof(CI))) == CartesianIndex(1, 1)

    @test isnothing(@inferred(ArrayInterfaceCore.known_last(1:4)))
    @test isnothing(@inferred(ArrayInterfaceCore.known_last(typeof(1:4))))
    @test @inferred(ArrayInterfaceCore.known_last(typeof(CI))) === nothing

    @test isnothing(@inferred(ArrayInterfaceCore.known_step(typeof(1:0.2:4))))
    @test isone(@inferred(ArrayInterfaceCore.known_step(1:4)))
    @test isone(@inferred(ArrayInterfaceCore.known_step(typeof(1:4))))
    @test isone(@inferred(ArrayInterfaceCore.known_step(typeof(Base.Slice(1:4)))))
    @test isone(@inferred(ArrayInterfaceCore.known_step(typeof(view(1:4, 1:2)))))
end

@testset "ndims_index" begin
    @test @inferred(ArrayInterfaceCore.ndims_index(CartesianIndices(()))) == 1
    @test @inferred(ArrayInterfaceCore.ndims_index(trues(2, 2))) == 2
    @test @inferred(ArrayInterfaceCore.ndims_index(CartesianIndex(2,2))) == 2
    @test @inferred(ArrayInterfaceCore.ndims_index(1)) == 1
end

@testset "ndims_shape" begin
    @test @inferred(ArrayInterfaceCore.ndims_shape(1)) === 0
    @test @inferred(ArrayInterfaceCore.ndims_shape(:)) === 1
    @test @inferred(ArrayInterfaceCore.ndims_shape(CartesianIndex(1, 2))) === 0
    @test @inferred(ArrayInterfaceCore.ndims_shape(CartesianIndices((2,2)))) === 2
    @test @inferred(ArrayInterfaceCore.ndims_shape([1 1])) === 2
end

@testset "indices_do_not_alias" begin
  @test ArrayInterfaceCore.instances_do_not_alias(Float64)
  @test !ArrayInterfaceCore.instances_do_not_alias(Matrix{Float64})
  @test ArrayInterfaceCore.indices_do_not_alias(Matrix{Float64})
  @test !ArrayInterfaceCore.indices_do_not_alias(BitMatrix)
  @test !ArrayInterfaceCore.indices_do_not_alias(Matrix{Matrix{Float64}})
  @test ArrayInterfaceCore.indices_do_not_alias(Adjoint{Float64,Matrix{Float64}})
  @test ArrayInterfaceCore.indices_do_not_alias(Transpose{Float64,Matrix{Float64}})
  @test ArrayInterfaceCore.indices_do_not_alias(typeof(view(rand(4,4)', 2:3, 1:2)))
  @test ArrayInterfaceCore.indices_do_not_alias(typeof(view(rand(4,4,4), CartesianIndex(1,2), 2:3)))
  @test ArrayInterfaceCore.indices_do_not_alias(typeof(view(rand(4,4)', 1:2, 2)))
  @test !ArrayInterfaceCore.indices_do_not_alias(typeof(view(rand(7),ones(Int,7))))
  @test !ArrayInterfaceCore.indices_do_not_alias(Adjoint{Matrix{Float64},Matrix{Matrix{Float64}}})
  @test !ArrayInterfaceCore.indices_do_not_alias(Transpose{Matrix{Float64},Matrix{Matrix{Float64}}})
  @test !ArrayInterfaceCore.indices_do_not_alias(typeof(view(fill(rand(4,4),4,4)', 2:3, 1:2)))
  @test !ArrayInterfaceCore.indices_do_not_alias(typeof(view(rand(4,4)', StepRangeLen(1,0,5), 1:2)))
end

@testset "IndicesInfo" begin

    struct SplatFirst end

    ArrayInterfaceCore.is_splat_index(::Type{SplatFirst}) = true

    @test @inferred(IndicesInfo(SubArray{Float64, 2, Vector{Float64}, Tuple{Base.ReshapedArray{Int64, 2, UnitRange{Int64}, Tuple{}}}, true})) ==
        IndicesInfo{1,(1,),((1,2),)}()

    @test @inferred(IndicesInfo{1}((Tuple{Vector{Int}}))) == IndicesInfo{1, (1,), (1,)}()

    @test @inferred(IndicesInfo{2}(Tuple{Vector{Int}})) == IndicesInfo{2, (:,), (1,)}()

    @test @inferred(IndicesInfo{1}(Tuple{SplatFirst})) == IndicesInfo{1, (1,), (1,)}()

    @test @inferred(IndicesInfo{2}(Tuple{SplatFirst})) == IndicesInfo{2, ((1,2),), ((1, 2),)}()

    @test @inferred(IndicesInfo{5}(typeof((:,[CartesianIndex(1,1),CartesianIndex(1,1)], 1, ones(Int, 2, 2), :, 1)))) ==
        IndicesInfo{5, (1, (2, 3), 4, 5, 0, 0), (1, 2, 0, (3, 4), 5, 0)}()

    @test @inferred(IndicesInfo{10}(Tuple{Vararg{Int,10}})) ==
        IndicesInfo{10, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)}()

    @test @inferred(IndicesInfo{10}(typeof((1, CartesianIndex(2, 1), 2, CartesianIndex(1, 2), 1, CartesianIndex(2, 1), 2)))) ==
        IndicesInfo{10, (1, (2, 3), 4, (5, 6), 7, (8, 9), 10), (0, 0, 0, 0, 0, 0, 0)}()

    @test @inferred(IndicesInfo{10}(typeof((fill(true, 4, 4), 2, fill(true, 4, 4), 2, 1, fill(true, 4, 4), 1)))) ==
        IndicesInfo{10, ((1, 2), 3, (4, 5), 6, 7, (8, 9), 10), (1, 0, 2, 0, 0, 3, 0)}()

    @test @inferred(IndicesInfo{10}(typeof((1, SplatFirst(), 2, SplatFirst(), CartesianIndex(1, 1))))) ==
        IndicesInfo{10, (1, (2, 3, 4, 5, 6), 7, 8, (9, 10)), (0, (1, 2, 3, 4, 5), 0, 6, 0)}()
end
