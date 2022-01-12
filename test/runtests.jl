using ArrayInterface
using ArrayInterface: StaticInt, True, False, NDIndex
using ArrayInterface: zeromatrix
import ArrayInterface: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance,
    device, contiguous_axis, contiguous_batch_size, stride_rank, dense_dims, static, NDIndex,
    is_lazy_conjugate, parent_type, dimnames
using BandedMatrices
using BlockBandedMatrices
using Base: setindex
using IfElse
using LinearAlgebra
using OffsetArrays
using Random
using SparseArrays
using StaticArrays
using SuiteSparse
using Test

if VERSION ≥ v"1.6"
    using Aqua
    Aqua.test_all(ArrayInterface)
end

@test isone(ArrayInterface.known_first(typeof(StaticArrays.SOneTo(7))))
@test ArrayInterface.known_last(typeof(StaticArrays.SOneTo(7))) == 7
@test ArrayInterface.known_length(typeof(StaticArrays.SOneTo(7))) == 7
@test @inferred(ArrayInterface.known_length(NDIndex((1,2,3)))) === 3

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


Sp=sparse([1,2,3],[1,2,3],[1,2,3])
@test has_sparsestruct(Sp)
rowind,colind=findstructralnz(Sp)
@test [Tri[rowind[i],colind[i]] for i in 1:length(rowind)]==[1,2,3]


@test !fast_scalar_indexing(qr(rand(10, 10)).Q)
if VERSION >= v"1.7"
    @test !fast_scalar_indexing(qr(rand(10, 10), ColumnNorm()).Q)
else
    @test !fast_scalar_indexing(qr(rand(10, 10), Val(true)).Q)
end
@test !fast_scalar_indexing(lq(rand(10, 10)).Q)
@test fast_scalar_indexing(Missing)  # test default

@testset "can_setindex" begin
    @test !@inferred(ArrayInterface.can_setindex(1:2))
    @test @inferred(ArrayInterface.can_setindex(Vector{Int}))
    @test !@inferred(ArrayInterface.can_setindex(UnitRange{Int}))
end

@testset "ArrayInterface.isstructured" begin
    @test !@inferred(ArrayInterface.isstructured(Matrix{Int}))  # default
    @test @inferred(ArrayInterface.isstructured(Hermitian{Complex{Int64}, Matrix{Complex{Int64}}}))
    @test @inferred(ArrayInterface.isstructured(Symmetric{Int,Matrix{Int}}))
    @test @inferred(ArrayInterface.isstructured(LowerTriangular{Int,Matrix{Int}}))
    @test @inferred(ArrayInterface.isstructured(UpperTriangular{Int,Matrix{Int}}))
    @test @inferred(ArrayInterface.isstructured(typeof(D)))
    @test @inferred(ArrayInterface.isstructured(typeof(Bu)))
    @test @inferred(ArrayInterface.isstructured(typeof(Tri)))
    @test @inferred(ArrayInterface.isstructured(typeof(STri)))
end

B=BandedMatrix(Ones(5,5), (-1,2))
B[band(1)].=[1,2,3,4]
B[band(2)].=[5,6,7]
@test has_sparsestruct(B)
rowind,colind=findstructralnz(B)
@test [B[rowind[i],colind[i]] for i in 1:length(rowind)]==[5,6,7,1,2,3,4]
B=BandedMatrix(Ones(4,6), (-1,2))
B[band(1)].=[1,2,3,4]
B[band(2)].=[5,6,7,8]
rowind,colind=findstructralnz(B)
@test [B[rowind[i],colind[i]] for i in 1:length(rowind)]==[5,6,7,8,1,2,3,4]

BB=BlockBandedMatrix(Ones(10,10),[1,2,3,4],[4,3,2,1],(1,0))
BB[Block(1,1)].=[1 2 3 4]
BB[Block(2,1)].=[5 6 7 8;9 10 11 12]
rowind,colind=findstructralnz(BB)
@test [BB[rowind[i],colind[i]] for i in 1:length(rowind)]==
    [1,5,9,2,6,10,3,7,11,4,8,12,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1]

dense=collect(Ones(8,8))
for i in 1:8
    dense[:,i].=[1,2,3,4,5,6,7,8]
end
BBB=BandedBlockBandedMatrix(dense, [4, 4] ,[4, 4], (1, 1), (1, 1))
rowind,colind=findstructralnz(BBB)
@test [BBB[rowind[i],colind[i]] for i in 1:length(rowind)]==
    [1,2,3,1,2,3,4,2,3,4,5,6,7,5,6,7,8,6,7,8,
     1,2,3,1,2,3,4,2,3,4,5,6,7,5,6,7,8,6,7,8]

@testset "ismutable" begin
    @test ArrayInterface.ismutable(rand(3))
    x = @SVector [1,2,3]
    @test ArrayInterface.ismutable(x) == false
    @test ArrayInterface.ismutable(view(x, 1:2)) == false
    x = @MVector [1,2,3]
    @test ArrayInterface.ismutable(x) == true
    @test ArrayInterface.ismutable(view(x, 1:2)) == true
    @test ArrayInterface.ismutable((0.1,1.0)) == false
    @test ArrayInterface.ismutable(Base.ImmutableDict{Symbol,Int64}) == false
    @test ArrayInterface.ismutable((;x=1)) == false
    @test ArrayInterface.ismutable(UnitRange{Int}) == false
    @test ArrayInterface.ismutable(Dict{Any,Any})
    @test ArrayInterface.ismutable(spzeros(1, 1))
    @test ArrayInterface.ismutable(spzeros(1))
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

@testset "lu_instance" begin
  for A in [
    randn(5, 5),
    @SMatrix(randn(5, 5)),
    @MMatrix(randn(5, 5)),
    sprand(50, 50, 0.5)
  ]
    @test lu_instance(A) isa typeof(lu(A))
  end
  @test lu_instance(1) === 1
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
    @test !@inferred(ArrayInterface.issingular(D))
    @test @inferred(ArrayInterface.issingular(UniformScaling(0)))
    @test !@inferred(ArrayInterface.issingular(Bu))
    @test !@inferred(ArrayInterface.issingular(STri))
    @test !@inferred(ArrayInterface.issingular(Tri))
end

@test zeromatrix(rand(4,4,4)) == zeros(4*4*4,4*4*4)

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
    @test parent_type(SizedVector{1, Int, Vector{Int}}) <: Vector{Int}
end

@testset "buffer" begin
    @test ArrayInterface.buffer(Sp) == [1, 2, 3]
    @test ArrayInterface.buffer(sparsevec([1, 2, 0, 0, 3, 0])) == [1, 2, 3]
    @test ArrayInterface.buffer(Diagonal([1,2,3])) == [1, 2, 3]
end

include("ranges.jl")

# Dummy array type with undetermined contiguity properties
struct DummyZeros{T,N} <: AbstractArray{T,N}
    dims :: Dims{N}
    DummyZeros{T}(dims...) where {T} = new{T,length(dims)}(dims)
end
DummyZeros(dims...) = DummyZeros{Float64}(dims...)
Base.size(x::DummyZeros) = x.dims
Base.getindex(::DummyZeros{T}, inds...) where {T} = zero(T)

struct Wrapper{T,N,P<:AbstractArray{T,N}} <: ArrayInterface.AbstractArray2{T,N}
    parent::P
end
ArrayInterface.parent_type(::Type{<:Wrapper{T,N,P}}) where {T,N,P} = P
Base.parent(x::Wrapper) = x.parent
ArrayInterface.device(::Type{T}) where {T<:Wrapper} = ArrayInterface.device(parent_type(T))

struct DenseWrapper{T,N,P<:AbstractArray{T,N}} <: DenseArray{T,N} end
ArrayInterface.parent_type(::Type{DenseWrapper{T,N,P}}) where {T,N,P} = P

@testset "Memory Layout" begin
    x = zeros(100);
    # R = reshape(view(x, 1:100), (10,10));
    # A = zeros(3,4,5);
    A = Wrapper(reshape(view(x, 1:60), (3,4,5)));
    B = A .== 0;
    D1 = view(A, 1:2:3, :, :);  # first dimension is discontiguous
    D2 = view(A, :, 2:2:4, :);  # first dimension is contiguous

    @test @inferred(ArrayInterface.defines_strides(x))
    @test @inferred(ArrayInterface.defines_strides(A))
    @test @inferred(ArrayInterface.defines_strides(D1))
    @test !@inferred(ArrayInterface.defines_strides(view(A, :, [1,2],1)))
    @test @inferred(ArrayInterface.defines_strides(DenseWrapper{Int,2,Matrix{Int}}))

    @test @inferred(device(A)) === ArrayInterface.CPUPointer()
    @test @inferred(device(B)) === ArrayInterface.CPUIndex()
    @test @inferred(device(-1:19)) === ArrayInterface.CPUIndex()
    @test @inferred(device((1,2,3))) === ArrayInterface.CPUTuple()
    @test @inferred(device(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.CPUPointer()
    @test @inferred(device(view(A, 1, :, 2:4))) === ArrayInterface.CPUPointer()
    @test @inferred(device(view(A, 1, :, 2:4)')) === ArrayInterface.CPUPointer()
    @test @inferred(device(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173))) === ArrayInterface.CPUPointer()
    @test @inferred(device(view(OffsetArray(A,2,3,-12), 4, :, -11:-9))) === ArrayInterface.CPUPointer()
    @test @inferred(device(view(OffsetArray(A,2,3,-12), 3, :, [-11,-10,-9])')) === ArrayInterface.CPUIndex()
    @test @inferred(device(OffsetArray(@SArray(zeros(2,2,2)),-123,29,3231))) === ArrayInterface.CPUTuple()
    @test @inferred(device(OffsetArray(@view(@SArray(zeros(2,2,2))[1,1:2,:]),-3,4))) === ArrayInterface.CPUTuple()
    @test @inferred(device(OffsetArray(@MArray(zeros(2,2,2)),8,-2,-5))) === ArrayInterface.CPUPointer()
    @test ismissing(device("Hello, world!"))
    @test @inferred(device(DenseWrapper{Int,2,Matrix{Int}})) === ArrayInterface.CPUPointer()
    #=
    @btime ArrayInterface.contiguous_axis($(reshape(view(zeros(100), 1:60), (3,4,5))))
      0.047 ns (0 allocations: 0 bytes)
    =#
    @test @inferred(contiguous_axis(@SArray(zeros(2,2,2)))) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis(A)) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis(B)) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis(-1:19)) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis(D1)) === ArrayInterface.StaticInt(-1)
    @test @inferred(contiguous_axis(D2)) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.StaticInt(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis(transpose(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])))) === ArrayInterface.StaticInt(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterface.StaticInt(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterface.StaticInt(-1)
    @test @inferred(contiguous_axis(PermutedDimsArray(@view(A[2,:,:]),(2,1)))) === ArrayInterface.StaticInt(-1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterface.StaticInt(-1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterface.StaticInt(1)
    @test @inferred(contiguous_axis((3,4))) === StaticInt(1)
    @test @inferred(contiguous_axis(rand(4)')) === StaticInt(2)
    @test @inferred(contiguous_axis(view(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])', :, 1)')) === StaticInt(-1)
    @test @inferred(contiguous_axis(DummyZeros(3,4))) === missing
    @test @inferred(contiguous_axis(PermutedDimsArray(DummyZeros(3,4), (2, 1)))) === missing
    @test @inferred(contiguous_axis(view(DummyZeros(3,4), 1, :))) === missing
    @test @inferred(contiguous_axis(view(DummyZeros(3,4), 1, :)')) === missing

    @test @inferred(ArrayInterface.contiguous_axis_indicator(@SArray(zeros(2,2,2)))) == (true,false,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(A)) == (true,false,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(B)) == (true,false,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(-1:10)) == (true,)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(PermutedDimsArray(A,(3,1,2)))) == (false,true,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) == (true,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) == (false,true)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) == (false,true,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) == (false,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) == (false,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) == (true,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,[1,3,4]]))) == (false,true,false)
    @test @inferred(ArrayInterface.contiguous_axis_indicator(DummyZeros(3,4))) === missing

    @test @inferred(contiguous_batch_size(@SArray(zeros(2,2,2)))) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(A)) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(B)) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(-1:18)) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(PermutedDimsArray(A,(3,1,2)))) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterface.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterface.StaticInt(-1)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterface.StaticInt(-1)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterface.StaticInt(0)
    let u_base = randn(10, 10)
        u_view = view(u_base, 3, :)
        u_reshaped_view = reshape(u_view, 1, size(u_base, 2))
        @test @inferred(contiguous_batch_size(u_view)) === ArrayInterface.StaticInt(-1)
        @test @inferred(contiguous_batch_size(u_reshaped_view)) === ArrayInterface.StaticInt(-1)
    end

    @test @inferred(stride_rank(@SArray(zeros(2,2,2)))) == (1, 2, 3)
    @test @inferred(stride_rank(A)) == (1,2,3)
    @test @inferred(stride_rank(B)) == (1,2,3)
    @test @inferred(stride_rank(-4:4)) == (1,)
    @test @inferred(stride_rank(view(A,:,:,1))) === (static(1), static(2))
    @test @inferred(stride_rank(view(A,:,:,1))) === ((ArrayInterface.StaticInt(1),ArrayInterface.StaticInt(2)))
    @test @inferred(stride_rank(PermutedDimsArray(A,(3,1,2)))) == (3, 1, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) == (1, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) == (2, 1)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) == (3, 1, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) == (3, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) == (2, 3)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) == (1, 3)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,2,1])')) == (2, 1)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,[1,3,4]]))) == (3, 1, 2)
    @test @inferred(stride_rank(DummyZeros(3,4)')) === missing
    @test @inferred(stride_rank(PermutedDimsArray(DummyZeros(3,4), (2, 1)))) === missing
    @test @inferred(stride_rank(view(DummyZeros(3,4), 1, :))) === missing
    uA = reinterpret(reshape, UInt64, A)
    @test @inferred(stride_rank(uA)) === stride_rank(A)
    rA = reinterpret(reshape, SVector{3,Float64}, A)
    @test @inferred(stride_rank(rA)) === (static(1), static(2))
    cA = copy(rA)
    rcA = reinterpret(reshape, Float64, cA)
    @test @inferred(stride_rank(rcA)) === stride_rank(A)

    #=
    @btime ArrayInterface.is_column_major($(PermutedDimsArray(A,(3,1,2))))
      0.047 ns (0 allocations: 0 bytes)
    @btime ArrayInterface.is_column_major($(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))))
      0.047 ns (0 allocations: 0 bytes)
    @btime ArrayInterface.is_column_major($(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')))
      0.047 ns (0 allocations: 0 bytes)

      PermutedDimsArray(A,(3,1,2))[2:3,1:2,:])
      @view(PermutedDimsArray(reshape(view(zeros(100), 1:60), (3,4,5)), (3,1,2)), 2:3, 1:2, :)
    =#

    @test @inferred(ArrayInterface.is_column_major(@SArray(zeros(2,2,2)))) === True()
    @test @inferred(ArrayInterface.is_column_major(A)) === True()
    @test @inferred(ArrayInterface.is_column_major(B)) === True()
    @test @inferred(ArrayInterface.is_column_major(-4:7)) === False()
    @test @inferred(ArrayInterface.is_column_major(PermutedDimsArray(A,(3,1,2)))) === False()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === True()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === False()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === False()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === False()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === True()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === True()
    @test @inferred(ArrayInterface.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[:,2,1])')) === False()
    @test @inferred(ArrayInterface.is_column_major(2.3)) === False()

    @test @inferred(dense_dims(@SArray(zeros(2,2,2)))) == (true,true,true)
    @test @inferred(dense_dims(A)) == (true,true,true)
    @test @inferred(dense_dims(B)) == (true,true,true)
    @test @inferred(dense_dims(-3:9)) == (true,)
    @test @inferred(dense_dims(PermutedDimsArray(A,(3,1,2)))) == (true,true,true)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) == (true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) == (false,true)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) == (false,true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,1:2]))) == (false,true,true)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) == (false,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) == (false,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) == (true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,[1,2]]))) == (false,true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,[1,2,3],:]))) == (false,false,false)
    # TODO Currently Wrapper can't function the same as Array because Array can change
    # the dimensions on reshape. We should be rewrapping the result in `Wrapper` but we
    # first need to develop a standard method for reconstructing arrays
    @test @inferred(dense_dims(vec(parent(A)))) == (true,)
    @test @inferred(dense_dims(vec(parent(A))')) == (true,true)
    @test @inferred(dense_dims(DummyZeros(3,4))) === missing
    @test @inferred(dense_dims(DummyZeros(3,4)')) === missing
    @test @inferred(dense_dims(PermutedDimsArray(DummyZeros(3,4), (2, 1)))) === missing
    @test @inferred(dense_dims(view(DummyZeros(3,4), :, 1))) === missing
    @test @inferred(dense_dims(view(DummyZeros(3,4), :, 1)')) === missing
    @test @inferred(ArrayInterface.is_dense(A)) === @inferred(ArrayInterface.is_dense(A)) === @inferred(ArrayInterface.is_dense(PermutedDimsArray(A,(3,1,2)))) === @inferred(ArrayInterface.is_dense(Array{Float64,0}(undef))) === True()
    @test @inferred(ArrayInterface.is_dense(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === @inferred(ArrayInterface.is_dense(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,[1,2]]))) === @inferred(ArrayInterface.is_dense(@view(PermutedDimsArray(A,(3,1,2))[2:3,[1,2,3],:]))) === False()

  
    C = Array{Int8}(undef, 2,2,2,2);
    doubleperm = PermutedDimsArray(PermutedDimsArray(C,(4,2,3,1)), (4,2,1,3));
    @test collect(strides(C))[collect(stride_rank(doubleperm))] == collect(strides(doubleperm))

    @test @inferred(ArrayInterface.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),1)) === Base.Slice(ArrayInterface.OptionallyStaticUnitRange(4,6))
    @test @inferred(ArrayInterface.indices(OffsetArray(view(PermutedDimsArray(A, (3,1,2)), 1, :, 2:4)', 3, -173),2)) === Base.Slice(ArrayInterface.OptionallyStaticUnitRange(-172,-170))

    Am = @MMatrix rand(2,10);
    @test @inferred(ArrayInterface.strides(view(Am,1,:))) === (StaticInt(2),)

    if isdefined(Base, :ReshapedReinterpretArray) # reinterpret(reshape,...) tests
        C1 = reinterpret(reshape, Float64, PermutedDimsArray(Array{Complex{Float64}}(undef, 3,4,5), (2,1,3)));
        C2 = reinterpret(reshape, Complex{Float64}, PermutedDimsArray(view(A,1:2,:,:), (1,3,2)));
        C3 = reinterpret(reshape, Complex{Float64}, PermutedDimsArray(Wrapper(reshape(view(x, 1:24), (2,3,4))), (1,3,2)));

        @test @inferred(ArrayInterface.defines_strides(C1))
        @test @inferred(ArrayInterface.defines_strides(C2))
        @test @inferred(ArrayInterface.defines_strides(C3))

        @test @inferred(device(C1)) === ArrayInterface.CPUPointer()
        @test @inferred(device(C2)) === ArrayInterface.CPUPointer()
        @test @inferred(device(C3)) === ArrayInterface.CPUPointer()

        @test @inferred(contiguous_batch_size(C1)) === ArrayInterface.StaticInt(0)
        @test @inferred(contiguous_batch_size(C2)) === ArrayInterface.StaticInt(0)
        @test @inferred(contiguous_batch_size(C3)) === ArrayInterface.StaticInt(0)

        @test @inferred(stride_rank(C1)) == (1,3,2,4)
        @test @inferred(stride_rank(C2)) == (2,1)
        @test @inferred(stride_rank(C3)) == (2,1)

        @test @inferred(contiguous_axis(C1)) === StaticInt(1)
        @test @inferred(contiguous_axis(C2)) === StaticInt(0)
        @test @inferred(contiguous_axis(C3)) === StaticInt(2)

        @test @inferred(ArrayInterface.contiguous_axis_indicator(C1)) == (true,false,false,false)
        @test @inferred(ArrayInterface.contiguous_axis_indicator(C2)) == (false,false)
        @test @inferred(ArrayInterface.contiguous_axis_indicator(C3)) == (false,true)

        @test @inferred(ArrayInterface.is_column_major(C1)) === False()
        @test @inferred(ArrayInterface.is_column_major(C2)) === False()
        @test @inferred(ArrayInterface.is_column_major(C3)) === False()

        @test @inferred(dense_dims(C1)) == (true,true,true,true)
        @test @inferred(dense_dims(C2)) == (false,false)
        @test @inferred(dense_dims(C3)) == (true,true)
    end
end

@testset "Static-Dynamic Size, Strides, and Offsets" begin
    A = zeros(3, 4, 5);
    A[:] = 1:60
    Ap = @view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])';
    S = @SArray zeros(2,3,4)
    A_trailingdim = zeros(2,3,4,1)
    Sp = @view(PermutedDimsArray(S,(3,1,2))[2:3,1:2,:]);
    M = @MArray zeros(2,3,4)
    Mp = @view(PermutedDimsArray(M,(3,1,2))[:,2,:])';
    Sp2 = @view(PermutedDimsArray(S,(3,2,1))[2:3,:,:]);
    Mp2 = @view(PermutedDimsArray(M,(3,1,2))[2:3,:,2])';
    D = @view(A[:,2:2:4,:]);
    R = StaticInt(1):StaticInt(2);
    Rnr = reinterpret(Int32, R);
    Ar = reinterpret(Float32, A);
    A2 = zeros(4, 3, 5)
    A2r = reinterpret(ComplexF64, A2)

    irev = Iterators.reverse(S)
    igen = Iterators.map(identity, S)
    iacc = Iterators.accumulate(+, S)
    iprod = Iterators.product(axes(S)...)
    iflat = Iterators.flatten(iprod)
    ienum = enumerate(S)
    ipairs = pairs(S)
    izip = zip(S,S)


    sv5 = @SVector(zeros(5)); v5 = Vector{Float64}(undef, 5);
    @test @inferred(ArrayInterface.size(sv5)) === (StaticInt(5),)
    @test @inferred(ArrayInterface.size(v5)) === (5,)
    @test @inferred(ArrayInterface.size(A)) === (3,4,5)
    @test @inferred(ArrayInterface.size(Ap)) === (2,5)
    @test @inferred(ArrayInterface.size(A)) === size(A)
    @test @inferred(ArrayInterface.size(Ap)) === size(Ap)
    @test @inferred(ArrayInterface.size(R)) === (StaticInt(2),)
    @test @inferred(ArrayInterface.size(Rnr)) === (StaticInt(4),)
    @test @inferred(ArrayInterface.known_length(Rnr)) === 4
    @test @inferred(ArrayInterface.size(A2)) === (4,3,5)
    @test @inferred(ArrayInterface.size(A2r)) === (2,3,5)

    @test @inferred(ArrayInterface.size(irev)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(iprod)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(iflat)) === (static(72),)
    @test @inferred(ArrayInterface.size(igen)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(iacc)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(ienum)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(ipairs)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(izip)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(zip(S, A_trailingdim))) === (StaticInt(2), StaticInt(3), StaticInt(4), static(1))
    @test @inferred(ArrayInterface.size(zip(A_trailingdim, S))) === (StaticInt(2), StaticInt(3), StaticInt(4), static(1))
    @test @inferred(ArrayInterface.size(S)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(Sp)) === (2, 2, StaticInt(3))
    @test @inferred(ArrayInterface.size(Sp2)) === (2, StaticInt(3), StaticInt(2))
    @test @inferred(ArrayInterface.size(S)) == size(S)
    @test @inferred(ArrayInterface.size(Sp)) == size(Sp)
    @test @inferred(ArrayInterface.size(parent(Sp2))) === (static(4), static(3), static(2))
    @test @inferred(ArrayInterface.size(Sp2)) == size(Sp2)
    @test @inferred(ArrayInterface.size(Sp2, StaticInt(1))) === 2
    @test @inferred(ArrayInterface.size(Sp2, StaticInt(2))) === StaticInt(3)
    @test @inferred(ArrayInterface.size(Sp2, StaticInt(3))) === StaticInt(2)
    @test @inferred(ArrayInterface.size(Wrapper(Sp2), StaticInt(3))) === StaticInt(2)

    @test @inferred(ArrayInterface.size(M)) === (StaticInt(2), StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(Mp)) === (StaticInt(3), StaticInt(4))
    @test @inferred(ArrayInterface.size(Mp2)) === (StaticInt(2), 2)
    @test @inferred(ArrayInterface.size(M)) == size(M)
    @test @inferred(ArrayInterface.size(Mp)) == size(Mp)
    @test @inferred(ArrayInterface.size(Mp2)) == size(Mp2)
    @test @inferred(ArrayInterface.size(D)) == size(D)

    @test @inferred(ArrayInterface.known_size(A)) === (missing, missing, missing)
    @test @inferred(ArrayInterface.known_size(Ap)) === (missing,missing)
    @test @inferred(ArrayInterface.known_size(Wrapper(Ap))) === (missing,missing)
    @test @inferred(ArrayInterface.known_size(R)) === (2,)
    @test @inferred(ArrayInterface.known_size(Wrapper(R))) === (2,)
    @test @inferred(ArrayInterface.known_size(Rnr)) === (4,)
    @test @inferred(ArrayInterface.known_size(Rnr, static(1))) === 4
    @test @inferred(ArrayInterface.known_size(Ar)) === (missing,missing, missing,)
    @test @inferred(ArrayInterface.known_size(Ar, static(1))) === missing
    @test @inferred(ArrayInterface.known_size(Ar, static(4))) === 1
    @test @inferred(ArrayInterface.known_size(A2)) === (missing, missing, missing)
    @test @inferred(ArrayInterface.known_size(A2r)) === (missing, missing, missing)

    @test @inferred(ArrayInterface.known_size(irev)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(igen)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(iprod)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(iflat)) === (72,)
    @test @inferred(ArrayInterface.known_size(iacc)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(ienum)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(izip)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(ipairs)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(zip(S, A_trailingdim))) === (2, 3, 4, 1)
    @test @inferred(ArrayInterface.known_size(zip(A_trailingdim, S))) === (2, 3, 4, 1)
    @test @inferred(ArrayInterface.known_length(Iterators.flatten(((x,y) for x in 0:1 for y in 'a':'c')))) === missing

    @test @inferred(ArrayInterface.known_size(S)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(Wrapper(S))) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(Sp)) === (missing, missing, 3)
    @test @inferred(ArrayInterface.known_size(Wrapper(Sp))) === (missing, missing, 3)
    @test @inferred(ArrayInterface.known_size(Sp2)) === (missing, 3, 2)
    @test @inferred(ArrayInterface.known_size(Sp2, StaticInt(1))) === missing
    @test @inferred(ArrayInterface.known_size(Sp2, StaticInt(2))) === 3
    @test @inferred(ArrayInterface.known_size(Sp2, StaticInt(3))) === 2
    @test @inferred(ArrayInterface.known_size(M)) === (2, 3, 4)
    @test @inferred(ArrayInterface.known_size(Mp)) === (3, 4)
    @test @inferred(ArrayInterface.known_size(Mp2)) === (2, missing)

    @test @inferred(ArrayInterface.strides(A)) === (StaticInt(1), 3, 12)
    @test @inferred(ArrayInterface.strides(Ap)) === (StaticInt(1), 12)
    @test @inferred(ArrayInterface.strides(A)) == strides(A)
    @test @inferred(ArrayInterface.strides(Ap)) == strides(Ap)
    @test @inferred(ArrayInterface.strides(Ar)) === (StaticInt{1}(), 6, 24)
    @test @inferred(ArrayInterface.strides(A2)) === (StaticInt(1), 4, 12)
    @test @inferred(ArrayInterface.strides(A2r)) === (StaticInt(1), 2, 6)

    @test @inferred(ArrayInterface.strides(S)) === (StaticInt(1), StaticInt(2), StaticInt(6))
    @test @inferred(ArrayInterface.strides(Sp)) === (StaticInt(6), StaticInt(1), StaticInt(2))
    @test @inferred(ArrayInterface.strides(Sp2)) === (StaticInt(6), StaticInt(2), StaticInt(1))
    @test @inferred(ArrayInterface.strides(view(Sp2, :, 1, 1)')) === (StaticInt(6), StaticInt(6))

    @test @inferred(ArrayInterface.stride(Sp2, StaticInt(1))) === StaticInt(6)
    @test @inferred(ArrayInterface.stride(Sp2, StaticInt(2))) === StaticInt(2)
    @test @inferred(ArrayInterface.stride(Sp2, StaticInt(3))) === StaticInt(1)

    @test @inferred(ArrayInterface.strides(M)) === (StaticInt(1), StaticInt(2), StaticInt(6))
    @test @inferred(ArrayInterface.strides(Mp)) === (StaticInt(2), StaticInt(6))
    @test @inferred(ArrayInterface.strides(Mp2)) === (StaticInt(1), StaticInt(6))
    @test @inferred(ArrayInterface.strides(M)) == strides(M)
    @test @inferred(ArrayInterface.strides(Mp)) == strides(Mp)
    @test @inferred(ArrayInterface.strides(Mp2)) == strides(Mp2)
    @test_throws MethodError ArrayInterface.strides(DummyZeros(3,4))

    @test @inferred(ArrayInterface.known_strides(A)) === (1, missing, missing)
    @test @inferred(ArrayInterface.known_strides(Ap)) === (1, missing)
    @test @inferred(ArrayInterface.known_strides(Ar)) === (1, missing, missing)
    @test @inferred(ArrayInterface.known_strides(reshape(view(zeros(100), 1:60), (3,4,5)))) === (1, missing, missing)
    @test @inferred(ArrayInterface.known_strides(A2)) === (1, missing, missing)
    @test @inferred(ArrayInterface.known_strides(A2r)) === (1, missing, missing)

    @test @inferred(ArrayInterface.known_strides(S)) === (1, 2, 6)
    @test @inferred(ArrayInterface.known_strides(Sp)) === (6, 1, 2)
    @test @inferred(ArrayInterface.known_strides(Sp2)) === (6, 2, 1)
    @test @inferred(ArrayInterface.known_strides(Sp2, StaticInt(1))) === 6
    @test @inferred(ArrayInterface.known_strides(Sp2, StaticInt(2))) === 2
    @test @inferred(ArrayInterface.known_strides(Sp2, StaticInt(3))) === 1
    @test @inferred(ArrayInterface.known_strides(Sp2, StaticInt(4))) === ArrayInterface.known_length(Sp2)
    @test @inferred(ArrayInterface.known_strides(view(Sp2, :, 1, 1)')) === (6, 6)

    @test @inferred(ArrayInterface.known_strides(M)) === (1, 2, 6)
    @test @inferred(ArrayInterface.known_strides(Mp)) === (2, 6)
    @test @inferred(ArrayInterface.known_strides(Mp2)) === (1, 6)

    @test @inferred(ArrayInterface.offsets(A)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(Ap)) === (StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(Ar)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(A2)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(A2r)) === (StaticInt(1), StaticInt(1), StaticInt(1))

    @test @inferred(ArrayInterface.offsets(S)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(Sp)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(Sp2)) === (StaticInt(1), StaticInt(1), StaticInt(1))

    @test @inferred(ArrayInterface.offsets(M)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(Mp)) === (StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterface.offsets(Mp2)) === (StaticInt(1), StaticInt(1))

    @test @inferred(ArrayInterface.known_offsets(A)) === (1, 1, 1)
    @test @inferred(ArrayInterface.known_offsets(Ap)) === (1, 1)
    @test @inferred(ArrayInterface.known_offsets(Ar)) === (1, 1, 1)
    @test @inferred(ArrayInterface.known_offsets(Ar, static(1))) === 1
    @test @inferred(ArrayInterface.known_offsets(Ar, static(4))) === 1
    @test @inferred(ArrayInterface.known_offsets(A2)) === (1, 1, 1)
    @test @inferred(ArrayInterface.known_offsets(A2r)) === (1, 1, 1)

    @test @inferred(ArrayInterface.known_offsets(S)) === (1, 1, 1)
    @test @inferred(ArrayInterface.known_offsets(Sp)) === (1, 1, 1)
    @test @inferred(ArrayInterface.known_offsets(Sp2)) === (1, 1, 1)

    @test @inferred(ArrayInterface.known_offsets(M)) === (1, 1, 1)
    @test @inferred(ArrayInterface.known_offsets(Mp)) === (1, 1)
    @test @inferred(ArrayInterface.known_offsets(Mp2)) === (1, 1)

    @test @inferred(ArrayInterface.known_offsets(R)) === (1,)
    @test @inferred(ArrayInterface.known_offsets(Rnr)) === (1,)
    @test @inferred(ArrayInterface.known_offsets(1:10)) === (1,)

    O = OffsetArray(A, 3, 7, 10);
    Op = PermutedDimsArray(O,(3,1,2));
    @test @inferred(ArrayInterface.offsets(O)) === (4, 8, 11)
    @test @inferred(ArrayInterface.offsets(Op)) === (11, 4, 8)

    @test @inferred(ArrayInterface.offsets((1,2,3))) === (StaticInt(1),)
    o = OffsetArray(vec(A), 8);
    @test @inferred(ArrayInterface.offset1(o)) === 9

    if VERSION ≥ v"1.6.0-DEV.1581"
        colors = [(R = rand(), G = rand(), B = rand()) for i ∈ 1:100];

        colormat = reinterpret(reshape, Float64, colors)
        @test @inferred(ArrayInterface.strides(colormat)) === (StaticInt(1), StaticInt(3))
        @test @inferred(ArrayInterface.dense_dims(colormat)) === (True(),True())
        @test @inferred(ArrayInterface.dense_dims(view(colormat,:,4))) === (True(),)
        @test @inferred(ArrayInterface.dense_dims(view(colormat,:,4:7))) === (True(),True())
        @test @inferred(ArrayInterface.dense_dims(view(colormat,2:3,:))) === (True(),False())

        Rr = reinterpret(reshape, Int32, R)
        @test @inferred(ArrayInterface.size(Rr)) === (StaticInt(2),StaticInt(2))
        @test @inferred(ArrayInterface.known_size(Rr)) === (2, 2)

        Sr = Wrapper(reinterpret(reshape, Complex{Int64}, S))
        @test @inferred(ArrayInterface.size(Sr)) == (static(3), static(4))
        @test @inferred(ArrayInterface.known_size(Sr)) === (3, 4)
        @test @inferred(ArrayInterface.strides(Sr)) === (static(1), static(3))
        Sr2 = reinterpret(reshape, Complex{Int64}, S);
        @test @inferred(ArrayInterface.dense_dims(Sr2)) === (True(),True())
        @test @inferred(ArrayInterface.dense_dims(view(Sr2,:,2))) === (True(),)
        @test @inferred(ArrayInterface.dense_dims(view(Sr2,:,2:3))) === (True(),True())
        @test @inferred(ArrayInterface.dense_dims(view(Sr2,2:3,:))) === (True(),False())

        Ar2c = reinterpret(reshape, Complex{Float64}, view(rand(2, 5, 7), :, 2:4, 3:5));
        @test @inferred(ArrayInterface.strides(Ar2c)) === (StaticInt(1), 5)
        Ar2c_static = reinterpret(reshape, Complex{Float64}, view(@MArray(rand(2, 5, 7)), :, 2:4, 3:5));
        @test @inferred(ArrayInterface.strides(Ar2c_static)) === (StaticInt(1), StaticInt(5))

        Ac2r = reinterpret(reshape, Float64, view(rand(ComplexF64, 5, 7), 2:4, 3:6));
        @test @inferred(ArrayInterface.strides(Ac2r)) === (StaticInt(1), StaticInt(2), 10)
        Ac2r_static = reinterpret(reshape, Float64, view(@MMatrix(rand(ComplexF64, 5, 7)), 2:4, 3:6));
        @test @inferred(ArrayInterface.strides(Ac2r_static)) === (StaticInt(1), StaticInt(2), StaticInt(10))

        Ac2t = reinterpret(reshape, Tuple{Float64,Float64}, view(rand(ComplexF64, 5, 7), 2:4, 3:6));
        @test @inferred(ArrayInterface.strides(Ac2t)) === (StaticInt(1), 5)
        Ac2t_static = reinterpret(reshape, Tuple{Float64,Float64}, view(@MMatrix(rand(ComplexF64, 5, 7)), 2:4, 3:6));
        @test @inferred(ArrayInterface.strides(Ac2t_static)) === (StaticInt(1), StaticInt(5))
    end
end

@testset "" begin
    include("array_index.jl")
end

@testset "Reshaped views" begin
    u_base = randn(10, 10)
    u_view = view(u_base, 3, :)
    u_reshaped_view1 = reshape(u_view, 1, :)
    u_reshaped_view2 = reshape(u_view, 2, :)

    @test @inferred(ArrayInterface.defines_strides(u_base))
    @test @inferred(ArrayInterface.defines_strides(u_view))
    @test @inferred(ArrayInterface.defines_strides(u_reshaped_view1))
    @test @inferred(ArrayInterface.defines_strides(u_reshaped_view2))

    # See https://github.com/JuliaArrays/ArrayInterface.jl/issues/160
    @test @inferred(ArrayInterface.strides(u_base)) == (StaticInt(1), 10)
    @test @inferred(ArrayInterface.strides(u_view)) == (10,)
    @test @inferred(ArrayInterface.strides(u_reshaped_view1)) == (10, 10)
    @test @inferred(ArrayInterface.strides(u_reshaped_view2)) == (10, 20)

    # See https://github.com/JuliaArrays/ArrayInterface.jl/issues/157
    @test @inferred(ArrayInterface.dense_dims(u_base)) == (True(), True())
    @test @inferred(ArrayInterface.dense_dims(u_view)) == (False(),)
    @test @inferred(ArrayInterface.dense_dims(u_reshaped_view1)) == (False(), False())
    @test @inferred(ArrayInterface.dense_dims(u_reshaped_view2)) == (False(), False())
end

@testset "Reinterpreted reshaped views" begin
    u_base = randn(1, 4, 4, 5);
    u_vectors = reshape(reinterpret(SVector{1, eltype(u_base)}, u_base),
                        Base.tail(size(u_base))...)
    u_view = view(u_vectors, 2, :, 3)
    u_view_reinterpreted = reinterpret(eltype(u_base), u_view)
    u_view_reshaped = reshape(u_view_reinterpreted, 1, length(u_view))

    # See https://github.com/JuliaArrays/ArrayInterface.jl/issues/163
    @test @inferred(ArrayInterface.strides(u_base)) == (StaticInt(1), 1, 4, 16)
    @test @inferred(ArrayInterface.strides(u_vectors)) == (StaticInt(1), 4, 16)
    @test @inferred(ArrayInterface.strides(u_view)) == (4,)
    @test @inferred(ArrayInterface.strides(u_view_reinterpreted)) == (4,)
    @test @inferred(ArrayInterface.strides(u_view_reshaped)) == (4, 4)
    @test @inferred(ArrayInterface.axes(u_vectors)) isa ArrayInterface.axes_types(u_vectors)
end

@test ArrayInterface.can_avx(ArrayInterface.can_avx) == false

@testset "can_change_size" begin
    @test ArrayInterface.can_change_size([1])
    @test ArrayInterface.can_change_size(Vector{Int})
    @test ArrayInterface.can_change_size(Dict{Symbol,Any})
    @test !ArrayInterface.can_change_size(Base.ImmutableDict{Symbol,Int64})
    @test !ArrayInterface.can_change_size(Tuple{})
end

@testset "known_length" begin
    @test ArrayInterface.known_length(@inferred(ArrayInterface.indices(SOneTo(7)))) == 7
    @test ArrayInterface.known_length(1:2) === missing
    @test ArrayInterface.known_length((1,)) == 1
    @test ArrayInterface.known_length((a=1,b=2)) == 2
    @test ArrayInterface.known_length([]) === missing
    @test ArrayInterface.known_length(CartesianIndex((1,2,3))) === 3

    x = view(SArray{Tuple{3,3,3}}(ones(3,3,3)), :, SOneTo(2), 2)
    @test @inferred(ArrayInterface.known_length(x)) == 6
    @test @inferred(ArrayInterface.known_length(x')) == 6

    itr = StaticInt(1):StaticInt(10)
    @inferred(ArrayInterface.known_length((i for i in itr))) == 10

    v = @SVector rand(8);
    A = @MMatrix rand(7,6);
    T = SizedArray{Tuple{5,4,3}}(zeros(5,4,3));
    @test @inferred(ArrayInterface.static_length(v)) === StaticInt(8)
    @test @inferred(ArrayInterface.static_length(A)) === StaticInt(42)
    @test @inferred(ArrayInterface.static_length(T)) === StaticInt(60)
end

@testset "indices" begin
    A23 = ones(2,3); SA23 = @SMatrix ones(2,3);
    A32 = ones(3,2); SA32 = @SMatrix ones(3,2);

    @test @inferred(ArrayInterface.indices(A23, (static(1),static(2)))) === (Base.Slice(StaticInt(1):2), Base.Slice(StaticInt(1):3))
    @test @inferred(ArrayInterface.indices((A23, A32))) == 1:6
    @test @inferred(ArrayInterface.indices((SA23, A32))) == 1:6
    @test @inferred(ArrayInterface.indices((A23, SA32))) == 1:6
    @test @inferred(ArrayInterface.indices((SA23, SA32))) == 1:6
    @test @inferred(ArrayInterface.indices(A23)) == 1:6
    @test @inferred(ArrayInterface.indices(SA23)) == 1:6
    @test @inferred(ArrayInterface.indices(A23, 1)) == 1:2
    @test @inferred(ArrayInterface.indices(SA23, StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, A32), (1, 2))) == 1:2
    @test @inferred(ArrayInterface.indices((SA23, A32), (StaticInt(1), 2))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, SA32), (1, StaticInt(2)))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((SA23, SA32), (StaticInt(1), StaticInt(2)))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, A23), 1)) == 1:2
    @test @inferred(ArrayInterface.indices((SA23, SA23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((SA23, A23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((A23, SA23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))
    @test @inferred(ArrayInterface.indices((SA23, SA23), StaticInt(1))) === Base.Slice(StaticInt(1):StaticInt(2))

    @test_throws AssertionError ArrayInterface.indices((A23, ones(3, 3)), 1)
    @test_throws AssertionError ArrayInterface.indices((A23, ones(3, 3)), (1, 2))
    @test_throws AssertionError ArrayInterface.indices((SA23, ones(3, 3)), StaticInt(1))
    @test_throws AssertionError ArrayInterface.indices((SA23, ones(3, 3)), (StaticInt(1), 2))
    @test_throws AssertionError ArrayInterface.indices((SA23, SA23), (StaticInt(1), StaticInt(2)))

    @test size(similar(ones(2, 4), ArrayInterface.indices(ones(2, 4), 1), ArrayInterface.indices(ones(2, 4), 2))) == (2, 4)
    @test axes(ArrayInterface.indices(ones(2,2))) === (StaticInt(1):4,)
    @test axes(Base.Slice(StaticInt(2):4)) === (Base.IdentityUnitRange(StaticInt(2):4),)
    @test Base.axes1(ArrayInterface.indices(ones(2,2))) === StaticInt(1):4
    @test Base.axes1(Base.Slice(StaticInt(2):4)) === Base.IdentityUnitRange(StaticInt(2):4)

    x = vec(A23); y = vec(A32);
    @test ArrayInterface.indices((x',y'),StaticInt(1)) === Base.Slice(StaticInt(1):StaticInt(1))
    @test ArrayInterface.indices((x,y), StaticInt(2)) === Base.Slice(StaticInt(1):StaticInt(1))
end

@testset "insert/deleteat" begin
    @test @inferred(ArrayInterface.insert([1,2,3], 2, -2)) == [1, -2, 2, 3]
    @test @inferred(ArrayInterface.deleteat([1, 2, 3], 2)) == [1, 3]

    @test @inferred(ArrayInterface.deleteat([1, 2, 3], [1, 2])) == [3]
    @test @inferred(ArrayInterface.deleteat([1, 2, 3], [1, 3])) == [2]
    @test @inferred(ArrayInterface.deleteat([1, 2, 3], [2, 3])) == [1]

    @test @inferred(ArrayInterface.insert((2,3,4), 1, -2)) == (-2, 2, 3, 4)
    @test @inferred(ArrayInterface.insert((2,3,4), 2, -2)) == (2, -2, 3, 4)
    @test @inferred(ArrayInterface.insert((2,3,4), 3, -2)) == (2, 3, -2, 4)

    @test @inferred(ArrayInterface.deleteat((2, 3, 4), 1)) == (3, 4)
    @test @inferred(ArrayInterface.deleteat((2, 3, 4), 2)) == (2, 4)
    @test @inferred(ArrayInterface.deleteat((2, 3, 4), 3)) == (2, 3)
    @test ArrayInterface.deleteat((1, 2, 3), [1, 2]) == (3,)
end

@testset "axes" begin
    include("axes.jl")
end

@testset "arrayinterface" begin
    include("arrayinterface.jl")
end
include("indexing.jl")
include("dimensions.jl")

@testset "broadcast" begin
    include("broadcast.jl")
end

@testset "lazy conj" begin
    a = rand(ComplexF64, 2)
    @test @inferred(is_lazy_conjugate(a)) == false
    b = a'
    @test @inferred(is_lazy_conjugate(b)) == true
    c = transpose(b)
    @test @inferred(is_lazy_conjugate(c)) == true
    d = c'
    @test @inferred(is_lazy_conjugate(d)) == false
    e = permutedims(d)
    @test @inferred(is_lazy_conjugate(e)) == false

    @test @inferred(is_lazy_conjugate([1,2,3]')) == false # We don't care about conj on `<:Real`
end
