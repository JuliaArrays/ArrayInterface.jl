using ArrayInterfaceCore
using ArrayInterfaceCore: zeromatrix
import ArrayInterfaceCore: has_sparsestruct, findstructralnz, fast_scalar_indexing, lu_instance,
    device, contiguous_axis, contiguous_batch_size, stride_rank, dense_dims, static, NDIndex,
    is_lazy_conjugate, parent_type, dimnames, zeromatrix
using Base: setindex
using LinearAlgebra
using Random
using SparseArrays
using Static
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

@testset "Reshaped views" begin
    u_base = randn(10, 10)
    u_view = view(u_base, 3, :)
    u_reshaped_view1 = reshape(u_view, 1, :)
    u_reshaped_view2 = reshape(u_view, 2, :)

    @test @inferred(ArrayInterfaceCore.defines_strides(u_base))
    @test @inferred(ArrayInterfaceCore.defines_strides(u_view))
    @test @inferred(ArrayInterfaceCore.defines_strides(u_reshaped_view1))
    @test @inferred(ArrayInterfaceCore.defines_strides(u_reshaped_view2))

    # See https://github.com/JuliaArrays/ArrayInterfaceCore.jl/issues/160
    @test @inferred(ArrayInterfaceCore.strides(u_base)) == (StaticInt(1), 10)
    @test @inferred(ArrayInterfaceCore.strides(u_view)) == (10,)
    @test @inferred(ArrayInterfaceCore.strides(u_reshaped_view1)) == (10, 10)
    @test @inferred(ArrayInterfaceCore.strides(u_reshaped_view2)) == (10, 20)

    # See https://github.com/JuliaArrays/ArrayInterfaceCore.jl/issues/157
    @test @inferred(ArrayInterfaceCore.dense_dims(u_base)) == (True(), True())
    @test @inferred(ArrayInterfaceCore.dense_dims(u_view)) == (False(),)
    @test @inferred(ArrayInterfaceCore.dense_dims(u_reshaped_view1)) == (False(), False())
    @test @inferred(ArrayInterfaceCore.dense_dims(u_reshaped_view2)) == (False(), False())
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

@testset "known_length" begin
    @test ArrayInterfaceCore.known_length(1:2) === nothing
    @test ArrayInterfaceCore.known_length((1,)) == 1
    @test ArrayInterfaceCore.known_length((a=1,b=2)) == 2
    @test ArrayInterfaceCore.known_length([]) === nothing
    @test ArrayInterfaceCore.known_length(CartesianIndex((1,2,3))) === 3
    @test @inferred(ArrayInterfaceCore.known_length(NDIndex((1,2,3)))) === 3

    itr = StaticInt(1):StaticInt(10)
    @inferred(ArrayInterfaceCore.known_length((i for i in itr))) == 10
end

A = zeros(3, 4, 5);
A[:] = 1:60
Ap = @view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])';
S = MArray(zeros(2,3,4))
A_trailingdim = zeros(2,3,4,1)
Sp = @view(PermutedDimsArray(S,(3,1,2))[2:3,1:2,:]);

Sp2 = @view(PermutedDimsArray(S,(3,2,1))[2:3,:,:]);

Mp = @view(PermutedDimsArray(S,(3,1,2))[:,2,:])';
Mp2 = @view(PermutedDimsArray(S,(3,1,2))[2:3,:,2])';

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

sv5 = MArray(zeros(5));
v5 = Vector{Float64}(undef, 5);

@testset "size" begin
    include("size.jl")
end

@testset "ArrayIndex" begin
    include("array_index.jl")
end

@testset "Range Interface" begin
    include("ranges.jl")
end

@testset "axes" begin
    include("axes.jl")
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

@testset "is_lazy_conjugate" begin
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

@testset "Pseudo-mutating" begin
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

    @testset "insert/deleteat" begin
        @test @inferred(ArrayInterfaceCore.insert([1,2,3], 2, -2)) == [1, -2, 2, 3]
        @test @inferred(ArrayInterfaceCore.deleteat([1, 2, 3], 2)) == [1, 3]

        @test @inferred(ArrayInterfaceCore.deleteat([1, 2, 3], [1, 2])) == [3]
        @test @inferred(ArrayInterfaceCore.deleteat([1, 2, 3], [1, 3])) == [2]
        @test @inferred(ArrayInterfaceCore.deleteat([1, 2, 3], [2, 3])) == [1]

        @test @inferred(ArrayInterfaceCore.insert((2,3,4), 1, -2)) == (-2, 2, 3, 4)
        @test @inferred(ArrayInterfaceCore.insert((2,3,4), 2, -2)) == (2, -2, 3, 4)
        @test @inferred(ArrayInterfaceCore.insert((2,3,4), 3, -2)) == (2, 3, -2, 4)

        @test @inferred(ArrayInterfaceCore.deleteat((2, 3, 4), 1)) == (3, 4)
        @test @inferred(ArrayInterfaceCore.deleteat((2, 3, 4), 2)) == (2, 4)
        @test @inferred(ArrayInterfaceCore.deleteat((2, 3, 4), 3)) == (2, 3)
        @test ArrayInterfaceCore.deleteat((1, 2, 3), [1, 2]) == (3,)
    end
end
