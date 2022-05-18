

v = Array{Float64}(undef, 4)
m = Array{Float64}(undef, 4, 3)
@test @inferred(ArrayInterfaceCore.axes(v')) === (StaticInt(1):StaticInt(1),Base.OneTo(4))
@test @inferred(ArrayInterfaceCore.axes(m')) === (Base.OneTo(3),Base.OneTo(4))
@test ArrayInterfaceCore.axes(v', StaticInt(1)) === StaticInt(1):StaticInt(1)
@test ArrayInterfaceCore.axes(v, StaticInt(2)) === StaticInt(1):StaticInt(1)

@testset "LazyAxis" begin
    A = zeros(3,4,5);
    SA = MArray(zeros(3,4,5))
    lz1 = ArrayInterfaceCore.LazyAxis{1}(A)
    slz1 = ArrayInterfaceCore.LazyAxis{1}(SA)
    lzc = ArrayInterfaceCore.LazyAxis{:}(A)
    slzc = ArrayInterfaceCore.LazyAxis{:}(SA)

    @test @inferred(first(lz1)) === @inferred(first(slz1))
    @test @inferred(first(lzc)) === @inferred(first(slzc))
    @test @inferred(last(lz1)) === @inferred(last(slz1))
    @test @inferred(last(lzc)) === @inferred(last(slzc))
    @test @inferred(length(lz1)) === @inferred(length(slz1))
    @test @inferred(length(lzc)) === @inferred(length(slzc))
    @test @inferred(Base.to_shape(lzc)) == length(slzc)
    @test @inferred(Base.checkindex(Bool, lzc, 1)) & @inferred(Base.checkindex(Bool, slzc, 1))
    @test axes(lzc)[1] == Base.axes1(lzc) == axes(Base.Slice(lzc))[1] == Base.axes1(Base.Slice(lzc))

    @test @inferred(getindex(lz1, 2)) == 2
    @test @inferred(getindex(lz1, 1:2)) == 1:2
    @test @inferred(getindex(lz1, 1:1:3)) == 1:1:3

    @test @inferred(ArrayInterfaceCore.parent_type(ArrayInterfaceCore.LazyAxis{:}(A))) <: Base.OneTo{Int}
    @test @inferred(ArrayInterfaceCore.parent_type(ArrayInterfaceCore.LazyAxis{4}(SA))) <: ArrayInterfaceCore.SOneTo{1}
    @test @inferred(ArrayInterfaceCore.parent_type(ArrayInterfaceCore.LazyAxis{:}(SA))) <: ArrayInterfaceCore.SOneTo{60}
    @test ArrayInterfaceCore.can_change_size(ArrayInterfaceCore.LazyAxis{1,Vector{Any}})

    Aperm = PermutedDimsArray(A, (3,1,2))
    Aview = @view(Aperm[:,1:2,1])
    @test map(parent, @inferred(ArrayInterfaceCore.lazy_axes(Aperm))) === @inferred(ArrayInterfaceCore.axes(Aperm))
    @test map(parent, @inferred(ArrayInterfaceCore.lazy_axes(Aview))) === @inferred(ArrayInterfaceCore.axes(Aview))
    @test map(parent, @inferred(ArrayInterfaceCore.lazy_axes(Aview'))) === @inferred(ArrayInterfaceCore.axes(Aview'))
    @test map(parent, @inferred(ArrayInterfaceCore.lazy_axes((1:2)'))) === @inferred(ArrayInterfaceCore.axes((1:2)'))

    @test_throws DimensionMismatch ArrayInterfaceCore.LazyAxis{0}(A)
end

@testset "`axes(A, dim)`` with `dim > ndims(A)` (#224)" begin
    m = 2
    n = 3
    B = Array{Float64, 2}(undef, m, n)
    b = view(B, :, 1)

    @test @inferred(ArrayInterfaceCore.axes(B, 1)) == 1:m
    @test @inferred(ArrayInterfaceCore.axes(B, 2)) == 1:n
    @test @inferred(ArrayInterfaceCore.axes(B, 3)) == 1:1

    @test @inferred(ArrayInterfaceCore.axes(B, static(1))) == 1:m
    @test @inferred(ArrayInterfaceCore.axes(B, static(2))) == 1:n
    @test @inferred(ArrayInterfaceCore.axes(B, static(3))) == 1:1

    @test @inferred(ArrayInterfaceCore.axes(b, 1)) == 1:m
    @test @inferred(ArrayInterfaceCore.axes(b, 2)) == 1:1
    @test @inferred(ArrayInterfaceCore.axes(b, 3)) == 1:1

    @test @inferred(ArrayInterfaceCore.axes(b, static(1))) == 1:m
    @test @inferred(ArrayInterfaceCore.axes(b, static(2))) == 1:1
    @test @inferred(ArrayInterfaceCore.axes(b, static(3))) == 1:1
end

@testset "SubArray Adjoint Axis" begin
  N = 4; d = rand(N);

  @test @inferred(ArrayInterfaceCore.axes_types(typeof(view(d',:,1:2)))) === Tuple{ArrayInterfaceCore.OptionallyStaticUnitRange{StaticInt{1}, StaticInt{1}}, Base.OneTo{Int64}}

end
if isdefined(Base, :ReshapedReinterpretArray)
  @testset "ReshapedReinterpretArray" begin
    a = rand(3, 5)
    ua = reinterpret(reshape, UInt64, a)
    @test ArrayInterfaceCore.axes(ua) === ArrayInterfaceCore.axes(a)
    @test ArrayInterfaceCore.axes(ua, 1) === ArrayInterfaceCore.axes(a, 1)
    @test @inferred(ArrayInterfaceCore.axes(ua)) isa ArrayInterfaceCore.axes_types(ua)
    u8a = reinterpret(reshape, UInt8, a)
    @test @inferred(ArrayInterfaceCore.axes(u8a)) isa ArrayInterfaceCore.axes_types(u8a)
    @test @inferred(ArrayInterfaceCore.axes(u8a, static(1))) isa ArrayInterfaceCore.axes_types(u8a, 1)
    @test @inferred(ArrayInterfaceCore.axes(u8a, static(2))) isa ArrayInterfaceCore.axes_types(u8a, 2)
    fa = reinterpret(reshape, Float64, copy(u8a))
    @inferred(ArrayInterfaceCore.axes(fa)) isa ArrayInterfaceCore.axes_types(fa)
  end
end

