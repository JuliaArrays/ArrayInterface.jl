
@testset "StaticInt" begin
    @test iszero(StaticInt(0))
    @test !iszero(StaticInt(1))
    @test !isone(StaticInt(0))
    @test isone(StaticInt(1))
    @test @inferred(one(StaticInt(1))) === StaticInt(1)
    @test @inferred(zero(StaticInt(1))) === StaticInt(0)
    @test @inferred(one(StaticInt)) === StaticInt(1)
    @test @inferred(zero(StaticInt)) === StaticInt(0) === StaticInt(StaticInt(Val(0)))
    @test eltype(one(StaticInt)) <: Int

    x = StaticInt(1)
    @test @inferred(Bool(x)) isa Bool
    @test @inferred(BigInt(x)) isa BigInt
    @test @inferred(Integer(x)) === x
    # test for ambiguities and correctness
    for i ∈ Any[StaticInt(0), StaticInt(1), StaticInt(2), 3]
        for j ∈ Any[StaticInt(0), StaticInt(1), StaticInt(2), 3]
            i === j === 3 && continue
            for f ∈ [+, -, *, ÷, %, <<, >>, >>>, &, |, ⊻, ==, ≤, ≥]
                (iszero(j) && ((f === ÷) || (f === %))) && continue # integer division error
                @test convert(Int, @inferred(f(i,j))) == f(convert(Int, i), convert(Int, j))
            end
        end
        i == 3 && break
        for f ∈ [+, -, *, /, ÷, %, ==, ≤, ≥]
            w = f(convert(Int, i), 1.4)
            x = f(1.4, convert(Int, i))
            @test convert(typeof(w), @inferred(f(i, 1.4))) === w
            @test convert(typeof(x), @inferred(f(1.4, i))) === x # if f is division and i === StaticInt(0), returns `NaN`; hence use of ==== in check.
            (((f === ÷) || (f === %)) && (i === StaticInt(0))) && continue
            y = f(convert(Int, i), 2 // 7)
            z = f(2 // 7, convert(Int, i))
            @test convert(typeof(y), @inferred(f(i, 2 // 7))) === y
            @test convert(typeof(z), @inferred(f(2 // 7, i))) === z 
        end
    end

    @test UnitRange{Int16}(StaticInt(-9), 17) === Int16(-9):Int16(17)
    @test UnitRange{Int16}(-7, StaticInt(19)) === Int16(-7):Int16(19)
    @test UnitRange(-11, StaticInt(15)) === -11:15
    @test UnitRange(StaticInt(-11), 15) === -11:15
    @test UnitRange(StaticInt(-11), StaticInt(15)) === -11:15
    @test float(StaticInt(8)) === 8.0
end

@testset "StaticBool" begin
    t = True()
    f = False()

    @test @inferred(StaticInt(t)) === StaticInt(1)
    @test @inferred(StaticInt(f)) === StaticInt(0)

    @test @inferred(~t) === f
    @test @inferred(~f) === t
    @test @inferred(!t) === f
    @test @inferred(!f) === t
    @test @inferred(+t) === StaticInt(1)
    @test @inferred(+f) === StaticInt(0)
    @test @inferred(-t) === StaticInt(-1)
    @test @inferred(-f) === StaticInt(0)

    @test @inferred(|(true, f))
    @test @inferred(|(f, true))
    @test @inferred(|(f, f)) === f
    @test @inferred(|(f, t)) === t
    @test @inferred(|(t, f)) === t
    @test @inferred(|(t, t)) === t

    @test !@inferred(Base.:(&)(true, f))
    @test !@inferred(Base.:(&)(f, true))
    @test @inferred(Base.:(&)(f, f)) === f
    @test @inferred(Base.:(&)(f, t)) === f
    @test @inferred(Base.:(&)(t, f)) === f
    @test @inferred(Base.:(&)(t, t)) === t

    @test @inferred(<(f, f)) === f
    @test @inferred(<(f, t)) === t
    @test @inferred(<(t, f)) === f
    @test @inferred(<(t, t)) === f

    @test @inferred(<=(f, f)) === t
    @test @inferred(<=(f, t)) === t
    @test @inferred(<=(t, f)) === f
    @test @inferred(<=(t, t)) === t

    @test @inferred(*(f, t)) === t & f
    @test @inferred(-(f, t)) === StaticInt(f) - StaticInt(t)
    @test @inferred(+(f, t)) === StaticInt(f) + StaticInt(t)

    @test @inferred(^(t, f)) == ^(true, false)
    @test @inferred(^(t, t)) == ^(true, true)

    @test @inferred(^(2, f)) == 1
    @test @inferred(^(2, t)) == 2

    @test @inferred(^(BigInt(2), f)) == 1
    @test @inferred(^(BigInt(2), t)) == 2

    @test @inferred(div(t, t)) === t
    @test_throws DivideError div(t, f)

    @test @inferred(rem(t, t)) === f
    @test_throws DivideError rem(t, f)
    @test @inferred(mod(t, t)) === f

    @test @inferred(all((t, t, t)))
    @test !@inferred(all((t, f, t)))
    @test !@inferred(all((f, f, f)))

    @test @inferred(any((t, t, t)))
    @test @inferred(any((t, f, t)))
    @test !@inferred(any((f, f, f)))

    x = StaticInt(1)
    y = StaticInt(0)
    z = StaticInt(-1)
    @test @inferred(ArrayInterface.eq(x, y)) === f
    @test @inferred(ArrayInterface.eq(x, x)) === t

    @test @inferred(ArrayInterface.ne(x, y)) === t
    @test @inferred(ArrayInterface.ne(x, x)) === f

    @test @inferred(ArrayInterface.gt(x, y)) === t
    @test @inferred(ArrayInterface.gt(y, x)) === f

    @test @inferred(ArrayInterface.ge(x, y)) === t
    @test @inferred(ArrayInterface.ge(y, x)) === f

    @test @inferred(ArrayInterface.lt(y, x)) === t
    @test @inferred(ArrayInterface.lt(x, y)) === f

    @test @inferred(ArrayInterface.le(y, x)) === t
    @test @inferred(ArrayInterface.le(x, y)) === f
end

