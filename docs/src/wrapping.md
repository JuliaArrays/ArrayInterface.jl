# Julia's Array Wrapping Interface

The following functions make it easier to handle array wrappers, such as `Adjoint`, which
can obscure an underlying array's properties under a layer of indirection.

```@docs
ArrayInterface.is_forwarding_wrapper
ArrayInterface.buffer
ArrayInterface.parent_type
```

## Additional Array Wrapping Libraries

If dealing with array wrappers, additionally consider:

- [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl): conversions for handling device (GPU) wrappers.