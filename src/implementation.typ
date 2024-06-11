= Implementation

The concrete implementation of FEEC in Rust.

To show the functioning of the implementation, we will be solving one of multiple of these problems:
- Elliptic problem: Hodge Poisson Equation $ (delta d + d delta) u = f $
- Parabolic problem: Hodge Heat Equation $ u_t + (delta d + d delta) u = f $
- Hyperbolic problem: Hodge Wave Equation $ u_(t t) + (delta d + d delta) u = f $

Other more advanced problems are:
- Maxwell's equations
- Stockes' equations (new NumPDE chapter 12)

It would be very nice to have a appealing visualization of the solution.
A possible approach that could yield very impressive visual results, could be to
write a GPU shader.

The implementation of this FEEC library is at least as complicated and as much work as
writing LehrFEM++ from scratch. Is this doable?

We want
- Arbitrary Spatial Dimension
but we will restrict us to the most simple setup:
- Only Simplicial meshes
- Whitney forms (only linear order 1 polynomials)
- Only Hodge-Poisson Problem

