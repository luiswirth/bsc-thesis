# Rust Implementation of Finite Element Exterior Calculus on Coordinate-Free Simplicial Complexes

This repository contains the [Typst](https://typst.app/) source files for the
Bachelor's Thesis of Luis Wirth under the supervision of Prof. Dr. Ralf Hiptmair, for Computational Science and Engineering at ETH Zürich.
For the implementation see [Formoniq](https://github.com/luiswirth/formoniq).

## Objective

The project aims to develop a finite element library in Rust based on the principles of Finite Element Exterior Calculus (FEEC) to solve partial differential equations (PDEs) formulated in terms of differential forms over simplicial complexes, using an intrinsic, coordinate-free approach.

The focus will be on solving elliptic Hodge-Laplace problems on simplicial meshes, where geometric properties are defined by the Regge metric, and linear (first-order) Whitney forms are used as basis functions.

## Background

Finite Element Exterior Calculus (FEEC) provides a unified framework that extends the finite element method using the language of differential geometry and algebraic topology. @hiptmair1999canonical By employing differential forms and (co-)chain complexes, FEEC offers a robust approach for preserving key topological and structural features in the solution of PDEs. @arnold2006 This framework is particularly well-suited for problems such as the Hodge-Laplace equation and Maxwell’s equations. @femster

Traditional finite element methods rely on explicit coordinate representations of the computational domain. However, a coordinate-free formulation aligns more naturally with the intrinsic nature of differential geometry. By representing the computational domain as a simplicial complex with an associated Riemannian metric, we can define geometric quantities (such as lengths, areas, and volumes) intrinsically, without explicit coordinates. 
This metric is an inner product on the tangent spaces and defines operators like the Hodge star, which are essential in the formulation of the Hodge-Laplace operator.

Rust was chosen for its strong guarantees in memory safety, performance, and modern language features, making it ideal for high-performance computing tasks like finite elements. 
The Rust ownership model, borrow checker, and type system act as a proof system to ensure there are no memory bugs, race conditions, or similar undefined behaviors in any program, while achieving performance levels comparable to C/C++. @klabnik2018rust @jung2017rustbelt

## Approach

+ _Coordinate-Free Simplicial Complex Data Structure_:
  Develop a mesh data structure that represents the computational domain as a simplicial complex without explicit coordinates for vertices. Instead, the mesh will store topological information (incidence and adjacency) and associate a metric tensor (geometry) on individual simplicies.
+ _Finite Element Spaces and Basis Functions_:
  Utilize Whitney forms as basis functions for discretizing differential forms on the simplicial complex. These forms are naturally defined on all simplices.
+ _Weak Formulation of the Hodge-Laplace Operator_:
  Derive the weak formulation of the Hodge-Laplace operator in the coordinate-free, intrinsic setting. This involves a generalization of integration by parts to differential forms. We should consider both the primal and mixed formulation.
+ _Element Matrices for Hodge-Laplace Problem_:
  Derive the formulas for the element matrix of a Hodge-Laplace problem. This involves explicit calculations of the exterior derivative, the codifferential, the Hodge star operator and inner products on differential forms.
+ _Assembly_:
  Assemble the Galerkin matrix from the element matrices. Using Rust's fearless concurrency feature, we can have a parallel implementation of the assembly process.
+ _Solving the system_:
  Due to the possibly high-dimensionality of the computational problem and the curse of dimensionality it would be beneficial to not only use a direct solver, but also matrix-free iterative solvers.
+ _Testing and Validation_:
  Test the implementation across multiple dimensions (e.g., 2D, 3D) to assess accuracy, convergence rates, and performance.
  Compare results to existing methods, including traditional finite element methods that use explicit coordinates and no FEEC.

## Significance

This project will result in a versatile finite element library that generalizes across dimensions, manifold geometries, and forms, broadening the applicability of finite element methods to a wider class of PDEs.
In theory, such a PDE library could be used to solve PDEs relativistically on 4D spacetime.

The project will also showcase Rust’s potential as a modern language for scientific computing, particularly in developing high-performance numerical tools that can handle complex mathematical structures.

## Prior Work

In the Julia ecosystem, there are several tools for exterior algebra and exterior calculus. Notable implementations of FEEC for arbitrary dimensional simplicial meshes and first-order Whitney forms already exist. @ddfjl However, these implementations typically rely on explicit coordinate representations.

##  Possible Extensions

- Support for Higher-Order Basis Functions:
  Extend the library to support higher-order Whitney forms and other basis functions, allowing for increased accuracy and flexibility.
- Solving Maxwell's Equations on non-trivial manifolds, such as 4D spacetime to showcase a real-life application.
- Variable Coefficient Functions:
  Implement parametric formulation of finite element spaces, enabling the inclusion of functor-like coefficient functions within the PDEs.
- Hodge Decomposition (generalization of Helmholtz Decomposition) and determining Betti numbers.
- Evolution Problems such as Heat and Wave Equation.
