# Rust Implementation of Finite Element Exterior Calculus on Coordinate-Free Simplicial Complexes

This repository contains the [Typst](https://typst.app/) source files for
the Bachelor's Thesis of Luis Wirth under the supervision of Prof. Dr. Ralf
Hiptmair, for Computational Science and Engineering at ETH Zürich. For the
implementation itself, see [Formoniq](https://github.com/luiswirth/formoniq).

## Abstract

This thesis presents the development of a novel finite element library in
Rust based on the principles of Finite Element Exterior Calculus (FEEC). The
library solves partial differential equations formulated using differential
forms on abstract, coordinate-free simplicial complexes in arbitrary dimensions,
employing an intrinsic Riemannian metric derived from edge lengths via Regge
Calculus. We focus on solving elliptic Hodge-Laplace eigenvalue and source
problems on the nD de Rham complex. We restrict ourselves to first-order Whitney
basis functions. The implementation is verified through convergence studies
using manufactured solutions, demonstrating expected theoretical accuracy.
