= Introduction 

Finite Element Exterior Calculus (FEEC) is a mathematical framework
for formulating the theory of the Finite Element Method. Instead of only relying
on vector calculus it makes use of the far more general theory of Differential Forms.


"Without referring to differential geometry, several authors had devised vector
valued finite elements that can be regarded as special cases of discrete
differential forms. Their constructions are formidably intricate and require
much technical effort. A substantial simplification can be achieved: One should
exploit the facilities provided by differential geometry for a completely
coordinate-free treatment of discrete differential forms. Once we have shed
the cumbersome vector calculus, everything can be constructed and studied with
unmatched generality and elegance. In particular, all orders of forms and all
degrees of polynomial approximation can be dealt with in the same framework.
This can be done for simplicial meshes in arbitrary dimension."
@hiptmair-whitney

There are very few implementations of FEEC available.
One is #link("https://github.com/Airini/FEECa")[FEECa].
Which is a Finite Element Exterior Calculus Implementation in Haskell.
However Haskell is not a language that is suited for High Performance Computing (HPC)
but rather known for it's mathmatical expressiveness.
It is not capable of solving PDEs.
Rust combines these two traits of being high performant, while still being
expressive.

Fenics has support for FEEC threough the package "Exterior".
The Exterior package can currently compute the finite elements (including the
nodal basis) P_r\Lambda^k(T) and P_r^-\Lambda^k(T) for arbitrary simplices T
(and its sub simplices) in Rn for arbirtrary r, k, n.

PyDEC is a Python library implementing Discrete Exterior Calculus (DEC) and
lowest order finite element exterior calculus (FEEC) i.e., Whitney forms.

