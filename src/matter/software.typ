= Software Design & Implementation Choices

In this chapter we want to briefly discuss some general software engineering
decisions for our library, #strong("formoniq").

== Rust

We have chosen Rust @klabnik:rust as the main programming language for the
implementation of our Finite Element library. There are various reasons for this
choice, some of which we briefly outline here.

=== Memory Safety + Performance

Rust is a strongly-typed, modern systems programming language that combines
performance on par with `C/C++` with strong memory safety guarantees.
Unlike traditional memory-safe languages that rely on garbage collection, Rust
achieves memory safety through a unique ownership and borrowing model inspired
by formal software verification and static analysis techniques. This ensures
safety at compile-time without compromising runtime performance @klabnik:rust.

The Rust compiler acts as a proof checker, requiring the programmer to provide
sufficient evidence for the safety of their code.
This is accomplished by extending Rust's strong type system with an
ownership and borrowing model that enforces clear semantics regarding
memory responsibility @jung:rustbelt.

This system completely eliminates entire classes of memory-related bugs
(dangling pointers, use-after-free, data races), making software significantly
more reliable. This isn't limited to simple single-threaded serial computations,
but extends to concurrent, parallel and distributed computing, ensuring
that data races can never occur. This "fearless concurrency" feature allows
developers to be fully confident that any parallel code written in Rust that
compiles will behave as expected @RustLang.


=== Expressiveness and Abstraction

Rust is a highly expressive language enabling powerful abstractions without
sacrificing performance, often referred to as zero-cost abstractions
@klabnik:rust. This allows for a direct realization of concepts, making Rust
well-suited for capturing precise mathematical structures naturally. Key
features include:

- *Traits and Generics*: Facilitate powerful polymorphism and code reuse without
  the rigid hierarchies of traditional object-oriented inheritance. Compile-time
  monomorphization ensures zero-cost abstraction. Traits mitigate the notorious
  template-related compiler errors of `C++`, as trait bounds explicitly state
  required behaviors within function and struct signatures.
- *Enums, Option, and Result*: Algebraic data types, particularly `enum` as a
  tagged union, enable type-safe state representation and a simple yet powerful
  form of polymorphism. `Option<T>` eliminates null-pointer issues by enforcing
  explicit handling of absence, and `Result<T, E>` provides structured error
  handling without exceptional control flow.
- *Expression-Based Language and Pattern Matching*: Most constructs are
  expressions returning values. Combined with powerful pattern matching for
  destructuring composite types, this leads to concise, readable, functional-style composable
  code.
- *Functional Programming and Iterators*: Rust embraces higher-order functions,
  closures, and efficient, lazy iterators with standard methods like `map`,
  `filter`, and `fold`, promoting declarative coding.

Together, these features allow developers to write concise, maintainable, and
high-performance software, combining modern paradigms with low-level control.

=== Infrastructure and Tooling

Beyond language features, Rust's exceptional, official tooling ecosystem
streamlines development @RustLang. This consistency contrasts favorably with the
fragmented `C++` ecosystem.

- *Cargo* is Rust's official package manager and build system, which is one of
  the most impressive pieces of tooling.  It eliminates the need for traditional
  build tools like Makefiles and CMake, which are often complex and difficult
  to maintain—not to mention the dozens of other build systems for `C++`.
  Cargo simplifies dependency management through its seamless integration with
  `crates.io`, Rust’s central package repository. Developers can effortlessly
  include third-party libraries by specifying them in the `Cargo.toml` file
  (which is versioned by git), with Cargo automatically handling downloading,
  compiling, and dependency resolution while enforcing semantic versioning.
  Publishing a crate is also straightforward.
- *Clippy* is Rust's official linter, offering valuable suggestions for
  improving code quality, adhering to best practices, and catching common
  mistakes. Our codebase does not produce a single warning or lint, passing all
  default checks for code quality.
- *Rustdoc* is Rust's official documentation tool, allowing developers to
  write inline documentation using Markdown, seamlessly integrated with code
  comments. This documentation can be compiled into a browsable website with
  `cargo doc` and is automatically published to `docs.rs` when a crate is uploaded
  to `crates.io`. The documentation for our libraries is also available there.
- *Rusttest* is the testing functionality built into Cargo for running unit
  and integration tests. Unit tests can be placed alongside the normal source code
  with a simple `#[test]` attribute, and the `cargo test` command will execute
  all test functions, verifying correctness. This command also ensures that all
  code snippets in the documentation are compiled and checked for runtime errors,
  keeping documentation up-to-date without requiring external test frameworks like
  Google Test.
- *Rustfmt* standardizes code formatting, eliminating debates about code style
  and ensuring consistency across projects. Our codebase fully adheres to Rustfmt's
  formatting guidelines. For conciseness however we will be breaking
  the formatting style when putting code inline into this document.

This comprehensive tooling ensures a smooth, efficient, and reliable development
experience.

=== Drawbacks

Despite its strengths, we encountered some challenges:

- *Ecosystem Maturity*: Being relatively young (1.0 release in 2015 @klabnik:rust),
  Rust's library ecosystem is still evolving compared to `C++`. Notably, the
  lack of sophisticated native sparse linear algebra solvers forced reliance on
  `C/C++` libraries for this project.
- *Learning Curve*: Rust's unique concepts (ownership, borrowing, lifetimes) and
  syntax present a steeper learning curve compared to some other languages
  @klabnik:rust.
- *Over-Engineering Risk*: The powerful type system can sometimes tempt
  over-engineering.


== General software architecture

Our primary goal is to model mathematical concepts faithfully, ensuring both
mathematical accuracy and code clarity. This aims to make the code
mathematically expressive and self-documenting for those familiar with the
underlying theory. While prioritizing mathematical rigor, we also recognize the
importance of good API design and performance. Data structures are designed with
efficiency and memory economy in mind, leveraging Rust's capabilities for
performance-critical computing.

=== Modularity

FEM libraries are typically large software projects with distinct components. To
promote reusability (e.g., using the mesh component in other applications) and
maintainability, our library is split into multiple crates built upon each
other, managed within a Cargo workspace @RustLang.

The core crates are:
- `common`: Shared utilities and basic types.
- `manifold`: Topological and geometrical mesh data structures.
- `exterior`: Exterior algebra data structures.
- `ddf`: Discrete differential forms consisting of Cochains and Whitney forms.
- `formoniq`: The main library assembling components and providing FEEC solvers.

All of these have been published to `crates.io` @RustLang.

The chapters of the thesis parallel the structure and outline of these libaries.


== External libraries

We briefly discuss the major external libraries used:

=== Nalgebra (linear algebra)

Numerical algorithms heavily rely on linear algebra libraries. Rust's `nalgebra`
@crate:nalgebra provides an equivalent to `C++`'s Eigen, using generics effectively
for static and dynamic matrix dimensions. It forms the foundation for nearly all
numerical value manipulation in our library. We also utilize its sparse matrix
capabilities via `nalgebra-sparse`.

=== PETSc & SLEPc (sparse solvers)

Due to the aforementioned immaturity in Rust's native sparse solver ecosystem,
we utilize PETSc @PETScManualRecent, @PETScManual1997, a comprehensive `C/C++`
library suite, for solving large sparse linear systems (specifically using its
direct solvers). For the associated eigenvalue problems, we use SLEPc
@SLEPcPaper2005, which builds upon PETSc.

To avoid having PETSc and SLEPc as depedencies, since they are very big, we
decoupled them by not using them through Rust bindings or FFI.
Instead we have a seperate very small PETSc/SLEPc solver program
that interfaces with our Rust program through file IO.
This keeps the build step of formoniq simple and nice.

=== Itertools (combinatorics)

Itertools @crate:itertools is a utility crate extending Rust's standard iterators
with additional adaptors and methods. We use it primarily for combinatoric
algorithms (permutations, combinations) essential for mesh topology and exterior
algebra operations.

=== IndexMap (ordered HashSet)

For our skeleton data structure that contains simplicies, we need a
bidirectional map, between simplicies and indicies. For this we use a
`indexmap::IndexSet`, which is like a `HashSet` but also with an index. This
is provided by the `indexmap` crate @crate:indexmap. This crate provides map
and set data structures that maintain insertion order, which is crucial for
ensuring consistent global numbering and reproducible results in our mesh data
structures.

=== Rayon (parallelism)

To leverage multi-core processors and accelerate computationally intensive
tasks, we utilize the `rayon` crate @crate:rayon. Rayon provides easy-to-use data
parallelism capabilities for Rust, allowing iterators to be processed in
parallel with minimal code changes. In our library, `rayon` is employed to
parallelize the assembly loop, where element matrices computed independently for
each cell are summed into the global sparse matrices, significantly speeding up
this process on multi-core machines.

=== mshio (gmsh imports)

To import mesh data generated by the popular mesh generator Gmsh @GmshPaper2009,
we use the `mshio` Rust crate @crate:mshio, which handles parsing of the `.msh`
file format. We will use it once in our mesh implementation.

