= Software Design & Implementation Choices

In this chapter we want to briefly discuss some general software engineering
decisions for our library.

== Rust

We have chosen Rust @RustLang as the main programming language for the implementation of
our Finite Element library.
There are various reasons for this choice, some of which we briefly outline
here.

=== Memory Safety + Performance

Rust is a strongly-typed, modern systems programming language that combines
performance on par with `C/C++` with strong memory safety guarantees @RustLang.
Unlike traditional memory-safe languages that rely on garbage collection,
Rust achieves memory safety through a unique approach inspired by formal
software verification and static analysis techniques, ensuring safety
at compile-time therefore not compromising performance.

The Rust compiler acts as a proof checker, requiring the programmer to provide
sufficient evidence for the safety of their code.
This is accomplished by extending Rust's strong type system with an
ownership and borrowing model that enforces clear semantics regarding
memory responsibility @RustLang.
This system completely eliminates an entire class of memory-related bugs, making
software significantly more reliable.

Not only does this model guarantee the absence of bugs such as dangling
pointers, use-after-free, memory-aliasing violations, and null-pointer
dereferences, but it also extends to concurrent and parallel programming,
ensuring that data races can never occur.
This "fearless concurrency" feature allows developers to be fully confident that
any parallel code written in Rust that compiles will behave as expected @RustLang.

=== Expressiveness and Abstraction

Rust is a highly expressive language that enables powerful abstractions
without sacrificing performance, often referred to as zero-cost abstractions @RustLang.
This allows for a more direct realization of ideas and concepts, making Rust
particularly well-suited for capturing precise mathematical structures and
expressing complex logic in a natural way. Below, we highlight some of the
features that the author finds particularly valuable.

- *Traits and Generics*: Rust's trait system facilitates powerful
  polymorphism, enabling code reuse and extensibility without the drawbacks of
  traditional object-oriented inheritance @RustLang. Unlike classical inheritance-based
  approaches, traits define shared behavior without enforcing a rigid hierarchy.
  Rust’s generics are monomorphized at compile time, ensuring zero-cost
  abstraction while allowing for highly flexible and reusable code structures.
  This approach eliminates the notorious template-related complexities of `C++`,
  as trait bounds explicitly state required behaviors within function and struct
  signatures.

- *Enums, Option, and Result*: Rust provides algebraic data types,
  particularly the sum type `enum`, which acts as a tagged union @RustLang. This simple
  yet powerful form of polymorphism allows developers to express complex state
  transitions in a type-safe manner. The standard library includes two widely
  used enums: `Option` and `Result`. The `Option` type eliminates null-pointer
  exceptions entirely by enforcing explicit handling of absence. The `Result` type
  enables structured error handling without introducing exceptional control flow,
  leveraging standard function returns for working with errors in a predictable
  way.

- *Expression-Based Language and Pattern Matching*: Unlike many imperative
  languages, Rust is expression-based, meaning that most constructs return
  values rather than merely executing statements @RustLang. For example, an `if` expression
  evaluates to the value of its selected branch, eliminating redundant variable
  assignments. Combined with Rust’s powerful pattern matching system, which allows
  destructing of composite types like enums, this leads to concise and readable
  code while enabling functional-style composition.

- *Functional Programming and Iterators*: Rust embraces functional programming
  principles such as higher-order functions, closures (lambdas), and iterators @RustLang.
  The iterator pattern allows for efficient, lazy evaluation of collections,
  reducing unnecessary memory allocations and improving performance. Functional
  constructs such as `map`, `filter`, and `fold` are available as standard methods
  on iterators, promoting declarative and expressive coding styles.


Together, these features make Rust an expressive language, providing developers
with the tools to write concise, maintainable, and high-performance software. By
combining modern programming paradigms with low-level control, Rust ensures both
safety and efficiency, making it an excellent choice for scientific computing
and systems programming.


=== Infrastructure and Tooling

Beyond its language features, Rust also stands out due to its exceptional
infrastructure and tooling ecosystem, which greatly enhances the development
workflow @RustLang.

A key advantage is the official nature of all tooling, which reduces
fragmentation and prevents competing tools from creating confusion over choices,
in contrast to the `C++` ecosystem. This consistency fosters a more streamlined
and productive development experience.

- *Cargo* is Rust's official package manager and build system, which is one of
  the most impressive pieces of tooling @RustLang. It eliminates the need for traditional
  build tools like Makefiles and CMake, which are often complex and difficult
  to maintain—not to mention the dozens of other build systems for `C++`.
  Cargo simplifies dependency management through its seamless integration with
  `crates.io`, Rust’s central package repository @RustLang. Developers can effortlessly
  include third-party libraries by specifying them in the `Cargo.toml` file, with
  Cargo automatically handling downloading, compiling, and dependency resolution
  while enforcing semantic versioning. Publishing a crate is equally simple via
  `cargo publish`, which we have also used to distribute the libraries developed
  for this thesis.
- *Clippy* is Rust's official linter, offering valuable suggestions for
  improving code quality, adhering to best practices, and catching common
  mistakes @RustLang. Our codebase does not produce a single warning or lint, passing all
  default checks for code quality.
- *Rustdoc* is Rust's official documentation tool, allowing developers to
  write inline documentation using Markdown, seamlessly integrated with code
  comments @RustLang. This documentation can be compiled into a browsable website with
  `cargo doc` and is automatically published to `docs.rs` when a crate is uploaded
  to `crates.io`. The documentation for our libraries is also available there.
- *Rusttest* is the testing functionality built into Cargo for running unit
  and integration tests @RustLang. Unit tests can be placed alongside the normal source code
  with a simple `#[test]` attribute, and the `cargo test` command will execute
  all test functions, verifying correctness. This command also ensures that all
  code snippets in the documentation are compiled and checked for runtime errors,
  keeping documentation up-to-date without requiring external test frameworks like
  Google Test.
- *Rustfmt* standardizes code formatting, eliminating debates about code style
  and ensuring consistency across projects @RustLang. Our codebase fully adheres to Rustfmt's
  formatting guidelines. For conciseness however we will be breaking
  the formatting style when putting code inline into this document.

Together, Rust’s comprehensive tooling ecosystem ensures a smooth, efficient,
and reliable development experience, reinforcing its position as a robust choice
for scientific computing and large-scale software development.


There are many more good reasons to choose Rust, such as it's great ecosystem
of libraries, which are some of the most impressive libraries the author has ever seen.


=== Drawbacks

We want to also mentioned some drawbacks of using Rust and challenges we've
encoutered.

- Rust is a relatively young programming language, as it had it's 1.0 release
in 2015 @RustLang. Due to this the library ecosystem is still evolving and solutions
that are available in `C++`, do not yet exist for Rust. A particular instance
that affects us, is the absence of sophisticated sparse linear algebra implementation.
Only basic sparse matrix implementation are available, but for solvers, we
we're forced to rely on `C/C++` libraries.

- Rust has a high learning curve and has a non-standard syntax
  with many concepts, that might make it hard for people unfamiliar with the language
  to read and understand it @RustLang.

- One can become too obsessed with expressing concepts in the powerful type system,
  leading to over-engineering, which badly influences the project.
- Rust can become very verbose due to it's many abstraction features.


== External libraries

We want to quickly discuss here the major external libraries,
we will be using in our project.

=== nalgebra (linear algebra)

For implementing numerical algorithms linear algebra libraries are invaluable.
`C++` has set a high standard with `Eigen` as it's major linear algebra library.
Rust offers a very direct equivalent called `nalgebra` @NalgebraLib, which just as Eigen
relies heavily on generics to represent both statically and dynamically know
matrix dimensions.
All basic matrix and vector operations are available.
We will be using nalgebra all over the place, pretty much always we have to deal
with numerical values.

Sparse matrices will also be relevant in our library.
For this we will be using `nalgebra-sparse`.

=== PETSc & SLEPc (solvers)

Unfortunately the rust sparse linear algebra ecosystem is rather immature.
Only very few sparse solvers are available in Rust.
For this reason we will be using one of the big `C/C++` sparse matrix libraries
called PETSc @PETScManualRecent, @PETScManual1997. We will be using direct solvers.

For eigensolvers we will be using SLEPc @SLEPcPaper2005, which builds on top of PETSc.

== General software architecture

We aim to model mathematical concepts as faithfully as possible, ensuring both
mathematical accuracy and code clarity.
This renders the code mathematically expressive and self-documenting for those
with the necessary background.
While we do not shy away from mathematical complexity or oversimplify for
the sake of accessibility, we recognize the importance of good API design and
HPC principles. Our goal is to strike a balance between mathematical rigor,
usability, and performance.

=== Modularity

As is the nature with most FEM libraries, they are rather big pieces
of software. They consists of many parts with different responsibilities.
So of which are useable standalone, for instance the mesh could also be used
for a different application. For this reason we split up our FEM library
into multiple libraries than built on top of each other.

We rely on a Cargo workspace @RustLang to organize the various parts of our library ecosystem.

We have the following crates:
- common
- manifold
- exterior
- ddf
- formoniq

All of which have been published to `crates.io` @RustLang.

===  Type safety

The implementation has a big emphasis on providing safety through the introduction
of many types that uphold guarantees regarding the contained data.
Constructed instances of types should always be valid.

=== Performance considerations

All data structures are written with performance in mind.

We are also always focused on a memory-economic representation of information.
