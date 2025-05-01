#import "setup.typ": *

#page[
  #{
    show heading: none
    //heading(numbering: none)[Title]
    heading()[Title]
  }

  
  #image("../res/ethz-logo.svg")

  #set align(center + horizon)

  #[
    #set text(size: 20pt)
    Bachelor's Thesis
  ]
  #v(1cm)

  #[
    #set text(size: 25pt, weight: "bold")
    Rust Implementation of \
    Finite Element Exterior Calculus on \
    Coordinate-Free Simplicial Complexes \
  ]

  #block(
    stroke: ("left": 0.5pt, "right": 0.5pt),
    fill: black.lighten(90%),
    inset:5pt
  )[
    *Abstract*\
    #set align(left)
    This thesis presents the development of a novel finite element library in
    Rust based on the principles of Finite Element Exterior Calculus (FEEC). The
    library solves partial differential equations formulated using differential
    forms on abstract, coordinate-free simplicial complexes in arbitrary
    dimensions, employing an intrinsic Riemannian metric derived from edge
    lengths via Regge Calculus. We focus on solving elliptic Hodge-Laplace
    eigenvalue and source problems on the $n$D de Rham complex. We restrict
    ourselves to first-order Whitney basis functions. The implementation
    is partially verified through convergence studies.
  ]

  #v(1cm)
  #text(size: 20pt, style: "italic")[
    Luis Wirth
  ]
  \
  #weblink("mailto:luwirth@ethz.ch", "luwirth@ethz.ch") \
  #weblink("http://ethz.lwirth.com", "ethz.lwirth.com")
  
  #v(0.5cm)
  Supervised by\
  #text(size: 15pt, style: "italic")[
    Prof. Dr. Ralf Hiptmair
  ]

  #v(0.75cm)
  #text(size: 15pt)[
    1st May 2025
    //#datetime.today().display("[day]th [month repr:long] [year]").
  ]

]
