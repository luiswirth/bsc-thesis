#import "setup.typ": *

#page[
  #image("../res/ethz-logo.svg")

  #hide[#heading()[Title]]

  #align(center + horizon)[
    #[
      #set text(size: 25pt, weight: "bold")
      Rust Implementation of \
      Finite Element Exterior Calculus on \
      Coordinate-Free Simplicial Manifolds\
    ]

    // Abstract
    #block(stroke: ("left": 0.5pt, "right": 0.5pt), inset:5pt)[
      We develop a finite element library in Rust based on the principles of
      Finite Element Exterior Calculus to solve partial differential equations
      formulated in terms of differential forms on simplicial complexes, using
      an intrinsic, coordinate-free approach based on Regge Calculus.
      The focus will be on solving elliptic Hodge-Laplace problems  and linear
      Whitney forms basis functions.
    ]

    #v(1cm)
    #text(size: 20pt, style: "italic")[
      Luis Wirth
    ]
    \
    #weblink("mailto:luwirth@ethz.ch", "luwirth@ethz.ch") \
    #weblink("https://ethz.lwirth.com", "ethz.lwirth.com")

    
    #v(1cm)
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
]
