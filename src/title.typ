#import "setup.typ": *

#page[
  #image("../res/ethz-logo.svg")

  #hide[#heading()[Title]]

  #align(center + horizon)[
    #[
      #set text(size: 25pt, weight: "bold")

      Arbitrary Dimension \
      Finite Element Exterior Calculus \
      Implementation in Rust
    ]

    #v(1cm)
    #text(size: 20pt, style: "italic")[
      Luis Wirth
    ]
    \
    #weblink("mailto:luwirth@student.ethz.ch", "luwirth@student.ethz.ch") \
    #weblink("https://ethz.lwirth.com", "ethz.lwirth.com")

    #v(0.75cm)
    #text(size: 15pt)[
      #datetime.today().display("[day]th [month repr:long] [year]").
    ]
  ]
]
