#import "setup.typ": *
#show: thesis-template

//#show image: none

#preface-style[
  #include "title.typ"
  #include "introduction.typ"
  #include "toc.typ"
]

#body-style[
  #include "matter/software.typ"
  #include "matter/mesh.typ"
  #include "matter/exterior.typ"
  #include "matter/ddf.typ"
  #include "matter/fem.typ"
  #include "matter/laplacian.typ"
  #include "matter/results.typ"
  #include "matter/conclusion.typ"
]

#appendix-style[
  #include "appendix.typ"
]

#postface-style[
  #bibliography("bibliography.bib")

  #{
    set page(margin: 0cm)
  
    show heading: none
    heading(numbering: none)[Declaration of Originality]

    image("../res/declaration-originality.svg")
  }
]
