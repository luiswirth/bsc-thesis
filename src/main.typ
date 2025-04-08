#import "setup.typ": *
#show: thesis-template

#preface-style[
  #include "title.typ"
  #include "introduction.typ"
  #include "toc.typ"
]

#body-style[
  #include "matter.typ"
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
