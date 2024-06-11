#import "setup.typ": *
#show: general-style

#preface-style[
  #include "title.typ"
  #include "abstract.typ"
  #include "toc.typ"
]

#body-style[
  #include "introduction.typ"
  #include "theory.typ"
  #include "implementation.typ"
]


#appendix-style[
  = Rust Source Code
  = Typst Source Code
]

#postface-style[
  #bibliography("bibliography.yaml")

  = Glossary
  = Declaration of originality
]
