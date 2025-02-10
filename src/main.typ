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
  //= Rust Source Code
  //= Typst Source Code
]

#postface-style[
  #bibliography("bibliography.yaml")
  //= Glossary
  //= Declaration of originality
]
