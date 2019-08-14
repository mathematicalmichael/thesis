

# chapter_02.tex
- [ ] sec:results (314)
- [ ] eq:objective (320)
- [ ] sec:metrics (334)
- [ ] sec:results (335)
- [ ] [TK - cite]
- [X] Example: 2D example with 1/10 sidelength in center, identity map.
  - [ ] just showcase the effect of finite approximation
  - [ ] write singular script to make this example and all the tables for it,
    set up a directory for it (no tables, in progress with notebook.)
- [ ] Sampling Section
  - [ ] discuss sample-based inversion for measures (2.3)
  - [ ] numerical approximation and analysis
  - [ ] descriptions of error
      - [ ] show identity example (showcase analytical BET vs approximate)
- [ ] Software Section
  - [ ] flesh out (break up into subsections)
- [ ] Examples Section
  - [ ] citations and sections messed up
  - [X] hellinger references need to be removed
  - [ ] exponential decay with sampling and set-based (no parameter ID)
  - [ ] heat rod (where is the code for this?)
      - [ ] recreate figures
      - [ ] remove references to skewness
      - [ ] segment out into its own tex file

# exponential_decay.tex
- [ ] h float specifier changed to ht

# chapter03.tex
- [ ] sec:results (22)
- [ ] sec:results (70)
- [ ] fix "N" to \nsamps

- [ ] Accuracy of Set-Based Inversion:
  - [X] find latex file that generated these images (fzf!)
  - [X] python file that writes images to file, bigger labels

  - [ ] algorithm 2: get rid of hellinger
  - [ ] get rid of hellinger everywhere.
  - [ ] Examples:
      - [ ] rotational invariance
          - [ ] file to generate data, figures, etc.
          - [ ] rotation folder (model) in examples
      - [ ] skewness map
          - [ ] skew model folder in examples
          - [ ] redo figures but prettier

- [ ] Accuracy of Sample-Based
  - [ ] all examples should output both methods
  - [ ] A sample-based solution:
      talk. talk.
  - [ ] rotational, skewness.

- [ ] Software Contributions
  - [ ] Comparison module
  - [ ] RANT! EASY TO START HERE.
      - [ ] talk about timeline
      - [ ] how it works/architecture

- [ ] Numerical Results and Analysis
  - [ ] THE FOCUS IS ON INCREASING N.
  - [ ] HERE YOU SHOWCASE THE METRICS CODE SYNTAX
  - [ ] previous exp/heatrod example just showed what the solutions looked like for some choice of N. now we study refining our space (accuracy of solutions).

  - [ ] convergence with repeated
      - [ ] is this about using data to estimate average? (build on ID example by saying our model was hit with error)
      - [ ] or is this about iterated ansatz?
  - [ ] nonlinear
      - [ ] exponential decay
      - [ ] move heatrod here??

# skew_example_3d.tex
- [ ] h float

- [ ] \headheight too small (12pt) ??? set to 12.05? Don't understand.


# chapter 4 data driven Maps
- [ ] missing pretty much everything. can start filling in theory based on the poster content.
- [X] stochastic framework figure
- [ ] start a rant about the figure

# other
- [ ] find the table-generating scripts
- [ ] measuresets.tex (proof of Lemma about computational sa)
- [ ] skew_example_3d.tex (dependence on dimension)
  - [ ] needs notational changes
- [ ] rotation_example.tex (move, update)
- [ ] skew_example.tex (move, update)
- [ ] clean up notation.tex
- [X] heat_1drod_set_vs_sample.tex (remove extra)
- [ ] experimental_setup.tex
- [ ] usepackages.tex (move stuff into it?)
- [ ] abstract.tex
