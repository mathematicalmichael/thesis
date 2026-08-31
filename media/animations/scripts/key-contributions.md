# What My PhD Actually Contributed

> Three contributions of the dissertation: data-consistent inversion, the MUD point for parameter identification, and open-source scientific software.

- **Composition:** `KeyContributions`
- **Format:** 1080×1920 @ 30 fps (YouTube Short)
- **Runtime:** 1:00.0

## Voiceover script

Read each block during its time window. The **caption** column is what
appears on screen (burned in); speak the **voiceover** text naturally —
the windows include breathing room.

| time | caption (on screen) | voiceover (read this) |
| --- | --- | --- |
| 0:00.0–0:05.0 | Years of a PhD → 3 contributions | I spent years of my life on one question: how do you turn noisy data into honest knowledge about the things you cannot observe directly? Here are the three things my dissertation actually contributed. |
| 0:05.0–0:20.7 | 1 · Data-Consistent Inversion | First: data-consistent inversion. Your model pushes a distribution of parameters forward into predictions. Data-consistent inversion pulls the observed distribution back onto the parameters — using measure theory, not Bayes' rule. The guarantee: push the answer back through the model and you reproduce exactly the distribution you observed. Bayesian posteriors don't promise that. Data-consistent solutions do — by construction. |
| 0:20.7–0:37.3 | 2 · One truth, noisy data → the MUD point | Second — and this is the core of the thesis — I extended that framework to the classic problem where a single true parameter hides behind noisy data. By pulling back distributions of residuals, you get the maximal updated density point: the MUD point. It matches the accuracy of least squares, but unlike the Bayesian MAP point, it is not dragged around by how strongly you state your prior beliefs. No hyper-parameter tuning required. |
| 0:37.3–0:50.0 | 3 · Open-source software for UQ | Third: a beautiful method nobody can run is a method nobody will use. So a serious chunk of the dissertation is software: open-source Python packages, fully reproducible experiments — even the document itself rebuilds from source with one command in Docker. |
| 0:50.0–1:00.0 | Rigorous math · honest uncertainty · runnable code | Rigorous mathematics, honest uncertainty, and code you can run tonight. The full dissertation and every experiment in it are linked below — and this is the first of a whole series of animations unpacking it. |

## Plain read-through

I spent years of my life on one question: how do you turn noisy data into honest knowledge about the things you cannot observe directly? Here are the three things my dissertation actually contributed.
First: data-consistent inversion. Your model pushes a distribution of parameters forward into predictions. Data-consistent inversion pulls the observed distribution back onto the parameters — using measure theory, not Bayes' rule. The guarantee: push the answer back through the model and you reproduce exactly the distribution you observed. Bayesian posteriors don't promise that. Data-consistent solutions do — by construction.
Second — and this is the core of the thesis — I extended that framework to the classic problem where a single true parameter hides behind noisy data. By pulling back distributions of residuals, you get the maximal updated density point: the MUD point. It matches the accuracy of least squares, but unlike the Bayesian MAP point, it is not dragged around by how strongly you state your prior beliefs. No hyper-parameter tuning required.
Third: a beautiful method nobody can run is a method nobody will use. So a serious chunk of the dissertation is software: open-source Python packages, fully reproducible experiments — even the document itself rebuilds from source with one command in Docker.
Rigorous mathematics, honest uncertainty, and code you can run tonight. The full dissertation and every experiment in it are linked below — and this is the first of a whole series of animations unpacking it.

---

_This file is generated from the composition source. Do not edit by hand:_
_run `npm run scripts:export` after changing `script.ts`._
