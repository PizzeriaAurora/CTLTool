# CTLTool: Refinement-Based CTL Specification Minimization

**CTLTool** is a Python toolkit developed to support the analysis and minimization of **temporal logic specifications**, particularly those written in **Computation Tree Logic (CTL)**. It implements the algorithms and theoretical framework presented in our ICSE submission, which introduces a novel method for identifying a **minimal sufficient set of properties** for system verification based on **refinement relations**.

---

## Abstract

In software and systems engineering, formal verification is crucial
for ensuring that system designs adhere to their specifications. As
systems evolve, the associated model and specification suite tend
to become more complex due to iterative development, feature
expansion and architectural changes. This growth increases the
computational and methodological demands of the verification pro-
cess. While prior work has extensively addressed model complexity
through techniques such as abstraction and compositional reason-
ing, comparatively little attention has been paid to the complexity
of the specifications themselves, particularly in terms of whether all
formalized properties are necessary for verification. This paper ad-
dresses this gap by introducing a formal framework for identifying
a minimal sufficient set of properties based on refinement relations.
Focusing on Computation Tree Logic, widely used due to its expres-
sive yet tractable branching-time semantics, we formally define
refinement relations among properties as well as their structural
characteristics. Based on this foundation, we develop an efficient
algorithm that computes the minimal set of properties sufficient for
verification. We evaluate our approach using datasets from both
academic and industrial case studies. Our results demonstrate that
the proposed method can reduce the size specifications by 70%
and accelerate verification processes by a factor of up to 30 times,
showcasing its practical potential in optimising verification work-
flows and guiding requirements engineering towards identifying
the minimal required set of temporal properties.
---

##  Features

- **Refinement checking** between CTL formulas
- **Minimality computation**: find a smallest sufficient subset of properties
- Support for **benchmark datasets** from academic and industrial case studies
- CLI and Python API for integration in verification pipelines

---

## 📦 Installation

### Prerequisites

- Python 3.10 or later

### Setup

We recommend creating a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then install 
```
git clone https://github.com/PizzeriaAurora/CTLTool.git
cd CTLTool
pip install -r requirements.txt
```


