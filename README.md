[![](https://img.shields.io/badge/doi-10.2196/21798-yellow.svg)](https://doi.org/10.2196/21798)

# AutoScore-Python

AutoScore is a novel machine learning framework to automate the development of interpretable clinical scoring models. 
AutoScore consists of six modules: 1) variable ranking with machine learning, 2) variable transformation, 3) score derivation, 4) model selection, 5) domain knowledge-based score fine-tuning, and 6) performance evaluation.
The original AutoScore structure is elaborated in [this article](http://dx.doi.org/10.2196/21798).
AutoScore was originally designed for binary outcomes and later extended to [survival outcomes](http://dx.doi.org/10.1016/j.jbi.2021.103959) and [ordinal outcomes](https://doi.org/10.48550/arxiv.2202.08407). 
AutoScore could seamlessly generate risk scores using a parsimonious set of variables for different types of clinical outcomes, which can be easily implemented and validated in clinical practice. 
Moreover, it enables users to build transparent and interpretable clinical scores quickly in a straightforward manner.

AutoScore was originally implemented in R, available at [this GitHub repository](https://github.com/nliulab/AutoScore).
Here we present the Python implementation of AutoScore for clinical data with binary outcomes.

Package Installation
---

* Install AutoScore from this GitHub repository
```
pip install git+https://github.com/nliulab/AutoScore-Python
```
* Import AutoScore package in your own code
```
import AutoScore
```

Usage
---

The five pipeline functions constitute the 5-step AutoScore-based process for generating point-based clinical scores for binary outcomes.

This 5-step process gives users the flexibility of customization (e.g., determining the final list of variables according to the parsimony plot, and fine-tuning the cutoffs in variable transformation):

* STEP (i) `AutoScore_rank()` - Rank variables with machine learning (AutoScore Module 1)
* STEP (ii) `AutoScore_parsimony()` - Select the best model with parsimony plot (AutoScore Modules 2 + 3 + 4)
* STEP (iii) `AutoScore_weighting()` - Generate the initial score with the final list of variables (Re-run AutoScore Modules 2 + 3)
* STEP (iv) `AutoScore_fine_tuning()` - Fine-tune the score by revising cut_vec with domain knowledge (AutoScore Module 5)
* STEP (v) `AutoScore_testing()` - Evaluate the final score with ROC analysis (AutoScore Module 6)

We also include several optional functions in the package, which could help with data analysis and result reporting.

To test our code on demo datasets, run test scripts in the `examples` directory. For example,
```
python test_sample_data.py # For a large dataset with 20,000 samples
python test_sample_small.py # For a small dataset with 1,000 samples
python test_sample_missing.py # For a large dataset with missing values
```
These test scripts also serve as examples for using AutoScore in Python.

Full tutorials (for the original R package) are available at our [bookdown page](https://nliulab.github.io/AutoScore/).

Citation
---

* Xie F, Chakraborty B, Ong MEH, Goldstein BA, Liu N. [AutoScore: A machine learning-based automatic clinical score generator and its application to mortality prediction using electronic health records](http://dx.doi.org/10.2196/21798). JMIR Medical Informatics 2020; 8(10): e21798.

Contact
---

-   Feng Xie (Email: <xief@u.duke.nus.edu>)
-   Yilin Ning (Email: <yilin.ning@duke-nus.edu.sg>)
-   Qiming Wu (Email: <wuqiming@duke-nus.edu.sg>)
-   Siqi Li (Email: <siqili@u.duke.nus.edu>)
-   Nan Liu (Email: <liu.nan@duke-nus.edu.sg>)

