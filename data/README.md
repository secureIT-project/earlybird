# Datasets

We fine-tune and test the CodeBERT model using the EarlyBIRD approach on 
four datasets. We use publicly available datasets and provide copies for 
convenience. Below you'll find pointers to their respective papers and 
sources. These data are provided under the license terms set by the 
original authors.


* [Devign](Devign) - This dataset contains functions in C/C++ from two 
  open-source projects labelled as vulnerable or non-vulnerable. We
  reuse the train/validation/test split from the 
  [CodeXGLUE Defect detection benchmark](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).
  The dataset is balanced: the ratio of non-vulnerable functions is 54%.

  * Paper: Y. Zhou, S. Liu, J. Siow, X. Du, and Y. Liu. “Devign: 
    Effective Vulnerability Identification by Learning Comprehensive
    Program Semantics via Graph Neural Networks.” In: International
    Conference on Neural Information Processing Systems (NeurIPS), 2018.

  * Data: <https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view> (link taken from the [CodeXGLUE Defect detection benchmark](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection); train/valid/test split: <https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection/dataset>

  * License: Creative Commons Zero v1.0 Universal (CC0 1.0 Universal)


* [ReVeal](ReVeal) - Similarly to Devign, ReVeal is a vulnerability
  detection dataset of C/C++ functions. The dataset is not balanced: 
  it contains 90% non-vulnerable code snippets. Both the Devign and
  ReVeal datasets contain real-world vulnerable and non-vulnerable
  functions from open-source projects.
  
  * Paper: S. Chakraborty, R. Krishna, Y. Ding, and B. Ray. “Deep 
    Learning Based Vulnerability Detection: Are We There Yet.” In: 
    IEEE Transactions on Software Engineering (2021). 
    doi: [10/gk52qr](https://doi.org/10/gk52qr).

  * Data: <https://github.com/VulDetProject/ReVeal>

  * License: MIT License


* [BIFI](BIFI) - The Break-It-Fix-It (BIFI) dataset contains function-
  level code snippets in Python with syntax errors. We use the original
  buggy functions and formulate a task of classifying the code into
  three classes: Unbalanced Parentheses with 43% of the total number 
  of code examples in BIFI, Indentation Error with 31% code samples,
  Invalid Syntax containing 26% samples. The provided train/test split
  is reused, and the validation set is extracted as 10% of train data.

  * Paper: M. Yasunaga and P. Liang. “Break-It-Fix-It: Unsupervised 
    Learning for Program Repair.” In:  In: International Conference 
    on Machine Learning, 2021,
    
  * Data: <https://github.com/michiyasunaga/BIFI> under p. 1, `orig_bad_code` 

  * License: MIT License

  
* [Exception Type](Exception Type) - The dataset consists of short
  functions in Python with an inserted HOLE token in place of one
  exception in code. The task is to predict one of 20 masked exception
  types for each input function and is unbalanced. The dataset was 
  initially created from the 
  [ETH Py150 Open corpus](https://www.sri.inf.ethz.ch/py150) as 
  described in the original paper. We reuse the train/validation/test
  split provided by the authors.

  * Paper: A. Kanade, P. Maniatis, G. Balakrishnan, and K. Shi. 
    “Learning and Evaluating Contextual Embedding of Source Code.” 
    In: Proceedings of the 37th International Conference on Machine
    Learning. PMLR, Nov. 2020, pp. 5110–5121
    
  * Data: <https://github.com/google-research/google-research/tree/master/cubert>

  * License: Creative Commons Attribution 4.0 International License (CC-BY-4.0)

