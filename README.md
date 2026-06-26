# [ICSE 2027] Hone: Adaptive System-Level Orchestration for Federated Intelligent Software Systems
This is the ICSE 2027 *Hone: Adaptive System-Level Orchestration for Federated Intelligent Software Systems* GitHub repository.

## System Code
Currently, to verify the project's authenticity and reproducibility, we release a portion of the code here; please refer to the __*code/*__ folder. To protect copyright, we hide the core Python files and some of the interface functionalities in the encapsulation. Thank you for your understanding. We specifically performed the following operations:
- `component_libs.py`: The components' structure used in the Hone framework; the entire file is hidden.
- `hone_utils.py`: The function utils required by Hone; the entire file is hidden.
- `fl_algs.py`: The core algorithm for Hone is temporarily hidden.
- `parameterParse.py`: The parameters interface for Hone is temporarily hidden.
- `main.py`: The function call interface is temporarily hidden.
The complete code will be released after the paper is published.

## Table of Contents
* Parameters and Model Configurations
* Algorithm Process
* Experimental Results and Server Runtime Logs
* Mathematical Foundations and Theoretical Analysis
* Figure

### Parameters and Model Configurations
To facilitate verification or reproduction, we summarize the default hyperparameter settings on the five datasets. These are not the absolute best parameters, which means that the parameter values may vary slightly across different RQs, and that the parameters that achieve the fastest convergence are not necessarily the same as the parameters that produce the best accuracy for a given task. However, we ensure that these are valid parameters for most tasks and can be used directly for validation. We summarize the dataset processing, the corresponding model construction, and the details of the baselines. Please refer to __*supplementaryDoc.pdf - Section A & B*__ for details.

### Algorithm Process
For the algorithm process of Hone, please refer to __*supplementaryDoc.pdf - Section C*__ for details.

### Experimental Results and Server Runtime Logs
Please refer to __*resultsAndServerRunningLogs.zip*__ to get all the server runtime logs and raw experimental results conducted in the paper. The results are given in *.pkl* format according to the experiment type.

### Mathematical Foundations and Theoretical Analysis
For the mathematical foundations and theoretical analysis of this paper, please refer to __*supplementaryDoc.pdf - Section D*__ for details. Specifically, it is the theoretical analysis and formula derivation of the theoretical guarantees of the Hone methodology.

### Figure
Please refer to the __*figure/*__ folder to obtain all the figures used in this paper.
