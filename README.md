# ML-DataTransformationIS Prototype Selection for multi-label by means of Data Transformation methods
Prototype selection algorithms for MEKA. The methods were adapted to multi-label learning by means of data transformation techniques. The implementations of the algorithms are not fully optimized and have some limitations, please review the header of each class.

The original MEKA software is necessary: https://github.com/Waikato/meka

* BR-CNN: Binary Relevance CNN, dependent BR is also available.
* BR-ENN: Binary Relevance ENN, dependent BR is also available.
* BR-RNG: Binary Relevance RNGE, dependent BR is also available.
* BR-LSS: Binary Relevance LSSm, dependent BR is also available.
* LP-CNN: Label powerset CNN.
* LP-ENN: Label powerset ENN.
* LP-RNG: Label powerset RNGE.
* LP-LSS: Label powerset LSSm.
* RA_k_EL-IS: Random _k_-labelsets. Can be used with all LP-x listed above.
* MLeNN: Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2014, September). MLeNN: a first approach to heuristic multilabel undersampling. _In International Conference on Intelligent Data Engineering and Automated Learning_ (pp. 1-9). Springer International Publishing.
* MLENN: Kanj, S., Abdallah, F., Denœux, T., & Tout, K. (2016). Editing training data for multi-label classification with the k-nearest neighbor rule. _Pattern Analysis and Applications_, 19(1), 145-161.

## Citation policy
The code was implemented by Álvar Arnaiz-González. The review process of the paper is in progress.

## Contributions
Some of the algorithms has been adapted to MEKA by means of a wrapper, their original source codes are available here:

* MLeNN: http://simidat.ujaen.es/papers/mlenn

