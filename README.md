# ML-DataTransformationIS PS for multi-label by means of Data Transformation methods
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
* RA*k*EL-IS: Random _k_-labelsets. Can be used with all LP-x listed above.
* MLeNN: Charte, F., Rivera, A. J., del Jesus, M. J., & Herrera, F. (2014, September). MLeNN: a first approach to heuristic multilabel undersampling. _In International Conference on Intelligent Data Engineering and Automated Learning_ (pp. 1-9). Springer International Publishing.
* MLENN: Kanj, S., Abdallah, F., Denœux, T., & Tout, K. (2016). Editing training data for multi-label classification with the k-nearest neighbor rule. _Pattern Analysis and Applications_, 19(1), 145-161.

## Citation policy
 **Á. Arnaiz-González, J-F. Díez Pastor, Juan J. Rodríguez, C. García Osorio.** _Study of data transformation techniques for adapting single-label prototype selection algorithms to multi-label learning._ Expert Systems with Applications, 109, 114-130. [doi: 10.1016/j.eswa.2018.05.017](https://doi.org/10.1016/j.eswa.2018.05.017)

```
@article{ArnaizGonzalez2018,
  title = "Study of data transformation techniques for adapting single-label prototype selection algorithms to multi-label learning",
  journal = "Expert Systems with Applications",
  volume = "109",
  pages = "114 - 130",
  year = "2018",
  issn = "0957-4174",
  doi = "10.1016/j.eswa.2018.05.017",
  author = "\'{A}lvar Arnaiz-Gonz\'{a}lez and Jos\'{e} F. D\'{i}ez-Pastor and Juan J. Rodr\'{i}guez and C\'{e}sar Garc\'{i}a-Osorio"
}
```
The paper is available online and accessible for free until July 15, 2018 through this link https://authors.elsevier.com/c/1X6pt3PiGT7gPl

## Contributions
Some of the algorithms has been adapted to MEKA by means of a wrapper, their original source codes are available here:

* MLeNN: http://simidat.ujaen.es/papers/mlenn

