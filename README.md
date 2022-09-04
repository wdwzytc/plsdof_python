Calculate the degrees of freedom for PLS with the "Lanczos" method, with details in section 3.3, Krämer (2011) [10.1198/jasa.2011.tm10107]

This code is just a translation from R code to Python code. Original R code is here: https://github.com/cran/plsdof 

With the X and y given in this repository, the python code and the original R code gave the same result, but I am not sure it will be the same in every cases. The rules of matrix calculation has some differences in R and Python. Hope it also works for you.

the code run well on my computer with:

pandas == 1.3.4

numpy == 1.20.3

sklearn == 1.1.1

matplotlib == 3.5.2

