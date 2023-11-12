# NLML

NLML is a github repository containing scripts which make performing methodologically correct machine learning analyses more easy. The scripts in this repository were initially developed to build person-specific and pooled prediction models for binge eating, binge drinking and alcohol use with ecological momentary assessment data. However, they can be used in all kinds of contexts.

## Getting started
As of yet, the functions have not been included in an R package. Therefore, you need to follow these steps to use the functions:

1. Download the functions you are interested in to a local folder
   
2. Source the functions: \
`setwd('/User/test/NLML')` \
`NLML_functions = list.files(pattern="*.R")` \
`sapply(NLML_functions,source,.GlobalEnv)` 


## Materials

Looking for information on how you can use the scripts? Consult the wiki!

[NLML WIKI](https://github.com/mikojeske/NLML/wiki/)
