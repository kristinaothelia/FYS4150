# How to run the code
---------------------

### **The help message:**

usage: prosjekt1.py [-h] [-t | -s | -l] [-a [A]] [-b [B]] [-c [C]] [-n [N]] [-E]

Project 1 in FYS4150 - Computational Physics

optional arguments:
  -h, --help     show this help message and exit
  -t, --thomas   Thomas solver
  -s, --special  Special solver
  -l, --LU       Lower Upper decomposition
  -a [A]         value beneath the diagonal
  -b [B]         value for the diagonal
  -c [C]         value above the diagonal
  -n [N]         value for the NxN matrix
  -E             calculates the relative error (usage: thomas solver)

--------------------------------------------------------------
### **Examples of how to run the code with the thomas solver**

#### Runs with default values
* python prosjekt1.py -t           

#### Runs with a=4 and b=5, c and n still default
* python prosjekt1.py -t -a 4 -b 5  

#### Runs with n=10 and calculates relative errors
* python prosjekt1.py -t -n 10 -E