# How to run the code
---------------------

### **The help message:**

usage: prosjekt2.py [-h] [-B | -Q] [-e [E]]

Project 2 in FYS4150 - Computational Physics

optional arguments:  
  -h, --help     show this help message and exit  
  -B, --beam     The buckling beam problem  
  -Q, --quantum  Quantum mechanics  
  -e [E]         number of electrons (use 1 or 2)  

--------------------------------------------------------------
### **Examples of how to run the code**

#### The Buckling Beam problem
* python prosjekt2.py -B          

#### Quantum 3D (one electron)
* python prosjekt2.py -Q
* python prosjekt2.py -Q -e 1

#### Quantum 3D (two electrons)
* python prosjekt2.py -Q -e 2


To get the iteration vs. N plot and CPU times 
for the Buckling Beam problem, set optimal_values = True
and run the Buckling Beam problem.

To get the heatmap plot, set optimal_values = True
and run the Quantum 3D (one electron) problem. 