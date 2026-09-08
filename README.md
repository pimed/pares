# pares
This is the inference code for pAres, the tool made available as part of the study titled:
 "Aggressive Cancer Subtype Detection on Prostate MRI via Radiology-Pathology-Genomic Learning" by Rusu. et. al. 2026 that is currently under review.

The code shows how to run the inference in pAres for a case from the Chimera challenge cohort.


Approach summarized here:  
![Overall Approach](images/Figures_Page_01.png)

# How to run

## Download the model weights (only needed once)

The models will be made available upon manuscript acceptance [here](https://stanfordmedicine.box.com/s/95a0n0jfxz8ksizylv4puf75bw177esx). 
Accession denied messages indicates that you don't have access to download the weights at this time. 
Please send a data access request to [mrusu@stanford.edu](mailto:mrusu@stanford.edu).  

## Unzip the model weights

`$ mkdir models`

`$ cp path\to\models\*.tgz models` or `$ cp path\to\models\*.tar models` 

`$ tar -xvzf Dataset202.tar; tar -xvzf Dataset203.tar; tar -xvzf Dataset361.tar; tar -xvzf Dataset309.tar; cd ../`

# Execute unit test

$`python test_one_case.py`