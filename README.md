# Representation of BLISS Words in Potts Network

After creating our artificial language [BLISS](https://github.com/pirmoradian/BLISS), we represented the words in a Potts network in a distributed fashion. These scripts were used to generate the semantic representaiton of BLISS words in the network. 


## Code description

Main files:

* neurep_mywords_main.py: The main file, which you need to run to generate the word representations.
* cnst.py: The constant file, where you can set the input and output file names and parameters.

Auxiliary files that neurep_mywords_main.py uses:

* neurep_mywords.py: Contains all functions relevant to the process of neural representation of BLISS words in the Potts network. **The functions are well-documented in this module.**  
* prjUtil.py: Contains several utility functions for setting the right file names for the corresponding representation. Many txt files in this directory are seen in this module.
* blissplot.py: Plots functions for different file types.
* KLdiv.py: Calculates Kullback–Leibler divergence between two different distributions.
* blissutil.py: Contains utility functions for working with dictionary.    


## Generating word representation

1. Set the path parameters in the constant file cnst.py: 

- MAIN_PATH: the main path on your computer, where these scripts are stored
- SEMPATH: the path where the semantic representation of words are stored
- SYNPATH: the path where the syntactic representation of words are stored
- FPATH: the path where the full representation (semantic & syntatci) of words are stored

2. Set which neural representation you like to generate by setting **neu** in cnst.py:
- If neu = 'neurep': semantic representation. The output will be stored in SEMPATH. 
- If neu = 'syneurep': syntactic representation. The output will be stored in SYNPATH.
- If neu = 'fneurep': full representation, i.e., semantic and syntactic representation. It reads from the semantic and syntactic representations that you have already generated and stored in SEMPATH and SYNPATH, and integrates them to be full representation and store it in FPATH.

3. Run the main python file: 

```
python neurep_mywords_main.py
```

4. Find the final representation of all words which can be passed to the Potts network in:
FPATH/myWORDS_fneurep_nhf400nhf20_07rplcd.csv


## Abbreviations
- 'npl': plural nouns
- 'nsg': singular nouns
- 'vpl': plural verbs
- 'vsg': singular verbs
- 'adj': adjectives
- 'fwd': function words
- 'psg': singular proper nouns
- 'ppl': plural proper nouns
- 'n2v2afp': all words, i.e. singular/plural nouns/verbs, adjectives, function words, proper nouns

## Developer

This package is developed by [Sahar Pirmoradian](https://www.researchgate.net/profile/Sahar_Pirmoradian). If you need help or have questions, don't hesitate to get in touch.


## Citation

If you use this code, please cite the corresponding proceeding and abstract where the word representation was explained:

S. Pirmoradian, A. Treves, “Encoding words into a Potts attractor network”, Proceedings of the 13th Neural Computation and Psychology Workshop, 2013 ([pdf](https://github.com/pirmoradian/wordRepresentation/blob/master/Pirmoradian%2C%20Treves%20-%202013%20-%20Proceedings%20of%20the%2013th%20Neural%20Computation%20and%20Psychology%20Workshop%20(NCPW).pdf))

S. Pirmoradian, A. Treves, "A talkative Potts attractor neural network welcomes BLISS words", BMC Neuroscience, 13(Suppl 1):P21, 2012 ([pdf](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-13-S1-P21))

For getting a broader picture about the project, and how these word representations could be used, please see [my PhD Thesis](https://github.com/pirmoradian/BLISS/blob/master/Thesis/SaharPirmoradian-Thesis.pdf), where I trained a network on the BLISS language with the word representations generated using the above scripts.

## Acknowledgments

* This project was done under the supervision of Prof. Alessandro Treves, in SISSA, Trieste, Italy.

