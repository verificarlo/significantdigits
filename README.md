# sigdigits package

Compute the number of significant digits based on the paper [Confidence Intervals for Stochastic Arithmetic](https://arxiv.org/abs/1807.09655).

# Example

`compute_sig.py` test significant digits computation accross different 
methods and precision for the Cramer test. 
You can test it on the provided file `cramer.npy`
containing 100 executions of `cramer.py` within [fuzzy](https://github.com/gkiar/fuzzy) environment.
Simply use `python3 compute_sig.py cramer.npy`

