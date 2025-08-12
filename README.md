# Gelu-Optimization-Research

CachedGELU implements a high-accuracy, table-based approximation of the GELU activation. It replaces expensive erf calls with a precomputed lookup table and linear interpolation to speed up inference  while keeping numerical error negligible.



# Methodology

CachedGELU accelerates GELU evaluation by precomputing activation values over a chosen range (default: −10 to 10) into a lookup table with a fixed number of evenly spaced grid points. For each input during inference:

The input is mapped to a fractional index in the table.

Fast first‐order linear interpolation is applied using precomputed neighbor slopes.

Out‐of‐range inputs fall back to the exact GELU formula for correctness.

This method eliminates expensive erf calls for in‐range inputs, offering constant‐time O(1) evaluation per element with minimal approximation error. The trade‐off between memory use and accuracy is adjustable via the table size N, and vectorized NumPy operations maximize runtime efficiency.(CPU Version)
