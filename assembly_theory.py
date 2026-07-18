import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def quantise_returns(returns, primitive_size=3):
    """
    Quantise returns into a discrete alphabet of size primitive_size.
    """
    if len(returns) < 2:
        return np.array([])
    # Use quantiles for equal-frequency binning
    quantiles = np.linspace(0, 100, primitive_size + 1)[1:-1]
    bins = np.percentile(returns, quantiles)
    if len(np.unique(bins)) == 1:
        bins = np.linspace(returns.min(), returns.max(), primitive_size + 1)[1:-1]
    discrete = np.digitize(returns, bins)
    return discrete

def assembly_index(sequence, primitive_size=3, max_steps=20):
    """
    Compute the assembly index of a discrete sequence.
    The assembly index is the minimum number of steps to construct the sequence
    from elementary building blocks using concatenation and copying.
    """
    if len(sequence) < 2:
        return len(sequence)
    # Use the Lempel-Ziv complexity as a proxy for assembly index
    # LZ complexity counts the number of distinct substrings
    # Assembly index is related to the size of the smallest grammar
    # We'll use a simplified version: the number of distinct substrings
    # plus the length of the sequence
    substrings = set()
    for i in range(len(sequence)):
        for j in range(i+1, min(i+max_steps, len(sequence))+1):
            substrings.add(tuple(sequence[i:j]))
    # Assembly index = number of distinct substrings + length
    # This captures the structural complexity
    # Higher value = more complex / more structured
    assembly = len(substrings) + len(sequence)
    return assembly

def assembly_index_score(returns, macro_factor, primitive_size=3, max_steps=20):
    """
    Compute the assembly index of the return sequence.
    Higher assembly index = more structured / predictable.
    """
    if len(returns) < 5:
        return 0.0
    # Quantise returns
    discrete = quantise_returns(returns, primitive_size)
    if len(discrete) < 3:
        return 0.0
    # Compute assembly index
    assembly = assembly_index(discrete, primitive_size, max_steps)
    # Scale by macro factor
    assembly = assembly * (1 + macro_factor * 0.5)
    return float(assembly)

def assembly_theory_score(returns, macro_df, primitive_size=3, max_steps=20):
    """
    Compute per-ETF assembly index score.
    Higher score = more structured / predictable.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 10:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Compute assembly index
    assembly = assembly_index_score(returns, macro_factor, primitive_size, max_steps)
    return float(assembly)
