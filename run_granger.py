import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings

# Suppress specific FutureWarning from statsmodels
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


def test_granger(data, gene_pair, maxlag=1, significance_level=0.05):
    gene_x, gene_y = gene_pair
    # Prepare data subset for the pair
    data_pair = data[[gene_x, gene_y]].dropna()

    if (
        data_pair.shape[0] > maxlag + 1
    ):  # Ensure there's enough data to perform the test
        result = grangercausalitytests(data_pair, maxlag=maxlag, verbose=False)
        p_values = [result[i + 1][0]["ssr_ftest"][1] for i in range(maxlag)]
        if min(p_values) < significance_level:
            return (gene_x, gene_y)
    return None


def run_granger_tests(df_time_series, maxlag=1):
    gene_list = df_time_series.columns.tolist()
    gene_pairs = list(combinations(gene_list, 2))

    pool = mp.Pool(processes=mp.cpu_count())
    func = partial(test_granger, df_time_series, maxlag=maxlag)

    results = []
    with tqdm(total=len(gene_pairs)) as pbar:
        for result in pool.imap(func, gene_pairs):
            if result is not None:
                results.append(result)
            pbar.update()
    pool.close()
    pool.join()

    return results


if __name__ == "__main__":
    # Assuming df_time_series is already loaded and prepared
    # scGREAT data, probably filtered out the genes with minimum difference in expression
    expr_df = pd.read_csv("/Users/factored/Dev/scGREAT/hESC500/BL--ExpressionData.csv")
    pseudotime_df = pd.read_csv(
        "/Users/factored/Dev/BEELINE/BEELINE-data/inputs/scRNA-Seq/hESC/PseudoTime.csv"
    )
    expr_df.set_index("Unnamed: 0", inplace=True)
    pseudotime_df.set_index("Unnamed: 0", inplace=True)

    # Sort by pseudotime
    df_expression = expr_df.T
    df_expression["PseudoTime"] = pseudotime_df["PseudoTime"]
    df_expression.sort_values("PseudoTime", inplace=True)
    df_expression.drop("PseudoTime", axis=1, inplace=True)

    df_time_series = df_expression
    significant_gene_pairs = run_granger_tests(df_time_series, maxlag=1)

    # save the results as object
    np.save("significant_gene_pairs.npy", significant_gene_pairs)
    breakpoint()
    print("Finished")
