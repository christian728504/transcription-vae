import numpy as np
import pandas as pd
import swifter
import os
import copy

import pybigtools
import json
from multiprocessing import Pool
import umap
from umap.parametric_umap import ParametricUMAP
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
import pickle
import gc

def multiprocessing_list(func, argument_list, num_processes):
    """Run a function in parallel and return results as a dictionary."""
    with Pool(processes=num_processes) as pool:
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
            result_list_tqdm.append(result)
            
    return result_list_tqdm

def multiprocessing_dict(func, argument_list, num_processes):
    """Run a function in parallel and return results as a dictionary."""
    result_dict = {}
    with Pool(processes=num_processes) as pool:
        for rDHS, signals in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
            result_dict[rDHS] = signals
    return result_dict

def process_ccre_data(
    input_file="/data/projects/encode/Registry/V4/GRCh38/Cell-Type-Specific/Individual-Files/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed", 
    output_file="ccre_df.tsv"
):
    """
    Process K562-specific cCREs, filtering for pELS, dELS, or PLS types.
    """
    main_chromosomes = [f"chr{i}" for i in range(1, 23)]

    ccre_df = pd.read_csv(input_file, sep='\t', header=None, engine='python')
    filtered_ccre_df = ccre_df[ccre_df[9].isin(['PLS', 'dELS'])]
    filtered_ccre_df = filtered_ccre_df[filtered_ccre_df[0].isin(main_chromosomes)]
    result_df = pd.DataFrame({
        'chrom': filtered_ccre_df[0],
        'start': filtered_ccre_df[1],
        'end': filtered_ccre_df[2],
        'rDHS': filtered_ccre_df[3],
        'cCRE_type': filtered_ccre_df[9]
    })
    result_df.to_csv(output_file, sep='\t', index=False)
    return result_df

def calculate_bigwig_statistics(bw_file):
    """
    Calculate statistics for a single bigWig file.
    """
    prefix = os.path.basename(bw_file).split('.bw')[0]
    main_chromosomes = [f"chr{i}" for i in range(1, 23)]
    
    bw = pybigtools.open(bw_file)
    
    total_sum = 0
    total_sum_squares = 0
    total_genome_bases = 0
    
    for chrom in main_chromosomes:
        if chrom in bw.chroms():
            chrom_len = bw.chroms()[chrom]
            total_genome_bases += chrom_len
            
            values = bw.values(chrom, 0, chrom_len, missing=0.0)
            total_sum += np.sum(values)
            total_sum_squares += np.sum(values**2)
    
    bw.close()
    
    mean = total_sum / total_genome_bases
    mean_of_squares = total_sum_squares / total_genome_bases
    variance = mean_of_squares - (mean * mean)
    std = np.sqrt(max(0, variance))
        
    return (prefix, mean, std)

def get_bigwig_statistics(bigwig_files, stats_file="bigwig_statistics.pkl", force_regenerate=False):
    """
    Calculate or load statistics for bigWig files.
    """
    if not force_regenerate and os.path.exists(stats_file):
        try:
            with open(stats_file, 'rb') as f:
                statistics = pickle.load(f)
                print(f"Loaded statistics from {stats_file}")
                return statistics
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    print(f"Calculating statistics for {len(bigwig_files)} bigWig files...")
    results = multiprocessing_list(calculate_bigwig_statistics, bigwig_files, num_processes=int(os.cpu_count()))
    
    statistics = {prefix: (mean, std) for prefix, mean, std in results}
    
    with open(stats_file, 'wb') as f:
        pickle.dump(statistics, f)
    
    print(f"Saved statistics to {stats_file}")
    return statistics

def process_single_ccre(args):
    """Extract and standardize signal for a single cCRE."""
    ccre_row, bigwig_files, window_size, statistics_dict = args
    chrom = ccre_row['chrom']
    start = ccre_row['start']
    end = ccre_row['end']
    rDHS = ccre_row['rDHS']
    
    half_window = window_size // 2
    center = (start + end) // 2
    region_start = max(0, center - half_window)
    region_end = region_start + window_size
    
    standardized_signals = {}
    
    for bw_file in bigwig_files:
        prefix = os.path.basename(bw_file).split('.bw')[0]
        bw = pybigtools.open(bw_file)
        
        if chrom in bw.chroms() and region_end <= bw.chroms()[chrom]:
            signal = np.array(bw.values(chrom, region_start, region_end, missing=0.0), dtype=np.float32)
            mean, std = statistics_dict[prefix]
            standardized_signals[prefix] = np.array((signal - mean) / std, dtype=np.float32)
        
        bw.close()
    
    return (rDHS, standardized_signals)


def bin_signal_matrix(matrix_df, bin_size=10, output_pkl="binned_signal_matrix.pkl"):
    """Bin signal matrix and immediately save to disk, managing memory carefully."""
    print(f"Binning with size {bin_size} using swifter...")
    
    length = matrix_df.iloc[0, 0].shape[0]
    
    def bin_array(arr):
        return np.mean(arr[:length].reshape(-1, bin_size), axis=1, dtype=np.float32)
    
    binned_df = matrix_df.swifter.applymap(bin_array).copy(deep=True)

    print(f"Clearing original matrix ...")
    del matrix_df
    gc.collect()
    
    return binned_df

def avg_signal_matrix(matrix_df, output='avg_signal_matrix.pkl'):
    """Average signal matrix by rDHS."""
    avg_df = matrix_df.swifter.applymap(np.mean)

    with open(output, 'wb') as f:
        pickle.dump(avg_df, f)

    return avg_df


def run_pca_umap(matrix_df, ccre_df, pca_components=None, n_neighbors=15, min_dist=0.1, metric='euclidean', unique=False, filename='umap'):
    """Run PCA to reduce dimensions before applying UMAP."""

    features = []
    for _, row in matrix_df.iterrows():
        features.append(np.concatenate([np.array(arr).flatten() for arr in row]))
    features = np.array(features, dtype=np.float32)
    
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(features)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, unique=unique, low_memory=True, verbose=True)
    embedding = reducer.fit_transform(reduced_features)

    with open('embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)
    
    color_map = {
        'PLS': '#FF0000',
        'pELS': '#FFA700',
        'dELS': '#FFCD00',
        'CA-H3K4me3': '#FFAAAA',
        'CA-CTCF': '#00B0F0',
        'CA-only': '#06DA93',
        'CA-TF': '#BE28E5',
        'TF-only': '#D876EC'
    }

    colors = [color_map.get(ctype, '#CCCCCC') for ctype in ccre_df['cCRE_type']]
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.25, s=2)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=10, label='PLS'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCD00', markersize=10, label='dELS')

    ]
    plt.legend(handles=legend_elements)
    plt.title('UMAP of rDHS by cCRE Type')
    plt.savefig(f'{filename}.png', dpi=300)


def umap_color_by_feature(avg_df, umap_embedding, output_dir='umap_feature_plots'):
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(avg_df.shape[1]), desc="Processing features"):
        feature_name = avg_df.columns[i]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=avg_df.iloc[:, i],
            cmap='managua',
            s=1,
            alpha=0.75,
            edgecolors='none'
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'Value')
        
        ax.set_title(f'UMAP Colored by {feature_name}', fontsize=14)
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_feature_{i}.png", dpi=300)
        plt.close(fig)

    print(f"All {avg_df.shape[1]} UMAP visualizations saved to {output_dir}/")
    

def main():
    """
    Main function to process cCRE data, extract signals from bigWig files,
    and perform dimensionality reduction with PCA and UMAP.
    """
    # # Step 1: Process cCRE data
    # print("Step 1: Processing cCRE data...")
    # ccre_df = process_ccre_data()
    
    # del ccre_df
    # gc.collect()
    
    # # Load the processed cCRE data
    # ccre_df = pd.read_csv("ccre_df.tsv", sep="\t")
    # print(ccre_df.head())
    
    # # Step 2: Get bigWig file paths
    # print("Step 2: Finding bigWig files...")
    # bigwig_downloaded = "results/bigwig/downloaded"
    # bigwig_longread = "results/bigwig/longread"
    
    # bigwig_files = []
    # for file in os.listdir(bigwig_downloaded):
    #     if file.endswith('.bw'):
    #         bigwig_files.append(os.path.join(bigwig_downloaded, file))
    # for file in os.listdir(bigwig_longread):
    #     if file.endswith('.bw'):
    #         bigwig_files.append(os.path.join(bigwig_longread, file))
    
    # print(f"Found {len(bigwig_files)} bigWig files")
    
    # # Step 3: Calculate or load bigWig statistics
    # print("Step 3: Calculating bigWig statistics...")
    # statistics_dict = get_bigwig_statistics(bigwig_files)
    
    # # Step 4: Prepare arguments for processing cCREs
    # args_list = [(row, bigwig_files, 1000, statistics_dict) 
    #             for _, row in ccre_df.iterrows()]
    
    # # Step 5: Process all cCREs to extract signals
    # print("Step 5: Processing all cCREs...")
    # signal_matrix_dict = multiprocessing_dict(process_single_ccre, args_list, num_processes=int(os.cpu_count()))
    
    # # Step 6: Convert to DataFrame and save
    # print("Step 6: Converting to DataFrame...")
    # signal_matrix_df = pd.DataFrame.from_dict(signal_matrix_dict, orient='index')
    # print(f"Signal matrix shape: {signal_matrix_df.shape}")
    
    # # Step 7: Bin signal matrix
    # print("Step 7: Binning signal matrix...")
    # binned_signal_matrix = bin_signal_matrix(signal_matrix_df, bin_size=10)
    
    # with open('binned_signal_matrix.pkl', 'wb') as f:
    #     pickle.dump(binned_signal_matrix, f)
    
    # # Step 8: Load binned signal matrix and perform quality checks
    # with open("binned_signal_matrix.pkl", "rb") as f: 
    #     binned_signal_matrix = pickle.load(f)
    # print(f"Binned signal matrix shape: {binned_signal_matrix.shape}")
    # print(binned_signal_matrix.head())

    # # Step 8.5 Average signal matrix
    # print("Step 8.5: Averaging signal matrix...")
    # avg_signal_matrix(binned_signal_matrix, output='avg_signal_matrix.pkl')
    
    # # Step 9 (optional): Run UMAP
    # print("Step 9: Running UMAP...")
    # run_umap(binned_signal_matrix, ccre_df, n_neighbors=30, 
    #             unique=False, metric='euclidean', filename='umap_PLES_dELS_euclidean')

    # # Step 10: Run PCA followed by UMAP
    # print("Step 9: Running PCA and UMAP...")
    # run_pca_umap(binned_signal_matrix, ccre_df, pca_components=288, n_neighbors=30, 
    #             unique=True, metric='manhattan', filename='umap_PLES_dELS_manhattan')

    with open ('avg_signal_matrix.pkl', 'rb') as f:
        avg_df = pickle.load(f)
    print(f"Average signal matrix shape: {avg_df.shape}")
    print(f"\nAverage signal matrix head: {avg_df.head()}")
    with open ('embedding.pkl', 'rb') as f:
        umap_embedding = pickle.load(f)

    umap_color_by_feature(avg_df, umap_embedding, output_dir='umap_feature_plots')
    
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()