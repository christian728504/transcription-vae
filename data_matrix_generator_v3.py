import numpy as np
import pandas as pd
import os
import pickle
from multiprocessing import Pool
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import pybigtools
import gc

def run_parallel(func, items, num_processes=None):
    """Run a function in parallel and return results."""
    if num_processes is None:
        num_processes = os.cpu_count()
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(func, items), total=len(items)))
    
    return results

def process_ccre_data(input_file="/data/projects/encode/Registry/V4/GRCh38/Cell-Type-Specific/Individual-Files/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed", 
                     output_file="ccre_df.tsv"):
    """Process K562-specific cCREs, filtering for relevant types."""
    main_chromosomes = [f"chr{i}" for i in range(1, 23)]

    ccre_df = pd.read_csv(input_file, sep='\t', header=None, engine='python')
    filtered_df = ccre_df[
        (ccre_df[9].isin(['PLS', 'dELS'])) & 
        (ccre_df[0].isin(main_chromosomes))
    ]
    
    result_df = pd.DataFrame({
        'chrom': filtered_df[0],
        'start': filtered_df[1],
        'end': filtered_df[2],
        'rDHS': filtered_df[3],
        'cCRE_type': filtered_df[9]
    })
    
    result_df.to_csv(output_file, sep='\t', index=False)
    return result_df

def calculate_bigwig_statistics(bw_file):
    """Calculate mean and standard deviation for a bigWig file."""
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
    variance = (total_sum_squares / total_genome_bases) - (mean * mean)
    std = np.sqrt(max(0, variance))
        
    return (prefix, mean, std)

def get_bigwig_statistics(bigwig_files, stats_file="bigwig_statistics.pkl", force_regenerate=False):
    """Calculate or load statistics for bigWig files."""
    if not force_regenerate and os.path.exists(stats_file):
        try:
            with open(stats_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    print(f"Calculating statistics for {len(bigwig_files)} bigWig files...")
    results = run_parallel(calculate_bigwig_statistics, bigwig_files)
    statistics = {prefix: (mean, std) for prefix, mean, std in results}
    
    with open(stats_file, 'wb') as f:
        pickle.dump(statistics, f)
    
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
            standardized_signals[prefix] = (signal - mean) / std
        
        bw.close()
    
    return (rDHS, standardized_signals)

def bin_signal_matrix(matrix_df, bin_size=10):
    """Bin signal matrix to reduce dimension."""
    print(f"Binning with size {bin_size}...")
    
    def bin_array(arr):
        length = arr.shape[0]
        return np.mean(arr[:length].reshape(-1, bin_size), axis=1, dtype=np.float32)
    
    # Apply binning to each cell in the DataFrame
    binned_df = pd.DataFrame(index=matrix_df.index)
    for col in matrix_df.columns:
        binned_df[col] = matrix_df[col].apply(bin_array)
    
    return binned_df

def avg_signal_matrix(matrix_df, output='avg_signal_matrix.pkl'):
    """Average signal matrix by rDHS."""
    avg_df = pd.DataFrame(index=matrix_df.index)
    for col in matrix_df.columns:
        avg_df[col] = matrix_df[col].apply(np.mean)

    with open(output, 'wb') as f:
        pickle.dump(avg_df, f)

    return avg_df

def run_pca_umap(matrix_df, ccre_df, pca_components=50, n_neighbors=15, min_dist=0.1, metric='euclidean', filename='umap'):
    """Run PCA to reduce dimensions before applying UMAP."""

    # Flatten each row's arrays into a single feature vector
    features = []
    for _, row in tqdm(matrix_df.iterrows(), total=len(matrix_df), desc="Preparing features"):
        features.append(np.concatenate([np.array(arr).flatten() for arr in row]))
    features = np.array(features, dtype=np.float32)
    
    # Run PCA
    print(f"Running PCA with {pca_components} components...")
    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(features)
    
    # Run UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        metric=metric, 
        low_memory=True, 
        verbose=True
    )
    embedding = reducer.fit_transform(reduced_features)

    # Save embedding
    with open('embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)
    
    # Create visualization
    color_map = {
        'PLS': '#FF0000',
        'dELS': '#FFCD00'
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
    plt.close()
    
    return embedding

def main():
    """Main function to process and analyze genomic data."""
    # Step 1: Process cCRE data
    print("Processing cCRE data...")
    ccre_df = process_ccre_data()
    
    # Step 2: Get bigWig file paths
    print("Finding bigWig files...")
    bigwig_dirs = ["results/bigwig/downloaded", "results/bigwig/longread"]
    bigwig_files = []
    
    for directory in bigwig_dirs:
        for file in os.listdir(directory):
            if file.endswith('.bw'):
                bigwig_files.append(os.path.join(directory, file))
    
    print(f"Found {len(bigwig_files)} bigWig files")
    
    # Step 3: Calculate bigWig statistics
    print("Calculating bigWig statistics...")
    statistics_dict = get_bigwig_statistics(bigwig_files)
    
    # Step 4: Process all cCREs to extract signals
    print("Processing all cCREs...")
    args_list = [(row, bigwig_files, 1000, statistics_dict) 
                for _, row in ccre_df.iterrows()]
    
    results = run_parallel(process_single_ccre, args_list)
    signal_matrix_dict = {rDHS: signals for rDHS, signals in results}
    
    # Step 5: Convert to DataFrame and save
    print("Creating signal matrix...")
    signal_matrix_df = pd.DataFrame.from_dict(signal_matrix_dict, orient='index')
    print(f"Signal matrix shape: {signal_matrix_df.shape}")
    
    # Step 6: Bin signal matrix
    print("Binning signal matrix...")
    binned_signal_matrix = bin_signal_matrix(signal_matrix_df, bin_size=10)
    
    with open('binned_signal_matrix.pkl', 'wb') as f:
        pickle.dump(binned_signal_matrix, f)
    
    # Step 7: Calculate average signal
    print("Calculating average signal...")
    avg_df = avg_signal_matrix(binned_signal_matrix)
    
    # Step 8: Run PCA and UMAP
    print("Running PCA and UMAP...")
    embedding = run_pca_umap(
        binned_signal_matrix, 
        ccre_df, 
        pca_components=50,  # Reduced from original 288
        n_neighbors=30, 
        metric='manhattan', 
        filename='umap_PLES_dELS_manhattan'
    )
    
    print("Analysis completed successfully.")

if __name__ == "__main__":
    main()