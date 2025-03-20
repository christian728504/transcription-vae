import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import umap
import pybigtools
import pickle
from multiprocessing import Pool

def process_ccre_data(input_file="/data/projects/encode/Registry/V4/GRCh38/Cell-Type-Specific/Individual-Files/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed", 
                     output_file="ccre_df.tsv"):
    """Process K562-specific cCREs, filtering for relevant types."""
    main_chromosomes = [f"chr{i}" for i in range(1, 23)]

    ccre_df = pd.read_csv(input_file, sep='\t', header=None, engine='python')
    filtered_df = ccre_df[
        (ccre_df[9].isin(['PLS', 'dELS', 'pELS'])) & 
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

def find_bigwig_files(bigwig_dir="results/aggregated"):
    bigwig_files = [os.path.join(bigwig_dir, f) for f in os.listdir(bigwig_dir) 
                   if f.endswith('.bw')]
    assay_names = [os.path.basename(f).split('.bw')[0] for f in bigwig_files]
    return bigwig_files, assay_names

def process_single_assay(args):
    bigwig_file, i, ccre_df, window_size = args
    num_ccres = len(ccre_df)
    signals = np.zeros((num_ccres, window_size), dtype=np.float32)
    
    bw = pybigtools.open(bigwig_file)
    
    for j, (_, row) in enumerate(ccre_df.iterrows()):
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        
        center = (start + end) // 2
        region_start = max(0, center - window_size // 2)
        region_end = region_start + window_size
        
        if chrom in bw.chroms() and region_end <= bw.chroms()[chrom]:
            values = bw.values(chrom, region_start, region_end, missing=0.0)
            signals[j] = np.array(values, dtype=np.float32)
    
    bw.close()
    return (i, signals)

def build_signal_matrix(ccre_df, bigwig_files, window_size=2000):
    num_ccres = len(ccre_df)
    num_assays = len(bigwig_files)
    signal_matrix = np.zeros((num_ccres, num_assays, window_size), dtype=np.float32)
    
    args_list = [(bigwig_files[i], i, ccre_df, window_size) for i in range(num_assays)]
    
    with Pool(processes=os.cpu_count()) as pool:
        for i, signals in tqdm(pool.imap(process_single_assay, args_list), total=num_assays):
            signal_matrix[:, i, :] = signals
    
    return signal_matrix

def plot_average_profiles(signal_matrix, ccre_df, assay_names, output_dir="average_profiles"):
    os.makedirs(output_dir, exist_ok=True)
    
    ccre_types = ccre_df['cCRE_type'].values
    unique_types = np.unique(ccre_types)
    colors = {'PLS' : '#FF0000', 
               'pELS' : '#FFA700', 
               'dELS' : '#FFCD00'}
    
    window_size = signal_matrix.shape[2]
    x_coords = np.arange(-window_size//2, window_size//2)
    
    for i, assay_name in enumerate(assay_names):
        plt.figure(figsize=(10, 6))
        
        for ctype in unique_types:
            type_indices = np.where(ccre_types == ctype)[0]
            avg_profile = np.nanmean(signal_matrix[type_indices, i, :], axis=0)
            
            plt.plot(x_coords, avg_profile, label=f"{ctype} (n={len(type_indices)})", 
                     color=colors.get(ctype, 'gray'), linewidth=2)
        
        plt.title(f"{assay_name}")
        plt.xlabel("Distance from Center (bp)")
        plt.ylabel("Signal")
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{assay_name}_profile.png", dpi=300)
        plt.close()

def run_umap_visualization(signal_matrix, ccre_df, output_dir="umap_plots"):
    os.makedirs(output_dir, exist_ok=True)
    num_samples = signal_matrix.shape[0]
    signal_2d = signal_matrix.reshape(num_samples, -1)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, verbose=True, metric='cosine')
    embedding = reducer.fit_transform(signal_2d)
    with open('embedding.pkl', 'wb') as f:
        pickle.dump(embedding, f)

    ccre_types = ccre_df['cCRE_type'].values
    color_map = {'PLS' : '#FF0000', 
               'pELS' : '#FFA700', 
               'dELS' : '#FFCD00'}
    colors = [color_map.get(ctype, 'gray') for ctype in ccre_types]
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.25, s=2)
    unique_types = np.unique(ccre_types)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(ctype, 'gray'), 
                             markersize=10, label=ctype) for ctype in unique_types]
    plt.legend(handles=legend_elements)
    
    plt.title('UMAP of Genomic Signals')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(f"{output_dir}/umap_ccre_types.png", dpi=300)
    plt.close()

def main():
    ccre_df = process_ccre_data()
    bigwig_files, assay_names = find_bigwig_files()
    signal_matrix = build_signal_matrix(ccre_df, bigwig_files)
    # plot_average_profiles(signal_matrix, ccre_df, assay_names)
    run_umap_visualization(signal_matrix, ccre_df)

if __name__ == "__main__":
    main()