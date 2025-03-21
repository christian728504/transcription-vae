import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import umap
import pybigtools
from gtfparse import read_gtf
import pickle
from multiprocessing import Pool
from scipy.stats import pearsonr

print('Imported dependencies successfully!')

def process_ccre_data(cell_type_specific_ccres="/data/projects/encode/Registry/V4/GRCh38/Cell-Type-Specific/Individual-Files/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed",
                     pls_gene_list="/data/projects/encode/Registry/V4/GRCh38/PLS-Gene-List.txt",
                     gencode_gtf="/data/common/genome/gencode.v47.basic.annotation.gtf",
                     output_file="ccre_df.tsv"):
    """Process K562-specific cCREs, filtering for relevant types."""
    main_chromosomes = [f"chr{i}" for i in range(1, 23)]

    ccre_df = pd.read_csv(cell_type_specific_ccres, sep='\t', header=None)
    filtered_df = ccre_df[
        (ccre_df[9].isin(['PLS', 'dELS'])) & 
        (ccre_df[0].isin(main_chromosomes))]
    
    result_df = pd.DataFrame({
        'chrom': filtered_df[0],
        'start': filtered_df[1],
        'end': filtered_df[2],
        'rDHS': filtered_df[3],
        'cCRE_type': filtered_df[9]
    })
    pls_gene_list = pd.read_csv(pls_gene_list, sep='\t', header=None)
    pls_gene_list.columns = ['rDHS', 'gene_id']
    result_df = result_df.merge(pls_gene_list, how='left', on='rDHS')

    gtf_df = read_gtf(gencode_gtf, result_type='pandas')
    genes_df = gtf_df[gtf_df['feature'] == 'gene'][['gene_id', 'strand']]
    result_df = result_df.merge(genes_df, on='gene_id', how='left')

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

def run_umap_visualization(signal_matrix, ccre_df, assay_names, output_dir="umap_plots", metric=None, color_by='ccre', load_from_file=False):
    if load_from_file:
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
    else:
        print(f'Signal matrix shape: {signal_matrix.shape}')
        os.makedirs(output_dir, exist_ok=True)
        num_samples = signal_matrix.shape[0]
        signal_2d = signal_matrix.reshape(num_samples, -1)
        print(f'Flattened signal matrix shape: {signal_2d.shape}')
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, verbose=True, metric=metric)
        embedding = reducer.fit_transform(signal_2d)
        with open('embedding.pkl', 'wb') as f:
            pickle.dump(embedding, f)

    if color_by == 'strand':
        strands = ccre_df['strand'].astype(str)
        color_map = {
            '+': '#FF4B4B',
            '-': '#4B4BFF',
            'nan': '#CCCCCC'}
        colors = [color_map[strand] for strand in strands]
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.25, s=2)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['+'], 
                markersize=10, label='+ strand'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['-'], 
                markersize=10, label='- strand'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['nan'], 
                markersize=10, label='Unknown')]
        plt.legend(handles=legend_elements)
        plt.title('UMAP of Genomic Signals')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(f"{output_dir}/umap_strand_{metric}.png", dpi=300)

    elif color_by == 'ccre':
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
        plt.savefig(f"{output_dir}/umap_ccre_{metric}.png", dpi=300)

    elif color_by == 'similarity':
        ccre_types = ccre_df['cCRE_type'].values
        unique_types = np.unique(ccre_types)

        for i, assay_name in enumerate(assay_names):
            assay_signals = signal_matrix[:, i, :]

            type_avg_profiles = {}
            for ctype in unique_types:
                type_indices = np.where(ccre_types == ctype)[0]
                type_avg_profiles[ctype] = np.mean(assay_signals[type_indices], axis=0)
            
            similarity_scores = []
            for j, ctype in enumerate(ccre_types):
                region_signal = assay_signals[j]
                avg_profile = type_avg_profiles.get(ctype, np.zeros_like(region_signal))

                epsilon = 1e-10
                dot_product = np.dot(region_signal, avg_profile)
                norm_region = np.linalg.norm(region_signal) + epsilon
                norm_avg = np.linalg.norm(avg_profile) + epsilon
                
                similarity = dot_product / (norm_region * norm_avg)
                similarity_scores.append(similarity)

            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=similarity_scores, cmap='coolwarm', 
                                alpha=0.5, s=3, vmin=-1, vmax=1)

            plt.colorbar(scatter, label="Similarity to type-specific average profile")
            plt.title(f'UMAP of Genomic Signals - {assay_name}\n(Colored by Type-Specific Profile Similarity)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.savefig(f"{output_dir}/umap_similarity_{assay_name}_{metric}.png", dpi=300)
    
    plt.close()

def main():
    ccre_df = process_ccre_data()
    bigwig_files, assay_names = find_bigwig_files()
    signal_matrix = build_signal_matrix(ccre_df, bigwig_files)
    # plot_average_profiles(signal_matrix, ccre_df, assay_names)
    run_umap_visualization(signal_matrix, ccre_df, assay_names, color_by='similarity', metric='cosine', load_from_file=True)

if __name__ == "__main__":
    main()