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

class GenomicSignalAnalyzer:
    def __init__(self, 
                 cell_type_specific_ccres="/data/projects/encode/Registry/V4/GRCh38/Cell-Type-Specific/Individual-Files/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed",
                 pls_gene_list="/data/projects/encode/Registry/V4/GRCh38/PLS-Gene-List.txt",
                 gencode_gtf="/data/common/genome/gencode.v40.basic.annotation.gtf",
                 ccre_output="ccre_df.tsv",
                 bigwig_dir="results/aggregated",
                 window_size=2000):
        self.cell_type_specific_ccres = cell_type_specific_ccres
        self.pls_gene_list = pls_gene_list
        self.gencode_gtf = gencode_gtf
        self.ccre_output = ccre_output
        self.bigwig_dir = bigwig_dir
        self.window_size = window_size
        self.ccre_df = None
        self.matrix = None
        self.bigwig_files = None
        self.assay_names = None
        
    def process_ccre_data(self):
        main_chromosomes = [f"chr{i}" for i in range(1, 23)]
        ccre_df = pd.read_csv(self.cell_type_specific_ccres, sep='\t', header=None)
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
        
        pls_gene_list = pd.read_csv(self.pls_gene_list, sep='\t', header=None)
        pls_gene_list.columns = ['rDHS', 'gene_id']
        
        rDHS_genes = pls_gene_list.groupby('rDHS')['gene_id'].apply(lambda x: ','.join(x)).reset_index()
        result_df = result_df.merge(rDHS_genes, how='left', on='rDHS')
        
        gtf_df = read_gtf(self.gencode_gtf, result_type='pandas')
        genes_df = gtf_df[gtf_df['feature'] == 'gene'][['gene_id', 'strand']]
        
        def determine_strand(row):
            if row['cCRE_type'] != 'PLS':
                return ""
            if ',' in str(row['gene_id']):
                return 'Bidirectional'
            else:
                gene_id = row['gene_id']
                strand_info = genes_df[genes_df['gene_id'] == gene_id]['strand'].values
                return strand_info[0]
        
        result_df['strand'] = result_df.apply(determine_strand, axis=1)
        result_df.to_csv(self.ccre_output, sep='\t', index=False)
        self.ccre_df = result_df
        return result_df
        
    def find_bigwig_files(self):
        bigwig_files = [os.path.join(self.bigwig_dir, f) for f in os.listdir(self.bigwig_dir) 
                       if f.endswith('.bw')]
        assay_names = [os.path.basename(f).split('.bw')[0] for f in bigwig_files]
        self.bigwig_files = bigwig_files
        self.assay_names = assay_names
        return bigwig_files, assay_names
    
    @staticmethod
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
    
    def build_matrix(self):
        if self.ccre_df is None:
            self.ccre_df = pd.read_csv(self.ccre_output, sep='\t')
        
        if self.bigwig_files is None:
            self.find_bigwig_files()
            
        num_ccres = len(self.ccre_df)
        num_assays = len(self.bigwig_files)
        matrix = np.zeros((num_ccres, num_assays, self.window_size), dtype=np.float32)
        
        args_list = [(self.bigwig_files[i], i, self.ccre_df, self.window_size) for i in range(num_assays)]
        
        with Pool(processes=os.cpu_count()) as pool:
            for i, signals in tqdm(pool.imap(self.process_single_assay, args_list), total=num_assays):
                matrix[:, i, :] = signals
        
        self.matrix = matrix.reshape(num_ccres, -1)

    
    def plot_average_profiles(self, output_dir="average_profiles"):
        if self.matrix is None:
            self.build_matrix()
            
        if self.ccre_df is None:
            self.ccre_df = pd.read_csv(self.ccre_output, sep='\t')
            
        if self.assay_names is None:
            self.find_bigwig_files()
        
        os.makedirs(output_dir, exist_ok=True)
        
        ccre_types = self.ccre_df['cCRE_type'].values
        unique_types = np.unique(ccre_types)
        colors = {'PLS': '#FF0000', 'pELS': '#FFA700', 'dELS': '#FFCD00'}
        
        x_coords = np.arange(-self.window_size//2, self.window_size//2)
        
        for i, assay_name in enumerate(self.assay_names):
            plt.figure(figsize=(10, 6))
            
            for ctype in unique_types:
                type_indices = np.where(ccre_types == ctype)[0]
                avg_profile = np.nanmean(self.matrix[type_indices, i, :], axis=0)
                
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
    
    def run_umap_visualization(self, output_dir="umap_plots", metric=None):
        if self.ccre_df is None:
            self.ccre_df = pd.read_csv(self.ccre_output, sep='\t')
            
        if self.matrix is None:
            self.build_matrix()
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"Shape of final matrix: {self.matrix.shape}")
            
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, verbose=True, metric=metric)
        embedding = reducer.fit_transform(self.matrix)
        with open('embedding.pkl', 'wb') as f:
            pickle.dump(embedding, f)

        colors = []
        strand_color_map = {'+': '#0066CC', '-': '#CC0000', 'Bidirectional': '#9933CC'}
        legend_labels = set()
        
        for i, row in self.ccre_df.iterrows():
            if row['cCRE_type'] == 'PLS':
                color = strand_color_map.get(row['strand'])
                legend_labels.add(f"PLS ({row['strand']})")
            else:
                color = 'gray'
                legend_labels.add(row['cCRE_type'])
            colors.append(color)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.25, s=2)
        
        legend_elements = []
        for strand, color in strand_color_map.items():
            if f"PLS ({strand})" in legend_labels:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f"PLS ({strand})"))
        
        for label in sorted(legend_labels):
            if not label.startswith("PLS"):
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label=label))
        
        plt.legend(handles=legend_elements)
        plt.title('UMAP of Genomic Signals by Strand (PLS only)')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(f"{output_dir}/umap_strand_{metric}.png", dpi=300)
        plt.close()

def main():
    analyzer = GenomicSignalAnalyzer()
    # analyzer.process_ccre_data()
    # analyzer.find_bigwig_files()
    # analyzer.build_matrix()
    # analyzer.plot_average_profiles()
    analyzer.run_umap_visualization(metric='cosine')

if __name__ == "__main__":
    print('Imported dependencies successfully!')
    main()