"""
Snakemake workflow for processing ENCODE files:
1. For long-read experiments: Process BAM files to generate strand-specific cut site coverage
2. For other experiments: Use existing bigWig files with appropriate naming convention
"""

import os
import pandas as pd
import re

configfile: "config.yaml"

# Load metadata
METADATA = pd.read_csv(config.get('metadata_file', 'metadata.tsv'), sep='\t')

# Split metadata into long-read BAM files and other bigWig files
BAM_METADATA = METADATA[(METADATA['file_format'] == 'bam')]

BIGWIG_METADATA = METADATA[METADATA['file_format'] == 'bigWig']

# Add direction column to BIGWIG_METADATA
def get_direction(output_type):
    if output_type == 'signal of unique reads':
        return 'forward'  # Unstranded treated as forward
    elif output_type == 'plus strand signal of unique reads':
        return 'forward'
    elif output_type == 'minus strand signal of unique reads':
        return 'reverse'
    else:
        raise ValueError(f"Unknown output type: {output_type}")

BIGWIG_METADATA['direction'] = BIGWIG_METADATA['output_type'].apply(get_direction)

# Ensure output directories exist
os.makedirs(config.get('output_dir', 'results'), exist_ok=True)
os.makedirs(os.path.join(config.get('output_dir', 'results'), 'bam'), exist_ok=True)
os.makedirs(os.path.join(config.get('output_dir', 'results'), 'bedgraph'), exist_ok=True)
os.makedirs(os.path.join(config.get('output_dir', 'results'), 'bigwig', 'longread'), exist_ok=True)
os.makedirs(os.path.join(config.get('output_dir', 'results'), 'bigwig', 'downloaded'), exist_ok=True)

# Create tuples for BAM files and bigWig files
BAM_TRIPLES = [(row['exp'], row['acc'], direction) 
               for _, row in BAM_METADATA.iterrows() 
               for direction in ["forward", "reverse"]]

BIGWIG_TRIPLES = [(row['exp'], row['acc'], row['direction']) 
                  for _, row in BIGWIG_METADATA.iterrows()]

# Get all unique experiment accessions
ALL_EXPERIMENTS = list(set(METADATA['exp'].unique().tolist()))

# Main rule
rule all:
    input:
        # BAM-derived bigWigs for long-read experiments
        expand(
            os.path.join(config.get('output_dir', 'results'), 'bigwig', 'longread', '{exp}.{acc}.{direction}.bw'),
            zip,
            exp=[t[0] for t in BAM_TRIPLES],
            acc=[t[1] for t in BAM_TRIPLES],
            direction=[t[2] for t in BAM_TRIPLES]
        ),
        # Pre-generated bigWigs for other experiments
        expand(
            os.path.join(config.get('output_dir', 'results'), 'bigwig', 'downloaded', '{exp}.{acc}.{direction}.bw'),
            zip,
            exp=[t[0] for t in BIGWIG_TRIPLES],
            acc=[t[1] for t in BIGWIG_TRIPLES],
            direction=[t[2] for t in BIGWIG_TRIPLES]
        )

# Download BAM files from ENCODE (for long-read experiments)
rule download_bam:
    output:
        bam = temp(os.path.join(config.get('output_dir', 'results'), 'bam', '{acc}.unsorted.bam'))
    params:
        url = lambda wildcards: BAM_METADATA[BAM_METADATA['acc'] == wildcards.acc]['url'].iloc[0]
    shell:
        """
        wget -O {output.bam} "{params.url}"
        """

# Sort BAM files
rule sort_bam:
    input:
        bam = os.path.join(config.get('output_dir', 'results'), 'bam', '{acc}.unsorted.bam')
    output:
        bam = os.path.join(config.get('output_dir', 'results'), 'bam', '{acc}.sorted.bam')
    shell:
        """
        samtools sort {input.bam} -o {output.bam}
        samtools index {output.bam}
        """

# Generate cut sites (for long-read experiments)
rule cut_sites:
    input:
        bam = os.path.join(config.get('output_dir', 'results'), 'bam', '{acc}.sorted.bam'),
        genome_sizes = config['genome_sizes']
    params:
        strand = lambda wildcards: '+' if wildcards.direction == 'forward' else '-'
    output:
        unsorted_bg = temp(os.path.join(config.get('output_dir', 'results'), 'bedgraph', '{acc}.{direction}.unsorted.bg')),
        filtered_bg = temp(os.path.join(config.get('output_dir', 'results'), 'bedgraph', '{acc}.{direction}.filtered.bg')),
        bg = os.path.join(config.get('output_dir', 'results'), 'bedgraph', '{acc}.{direction}.bg')
    shell:
        """
        bedtools genomecov -ibam {input.bam} -5 -strand {params.strand} -bga > {output.unsorted_bg}
        awk 'NR==FNR {{allowed[$1]=1; next}} ($1 in allowed)' {input.genome_sizes} {output.unsorted_bg} > {output.filtered_bg}
        bedSort {output.filtered_bg} {output.bg}
        """

# Convert bedgraph to bigwig (for long-read experiments)
rule bedgraph_to_bigwig:
    input:
        bg = os.path.join(config.get('output_dir', 'results'), 'bedgraph', '{acc}.{direction}.bg'),
        genome_sizes = config['genome_sizes']
    output:
        bw = os.path.join(config.get('output_dir', 'results'), 'bigwig', 'longread', '{exp}.{acc}.{direction}.bw')
    params:
        # Empty params section to allow access to all wildcards
    wildcard_constraints:
        acc = "|".join([re.escape(row['acc']) for _, row in BAM_METADATA.iterrows()])
    shell:
        """
        bedGraphToBigWig {input.bg} {input.genome_sizes} {output.bw}
        """

# Download pre-generated bigWig files (for other experiments)
rule download_bigwig:
    output:
        bw = os.path.join(config.get('output_dir', 'results'), 'bigwig', 'downloaded', '{exp}.{acc}.{direction}.bw')
    params:
        url = lambda wildcards: BIGWIG_METADATA[
            (BIGWIG_METADATA['exp'] == wildcards.exp) & 
            (BIGWIG_METADATA['acc'] == wildcards.acc) & 
            (BIGWIG_METADATA['direction'] == wildcards.direction)
        ]['url'].iloc[0]
    wildcard_constraints:
        acc = "|".join([re.escape(row['acc']) for _, row in BIGWIG_METADATA.iterrows()])
    shell:
        """
        wget -O {output.bw} "{params.url}"
        """