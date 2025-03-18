import requests
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

class EncodeMetadataParser:
    """
    A library for parsing and retrieving metadata from the ENCODE API,
    specifically focused on extracting bigWig and BAM files based on
    experiment type and strand specificity.
    """
    
    def __init__(self, auth_id=None, auth_pw=None):
        """
        Initialize the ENCODE metadata parser with optional authentication.
        
        Parameters:
        -----------
        auth_id : str, optional
            ENCODE authentication ID
        auth_pw : str, optional
            ENCODE authentication password
        """
        self.auth_id = auth_id
        self.auth_pw = auth_pw
        
    def fetch(self, url, quiet=True):
        """
        Fetch data from the ENCODE API.
        
        Parameters:
        -----------
        url : str
            URL to fetch data from
        quiet : bool, default=True
            Whether to suppress output
            
        Returns:
        --------
        dict
            JSON response from the API
        """
        headers = {'accept': 'application/json'}
        
        if not quiet:
            print(f"Fetching URL: {url}")
            
        if self.auth_id and self.auth_pw:
            response = requests.get(url, auth=(self.auth_id, self.auth_pw), headers=headers)
        else:
            response = requests.get(url, headers=headers)
            
        response.raise_for_status()
        return response.json()
    
    def flatten(self, nested_list):
        """
        Flatten a nested list into a single list.
        
        Parameters:
        -----------
        nested_list : list of lists
            A list where each element is itself a list.
            
        Returns:
        --------
        list
            A flattened list with all elements from sublists combined.
        """
        return [item for sublist in nested_list for item in sublist]
    
    def run_imap_multiprocessing(self, func, argument_list, num_processes):
        """
        Run a function in parallel using multiprocessing with a progress bar.
        
        Parameters:
        -----------
        func : function
            The function to be applied in parallel to each element of `argument_list`.
        
        argument_list : list
            List of arguments to be passed to `func`, each processed independently.
        
        num_processes : int
            Number of worker processes to spawn.
            
        Returns:
        --------
        list
            A list of results collected from applying `func` to each element in `argument_list`.
        """
        with Pool(processes=num_processes) as pool:
            result_list_tqdm = []
            for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
                result_list_tqdm.append(result)
                
        return result_list_tqdm
    
    def get_files(self, exp):
        """
        Get information for individual experiment files based on specified criteria.
        
        Parameters:
        -----------
        exp : str
            Experiment accession
            
        Returns:
        --------
        list
            List of file data records
        """
        url = f"https://www.encodeproject.org/experiments/{exp}/?format=json"
        data = self.fetch(url)
        
        toReturn = []
        
        # Get experiment metadata
        assay = data.get("assay_term_name", "")
        biosample_summary = data.get('biosample_summary', "")
        
        # Get biosample metadata
        try:
            biosample_term_name = data["biosample_ontology"]["term_name"]
            biosample_term_id = data["biosample_ontology"]["term_id"]
            organ_slims = ",".join(data["biosample_ontology"].get("organ_slims", []))
            cell_slims = ",".join(data["biosample_ontology"].get("cell_slims", []))
            system_slims = ",".join(data["biosample_ontology"].get("system_slims", []))
            developmental_slims = ",".join(data["biosample_ontology"].get("developmental_slims", []))
            synonyms = ",".join(data["biosample_ontology"].get("synonyms", []))
        except:
            biosample_term_name = ""
            biosample_term_id = ""
            organ_slims = ""
            cell_slims = ""
            system_slims = ""
            developmental_slims = ""
            synonyms = ""
            
        # Get target metadata
        try:
            target = data["target"]["label"]
        except:
            target = None
            
        # Determine strand specificity from replicates
        strand_specificity = "unstranded"  # Default
        replicates = data.get("replicates", [])
        for rep in replicates:
            if "library" in rep and "strand_specificity" in rep["library"]:
                strand_specificity = rep["library"]["strand_specificity"]
                break
                
        # Check if this is a long read experiment
        is_long_read = assay.lower() in ["long read rna-seq", "long read"]
            
        # Process files
        for file_info in data.get("files", []):
            # Only use released files
            if file_info.get("status") != "released":
                continue
                
            # Check assembly
            assembly = file_info.get("assembly")
            if assembly != "GRCh38":
                continue
                
            # Get file accession
            try:
                acc = file_info["accession"]
            except:
                continue
                
            # Get file format
            file_format = file_info.get("file_format", "")
            output_type = file_info.get('output_type', '')
            
            # For long read datasets, get BAM files with 'alignments' output type
            if is_long_read and file_format == "bam" and output_type == "alignments":
                biorep = file_info.get('biological_replicates', "")[0]
                url = file_info.get('cloud_metadata', {}).get('url', '')
                md5sum = file_info.get('md5sum', '')
                preferred_default = file_info.get('preferred_default', False)
                
                toReturn.append([exp, acc, assay, target, biosample_summary, biosample_term_id, 
                                biosample_term_name, cell_slims, biorep, file_format, output_type, 
                                strand_specificity, preferred_default, url, md5sum])
            
            # For all other experiment types, get bigWig files based on strand specificity
            elif not is_long_read and file_format == "bigWig":
                is_valid_output_type = False
                
                if strand_specificity == "unstranded" and output_type == "signal of unique reads":
                    is_valid_output_type = True
                elif strand_specificity != "unstranded" and (
                    output_type == "plus strand signal of unique reads" or 
                    output_type == "minus strand signal of unique reads"
                ):
                    is_valid_output_type = True
                    
                if is_valid_output_type:
                    biorep = file_info.get('biological_replicates', "")[0]
                    url = file_info.get('cloud_metadata', {}).get('url', '')
                    md5sum = file_info.get('md5sum', '')
                    preferred_default = file_info.get('preferred_default', False)
                    
                    toReturn.append([exp, acc, assay, target, biosample_summary, biosample_term_id, 
                                    biosample_term_name, cell_slims, biorep, file_format, output_type, 
                                    strand_specificity, preferred_default, url, md5sum])
                    
        return toReturn
    
    def get_metadata(self, url, num_processes=12):
        """
        Get metadata for all experiments that match the search criteria.
        
        Parameters:
        -----------
        url : str
            Search URL to fetch experiment list
        num_processes : int, default=12
            Number of processes to use for parallel processing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing metadata for all matching files
        """
        # Fetch experiment list
        metadata = self.fetch(url)
        exp_list = [item["accession"] for item in metadata.get("@graph", [])]
        
        # Process experiments in parallel
        metadata_results = self.run_imap_multiprocessing(self.get_files, exp_list, num_processes)
        
        # Flatten results and create DataFrame
        metadata_df = pd.DataFrame(
            self.flatten(metadata_results),
            columns=["exp", "acc", "assay", "target", "biosample_summary", 
                    "biosample_term_id", "biosample_term_name", "cell_slims", "biorep", 
                    "file_format", "output_type", "strand_specificity", "preferred_default", "url", "md5sum"]
        )
        
        # Count unique experiments
        unique_experiments = metadata_df["exp"].nunique()
        print(f"Number of unique experiments: {unique_experiments}")
        
        return metadata_df