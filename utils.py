import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import os
import pandas as pd

def import_datasets(read_dir, expr_filename, metadata_filename):
    """
    Import expression matrix and metadata from specified files.
    
    Parameters:
    -----------
    read_dir : str
        Directory path containing the data files
    expr_filename : str
        Filename of the expression matrix
    metadata_filename : str
        Filename of the metadata
    
    Returns:
    --------
    tuple
        A tuple containing metadata and expression matrix DataFrames
    """
    # Constructing file paths
    expr_filepath = os.path.join(read_dir, expr_filename)
    metadata_filepath = os.path.join(read_dir, metadata_filename)

    # Read expression matrix and metadata into dfs
    try:
        expr_mat = pd.read_csv(expr_filepath, sep='\t', header=0, index_col=0)
        metadata = pd.read_csv(metadata_filepath, sep='\t', header=0, index_col=0)
    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: One of the files is empty.")
        return None, None

    # Rename the 'ct' column in metadata to 'CellType' if it exists
    if 'ct' in metadata.columns:
        metadata.rename(columns={'ct': 'CellType'}, inplace=True)

    # Filter the expression matrix to include only columns present in the metadata
    expr_mat = expr_mat.loc[:, metadata.index]

    # Basic data validation
    print("Expression Matrix Shape:", expr_mat.shape)
    print("Metadata Shape:", metadata.shape)
    
    # Optional: Check for any missing values
    print("\nMissing Values:")
    print("Expression Matrix Missing Values:")
    print(expr_mat.isnull().sum())
    print("\nMetadata Missing Values:")
    print(metadata.isnull().sum())

    return metadata, expr_mat

def plot_gene_distribution(metadata, title='Distribution of Genes per Cell'):
    """
    Create a visually appealing histogram of gene distribution.
    
    Parameters:
    -----------
    metadata : pandas.DataFrame
        DataFrame containing 'nGene' column
    title : str, optional
        Title of the plot (default: 'Distribution of Genes per Cell')
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    sns.histplot(
        metadata['nGene'], 
        kde=True,
        bins=50, 
        color='#3498db',
        edgecolor='white'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Genes', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    gene_stats = f'Mean: {metadata["nGene"].mean():.2f}\nMedian: {metadata["nGene"].median():.2f}'
    plt.text(0.95, 0.95, gene_stats, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_umi_distribution(metadata, title='Distribution of UMIs per Cell'):
    """
    Create a visually appealing histogram of UMI distribution.
    
    Parameters:
    -----------
    metadata : pandas.DataFrame
        DataFrame containing 'nUMI' column
    title : str, optional
        Title of the plot (default: 'Distribution of UMIs per Cell')
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    sns.histplot(
        metadata['nUMI'], 
        kde=True,
        bins=50, 
        color='#e74c3c',
        edgecolor='white'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of UMIs', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    umi_stats = f'Mean: {metadata["nUMI"].mean():.2f}\nMedian: {metadata["nUMI"].median():.2f}'
    plt.text(0.95, 0.95, umi_stats, 
             transform=plt.gca().transAxes, 
             verticalalignment='top', 
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_batch_proportion(metadata, title='Proportion of Cells across Batches'):
    """
    Create a visually appealing bar plot of cell proportions across batches.
    
    Parameters:
    -----------
    metadata : pandas.DataFrame
        DataFrame containing 'batch' column
    title : str, optional
        Title of the plot (default: 'Proportion of Cells across Batches')
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    ax = sns.countplot(x='batch', data=metadata, palette='viridis')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Number of Cells', fontsize=12)
    
    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_celltype_batch_proportion(metadata, title='Proportion of Cells per Cell Type across Batches'):
    """
    Create a visually appealing bar plot of cell types across batches.
    
    Parameters:
    -----------
    metadata : pandas.DataFrame
        DataFrame containing 'CellType' and 'batch' columns
    title : str, optional
        Title of the plot (default: 'Proportion of Cells per Cell Type across Batches')
    """
    plt.figure(figsize=(16, 8), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    ax = sns.countplot(x='CellType', hue='batch', data=metadata, palette='Set2')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cell Type', fontsize=12)
    plt.ylabel('Number of Cells', fontsize=12)
    plt.xticks(rotation=90)
    
    # Adjust layout to prevent cutting off x-axis labels
    plt.tight_layout()
    plt.show()

def run_dimensional_reduction_analysis(expr_mat, metadata):
    """
    Perform PCA and UMAP dimensional reduction and visualization.
    
    Parameters:
    -----------
    expr_mat : pandas.DataFrame
        Expression matrix
    metadata : pandas.DataFrame
        Metadata associated with the expression matrix
    """
    # Prepare data for scanpy AnnData
    adata = sc.AnnData(X=expr_mat.T, obs=metadata)
    
    # Normalize & log transform data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Run PCA
    sc.tl.pca(adata)
    
    # PCA plots
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sc.pl.pca(adata, color=['batch'], title='PCA - Batch', show=False)
    
    plt.subplot(1, 2, 2)
    sc.pl.pca(adata, color=['CellType'], title='PCA - Cell Type', show=False)
    plt.tight_layout()
    plt.show()
    
    # Run UMAP
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    
    # UMAP plots
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sc.pl.umap(adata, color=['batch'], title='UMAP - Batch', show=False)
    
    plt.subplot(1, 2, 2)
    sc.pl.umap(adata, color=['CellType'], title='UMAP - Cell Type', show=False)
    plt.tight_layout()
    plt.show()
    
    return adata