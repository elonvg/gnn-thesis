import pandas as pd


def load_data(path, selected_columns=None, cut=None):
    dataframe = pd.read_csv(path, low_memory=False)

    if selected_columns is None:
        selected_columns = [
            'SK_unique_id',
            'species_common_name',
            'species_latin_name',
            'CAS',
            'chemical_name',
            'conc_unit',
            'conc',
            'duration',
            'duration_unit',
            'effect',
            'endpoint',
            'SMILES',
            'organism_lifestage_categorized',
            'administration_route_categorized',
            'NCBI_sci_name',
            'NCBI_last_known_rank',
            'NCBI_rank_superkingdom',
            'NCBI_rank_kingdom',
            'NCBI_rank_phylum',
            'NCBI_rank_subphylum',
            'NCBI_rank_class',
            'NCBI_rank_order',
            'NCBI_rank_family',
            'NCBI_rank_genus',
            'NCBI_rank_species',
            'species_group_corrected'
        ]

    dataframe = dataframe[list(selected_columns)]

    if cut is not None:
        dataframe = dataframe.iloc[:cut]

    return dataframe


def load_base_dataframe(config):
    return load_data(config["path"], config["selected_columns"], config.get("cut"))
