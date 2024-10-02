import numpy as np 
import pandas as pd
import torch 
import scipy as sp 
import networkx as nx
import copy
from tqdm import tqdm 
import os 
from torch import nn
import collections
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import MessagePassing
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.utils import  degree
from functools import reduce


tpc_e_tables = { 'AccountPermission' : ['AP_IDENT_T', 'AP_ACL', 'AP_TAX_ID', 'AP_L_NAME', 'AP_F_NAME'],
             'Customer' : ['C_ID', 'C_TAX_ID', 'C_ST_ID', 'C_L_NAME', 'C_F_NAME', 'C_M_NAME', 'C_GNDR', 'C_TIER', 'C_DOB', 'C_AD_ID', 'C_CTRY_1', 'C_AREA_1', 'C_LOCAL_1', 'C_EXT_1', 'C_CTRY_2', 'C_AREA_2', 'C_LOCAL_2', 'C_EXT_2', 'C_CTRY_3', 'C_AREA_3', 'C_LOCAL_3', 'C_EXT_3', 'C_EMAIL_1', 'C_EMAIL_2'],
             'CustomerAccount' : ['CA_ID', 'CA_B_ID', 'CA_C_ID', 'CA_NAME', 'CA_TAX_ST', 'CA_BAL'],
             'CustomerTaxrate' : ['CX_TX_ID', 'CX_C_ID'],
             'Holding' : ['H_T_ID', 'H_CA_ID', 'H_S_SYMB', 'H_DTS', 'H_PRICE', 'H_QTY'],
             'HoldingHistory' : ['HH_H_T_ID', 'HH_T_ID', 'HH_BEFORE_QTY', 'HH_AFTER_QTY'],
             'HoldingSummary' : ['HS_CA_ID', 'HS_S_SYMB', 'HS_QTY '],
             'WatchItem' : ['WI_WL_ID', 'WI_S_SYMB'],
             'WatchList' : ['WL_ID', 'WL_C_ID'],
             'Broker' : ['B_ID', 'B_ST_ID', 'B_NAME', 'B_NUM_TRADES', 'B_COMM_TOTAL'],
             'CashTransaction' : ['CT_T_ID', 'CT_DTS', 'CT_AMT', 'CT_NAME'],
             'Charge' : ['CH_TT_ID', 'CH_C_TIER', 'CH_CHRG'],
             'CommissionRate' : ['CR_C_TIER', 'CR_TT_ID', 'CR_EX_ID', 'CR_FROM_QTY', 'CR_TO_QTY', 'CR_RATE'],
             'Settlement' : ['SE_T_ID', 'SE_CASH_TYPE', 'SE_CASH_DUE_DATE', 'SE_AMT'],
             'Trade' : ['T_ID', 'T_DTS', 'T_ST_ID', 'T_TT_ID', 'T_IS_CASH', 'T_S_SYMB', 'T_QTY', 'T_BID_PRICE', 'T_CA_ID', 'T_EXEC_NAME', 'T_TRADE_PRICE', 'T_CHRG', 'T_COMM', 'T_TAX', 'T_LIFO'],
             'TradeHistory' : ['TH_T_ID', 'TH_DTS', 'TH_ST_ID'],
             'TRADE_REQUEST' : ['TR_T_ID', 'TR_TT_ID', 'TR_S_SYMB', 'TR_QTY', 'TR_BID_PRICE', 'TR_B_ID'],
             'TradeType' : ['TT_ID', 'TT_NAME', 'TT_IS_SELL', 'TT_IS_MRKT'],
             'Company' : ['CO_ID', 'CO_ST_ID', 'CO_NAME', 'CO_IN_ID', 'CO_SP_RATE', 'CO_CEO', 'CO_AD_ID', 'CO_DESC', 'CO_OPEN_DATE'],
             'CompanyCompetitor' : ['CP_CO_ID', 'CP_COMP_CO_ID', 'CP_IN_ID'],
             'DailyMarket' : ['DM_DATE', 'DM_S_SYMB', 'DM_CLOSE', 'DM_HIGH', 'DM_LOW', 'DM_VOL'],
             'Exchange' : ['EX_ID', 'EX_NAME', 'EX_NUM_SYMB', 'EX_OPEN', 'EX_CLOSE', 'EX_DESC', 'EX_AD_ID'],
             'Financial' : ['FI_CO_ID', 'FI_YEAR', 'FI_QTR', 'FI_QTR_START_DATE', 'FI_REVENUE', 'FI_NET_EARN', 'FI_BASIC_EPS', 'FI_DILUT_EPS', 'FI_MARGIN', 'FI_INVENTORY', 'FI_ASSETS', 'FI_LIABILITY', 'FI_OUT_BASIC', 'FI_OUT_DILUT'],
             'Industry' : ['IN_ID', 'IN_NAME', 'IN_SC_ID'],
             'LastTrade' : ['LT_S_SYMB', 'LT_DTS', 'LT_PRICE', 'LT_OPEN_PRICE', 'LT_VOL'],
             'NewsItem' : ['NI_ID', 'NI_HEADLINE', 'NI_SUMMARY', 'NI_ITEM', 'NI_DTS', 'NI_SOURCE', 'NI_AUTHOR'],
             'NewsXRef' : ['NX_NI_ID', 'NX_CO_ID'],
             'Sector' : ['SC_ID', 'SC_NAME'],
             'Security' : ['S_SYMB', 'S_ISSUE', 'S_ST_ID', 'S_NAME', 'S_EX_ID', 'S_CO_ID', 'S_NUM_OUT', 'S_START_DATE', 'S_EXCH_DATE', 'S_PE', 'S_52WK_HIGH', 'S_52WK_HIGH_DATE', 'S_52WK_LOW', 'S_52WK_LOW_DATE', 'S_DIVIDEND', 'S_YIELD'],
             'Address' : ['AD_ID', 'AD_LINE1', 'AD_LINE2', 'AD_ZC_CODE', 'AD_CTRY'],
             'StatusType' : ['ST_ID', 'ST_NAME'],
             'TaxRate' : ['TX_ID', 'TX_NAME', 'TX_RATE'],
             'ZipCode' : ['ZC_CODE', 'ZC_TOWN', 'ZC_DIV']}

tpc_e_prefix = {'AccountPermission' : 'AP_',
             'Customer' : 'C_',
             'CustomerAccount' : 'CA_',
             'CustomerTaxrate' : 'CX_',
             'Holding' : 'H_',
             'HoldingHistory' : 'HH_',
             'HoldingSummary' : 'HS_',
             'WatchItem' : 'WI_',
             'WatchList' : 'WL_',
             'Broker' : 'B_',
             'CashTransaction' : 'CT_',
             'Charge' : 'CH_',
             'CommissionRate' : 'CR_',
             'Settlement' : 'SE_',
             'Trade' : 'T_',
             'TradeHistory' : 'TH_',
             'TRADE_REQUEST' : 'TR_',
             'TradeType' : 'TT_',
             'Company' : 'CO_',
             'CompanyCompetitor' : 'CP_',
             'DailyMarket' : 'DM_',
             'Exchange' : 'EX_',
             'Financial' : 'FI_',
             'Industry' : 'IN_',
             'LastTrade' : 'LT_',
             'NewsItem' : 'NI_',
             'NewsXRef' : 'NX_',
             'Sector' : 'SC_',
             'Security' : 'S_',
             'Address' : 'AD_',
             'StatusType' : 'ST_',
             'TaxRate' : 'TX_',
             'ZipCode' : 'ZC_'}


genes_tables = {'interactions': ['IT_GeneID1', 'IT_GeneID2', 'IT_Type', 'IT_Expression_Corr'],
 'genes': ['GE_GeneID',
  'GE_Essential',
  'GE_Class',
  'GE_Complex',
  'GE_Phenotype',
  'GE_Motif',
  'GE_Chromosome',
  'GE_Function',
  'GE_Localization'],
 'classification': ['CL_GeneID', 'CL_Localization']}

genes_prefix = {'interactions' : 'IT_',
         'genes' : 'GE_',
         'classification' : 'CL_'}


hepatitis_tables = {'inf': ['IF_dur', 'IF_a_id'],
 'bio': ['BI_fibros', 'BI_activity', 'BI_b_id'],
 'dispat': ['DI_m_id', 'DI_sex', 'DI_age', 'DI_Type'],
 'rel13': ['R3_a_id', 'R3_m_id'],
 'indis': ['ID_got',
  'ID_gpt',
  'ID_alb',
  'ID_tbil',
  'ID_dbil',
  'ID_che',
  'ID_ttt',
  'ID_ztt',
  'ID_tcho',
  'ID_tp',
  'ID_in_id'],
 'rel11': ['R1_b_id', 'R1_m_id'],
 'rel12': ['R2_in_id', 'R2_m_id']}

hepatitis_prefix = {'inf': 'IF_',
             'bio': 'BI_',
             'dispat' : 'DI_',
             'rel13' : 'R3_',
             'indis': 'ID_',
             'rel11' : 'R1_',
             'rel12' : 'R2_'}


mondial_tables = {'locatedon': ['LO_City', 'LO_Province', 'LO_Country', 'LO_Island'],
 'geo_mountain': ['GE_Mountain', 'GE_Country', 'GE_Province'],
 'islandin': ['IS_Island', 'IS_Sea', 'IS_Lake', 'IS_River'],
 'mergeswith': ['ME_Sea1', 'ME_Sea2'],
 'geo_river': ['GO_River', 'GO_Country', 'GO_Province'],
 'organization': ['OR_Abbreviation',
  'OR_Name',
  'OR_City',
  'OR_Country',
  'OR_Province',
  'OR_Established'],
 'population': ['PO_Country', 'PO_Population_Growth', 'PO_Infant_Mortality'],
 'country': ['CO_Name',
  'CO_Code',
  'CO_Capital',
  'CO_Province',
  'CO_Area',
  'CO_Population'],
 'ismember': ['IM_Country', 'IM_Organization', 'IM_Type'],
 'mountain': ['MO_Name',
  'MO_Mountains',
  'MO_Height',
  'MO_Type',
  'MO_Longitude',
  'MO_Latitude'],
 'island': ['IL_Name',
  'IL_Islands',
  'IL_Area',
  'IL_Height',
  'IL_Type',
  'IL_Longitude',
  'IL_Latitude'],
 'geo_estuary': ['GS_River', 'GS_Country', 'GS_Province'],
 'borders': ['BO_Country1', 'BO_Country2', 'BO_Length'],
 'lake': ['LA_Name',
  'LA_Area',
  'LA_Depth',
  'LA_Altitude',
  'LA_Type',
  'LA_River',
  'LA_Longitude',
  'LA_Latitude'],
 'geo_source': ['GU_River', 'GU_Country', 'GU_Province'],
 'located': ['LC_City',
  'LC_Province',
  'LC_Country',
  'LC_River',
  'LC_Lake',
  'LC_Sea'],
 'desert': ['DE_Name', 'DE_Area', 'DE_Longitude', 'DE_Latitude'],
 'language': ['LE_Country', 'LE_Name', 'LE_Percentage'],
 'geo_desert': ['GD_Desert', 'GD_Country', 'GD_Province'],
 'geo_sea': ['GA_Sea', 'GA_Country', 'GA_Province'],
 'mountainonisland': ['MD_Mountain', 'MD_Island'],
 'geo_island': ['GI_Island', 'GI_Country', 'GI_Province'],
 'target': ['TA_Country', 'TA_Target'],
 'geo_lake': ['GL_Lake', 'GL_Country', 'GL_Province'],
 'continent': ['CE_Name', 'CE_Area'],
 'ethnicgroup': ['ET_Country', 'ET_Name', 'ET_Percentage'],
 'economy': ['EC_Country',
  'EC_GDP',
  'EC_Agriculture',
  'EC_Service',
  'EC_Industry',
  'EC_Inflation'],
 'province': ['PR_Name',
  'PR_Country',
  'PR_Population',
  'PR_Area',
  'PR_Capital',
  'PR_CapProv'],
 'river': ['RI_Name',
  'RI_River',
  'RI_Lake',
  'RI_Sea',
  'RI_Length',
  'RI_SourceLongitude',
  'RI_SourceLatitude',
  'RI_Mountains',
  'RI_SourceAltitude',
  'RI_EstuaryLongitude',
  'RI_EstuaryLatitude'],
 'city': ['CY_Name',
  'CY_Country',
  'CY_Province',
  'CY_Population',
  'CY_Longitude',
  'CY_Latitude'],
 'sea': ['SE_Name', 'SE_Depth'],
 'encompasses': ['EN_Country', 'EN_Continent', 'EN_Percentage'],
 'politics': ['PS_Country',
  'PS_Independence',
  'PS_Dependent',
  'PS_Government'],
 'religion': ['RE_Country', 'RE_Name', 'RE_Percentage']}


mondial_prefix = {'locatedon': 'LO_',
            'geo_mountain': 'GE_',
            'islandin': 'IS_',
            'mergeswith': 'ME_',
            'geo_river': 'GO_',
            'organization': 'OR_',
            'population': 'PO_',
            'country': 'CO_',
            'ismember': 'IM_',
            'mountain': 'MO_',
            'island': 'IL_',
            'geo_estuary': 'GS_',
            'borders': 'BO_',
            'lake': 'LA_',
            'geo_source': 'GU_',
            'located': 'LC_',
            'desert': 'DE_',
            'language': 'LE_',
            'geo_desert': 'GD_',
            'geo_sea': 'GA_',
            'mountainonisland': 'MD_',
            'geo_island': 'GI_',
            'target': 'TA_',
            'geo_lake': 'GL_',
            'continent': 'CE_',
            'ethnicgroup': 'ET_',
            'economy': 'EC_',
            'province': 'PR_',
            'river': 'RI_',
            'city': 'CY_',
            'sea': 'SE_',
            'encompasses': 'EN_',
            'politics': 'PS_',
            'religion': 'RE_'}


mutagenesis_prefix = {'molecule': 'MO_',
                      'atom': 'AT_',
                      'bond': 'BO_'}

mutagenesis_tables = {'molecule': ['MO_molecule_id',
  'MO_ind1',
  'MO_inda',
  'MO_logp',
  'MO_lumo',
  'MO_mutagenic'],
 'atom': ['AT_atom_id',
  'AT_molecule_id',
  'AT_element',
  'AT_type',
  'AT_charge'],
 'bond': ['BO_atom1_id', 'BO_atom2_id', 'BO_type']}


world_tables = {'countrylanguage': ['CO_CountryCode',
  'CO_Language',
  'CO_IsOfficial',
  'CO_Percentage'],
 'country': ['CU_Code',
  'CU_Name',
  'CU_Continent',
  'CU_Region',
  'CU_SurfaceArea',
  'CU_IndepYear',
  'CU_Population',
  'CU_LifeExpectancy',
  'CU_GNP',
  'CU_GNPOld',
  'CU_LocalName',
  'CU_GovernmentForm',
  'CU_HeadOfState',
  'CU_Capital',
  'CU_Code2'],
 'city': ['CI_ID',
  'CI_Name',
  'CI_CountryCode',
  'CI_District',
  'CI_Population']}

world_prefix = {'countrylanguage': 'CO_',
                'country' : 'CU_',
                'city' : 'CI_'}


databases_dictionnary = {'tpce' : (tpc_e_tables, tpc_e_prefix),
                         'genes' : (genes_tables, genes_prefix),
                         'hepatitis' : (hepatitis_tables, hepatitis_prefix),
                         'mondial' : (mondial_tables, mondial_prefix),
                         'mutagenesis' : (mutagenesis_tables, mutagenesis_prefix),
                         'world' : (world_tables, world_prefix)}




class Order_one_GNN(MessagePassing):
    def __init__(self, in_dim, out_dim, nbr_filter_taps=2, *args, **kwargs) -> None:
        super().__init__()
        self.model_parameters = nn.ParameterList()
        for _ in range(nbr_filter_taps):
            H = nn.Parameter(torch.rand(in_dim, out_dim), requires_grad=True)
            self.model_parameters.append(H)

    def forward(self, X, edge_index):
        X1 = torch.matmul(X, self.model_parameters[0])
        X2 = torch.matmul(X, self.model_parameters[1])
        row, col = edge_index
        deg = degree(col, X.size(0), dtype=X.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        X2 = self.propagate(edge_index, x=X2, norm=norm)
        X_out = X1 + X2
        return X_out
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



class Node_GNN(nn.Module):
    def __init__(self, d_F, nbr_layers, *args, **kwargs) -> None: # , is_normalized
        super().__init__()
        assert(len(d_F)!=0 and len(d_F) +1 == nbr_layers)
        #self.is_normalized = is_normalized

        dim_parameters = [(1, d_F[0])]
        for i in range(1, len(d_F)):
            dim_parameters.append((d_F[i-1], d_F[i]))
        dim_parameters.append((d_F[-1], 1))

        self.convs = torch.nn.ModuleList()
        for dims in dim_parameters:
            in_dim, out_dim = dims
            self.convs.append(Order_one_GNN(in_dim, out_dim))

    def forward(self, X, edge_index):
        X_out = torch.zeros_like(X)  # Create a new tensor for X_out
        for i in range(X.shape[1]):
            X_in = X[:, i].unsqueeze(1)  # Add an extra dimension to match the expected input
            for conv in self.convs[:-1]:
                X_in = conv(X_in,  edge_index)
                X_in = F.elu(X_in)
            X_in = self.convs[-1](X_in, edge_index)
            X_out[:, i] = X_in.squeeze(1)  # Remove the extra dimension
        return X_out

def preprocessing(path, database_name, databases_dictionnary = databases_dictionnary):
    '''
    database_name = "tpce", "genes", "hepatitis", "modial", "mutagenesis", "world"
    '''
    table_columns, table_prefix = databases_dictionnary[database_name]
    os.chdir(path)
    files = os.listdir()
    databases_dictionnary[database_name]
    relational_structure = {}
    omega = []
    rel = []
    tuples = []

    if database_name == 'tpce':
        for relation in table_columns.keys():
            working_rel = copy.deepcopy(relation)
            working_rel += '.txt' 
            if working_rel in files:
                df = pd.read_csv(working_rel, delimiter = '|', index_col=False)
                nbr_tuple = df.shape[0]
                working_tuple = [table_prefix[relation] + 't_' + str(i) for i in range(nbr_tuple)]
                df.columns = table_columns[relation]
                df.index = working_tuple
                relational_structure[table_prefix[relation]] = df
                omega += table_columns[relation]
                tuples += working_tuple
    else: 
        for relation in table_columns.keys():
            working_rel = copy.deepcopy(relation)
            working_rel = working_rel + '.csv'
            if working_rel in files:
                df = pd.read_csv(working_rel, delimiter = ',', index_col=False)
                nbr_tuple = df.shape[0]
                working_tuple = [table_prefix[relation] + 't_' + str(i) for i in range(nbr_tuple)]
                df.columns = table_columns[relation]
                df.index = working_tuple
                relational_structure[table_prefix[relation]] = df
                omega += table_columns[relation]
                tuples += working_tuple
                
    rel = list(relational_structure.keys())

    at = []
    for el in omega: 
        idx = el.index('_')
        working_list = [el + ',' + sub_el for sub_el in tuples if el[:idx + 1] == sub_el[:idx + 1]]
        at += working_list
    return rel, at, tuples, omega, relational_structure

def make_tuple_click(relational_structure, tuples):

    def make_click(Graph, relational_structure):
        new_edges = []
        for relation in relational_structure:
            working_tuples = list(relational_structure[relation].index)
            for i in range(len(working_tuples)):
                new_edges += [(working_tuples[i], working_tuples[j]) for j in range(i+1, len(working_tuples))]
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    def make_edges_values_tuples(Graph, relational_structure):
        new_edges = []
        for relation in relational_structure:
            working_tuples = list(relational_structure[relation].index)
            for el in working_tuples:
                working_row = relational_structure[relation].loc[el]
                working_row = working_row[pd.notna(working_row)]
                index = list(working_row.index)
                working_row = list(working_row)
                edges_to_add = [[subel, el] for subel in working_row]
                for idx, subel in zip(index, working_row):
                    Graph.add_node(subel, structure = idx, embedding = [])
                Graph.add_edges_from(edges_to_add)
                new_edges += edges_to_add
        return Graph, new_edges

    Graph = nx.Graph()

    for el in tuples:
        Graph.add_node(el, structure = 'tuple', embedding = [])

    Graph, edges_values_tuples = make_edges_values_tuples(Graph, relational_structure)
    Graph, click = make_click(Graph, relational_structure)
    edge_list = edges_values_tuples + click

    return Graph, edge_list


def make_standard_with_Gaifman(rel, at, tuples, omega, relational_structure):

    def make_relation_tuples(Graph, rel, tuples):
        new_edges =[]
        for relation in rel:
            working_rel = relation
            working_rel_tuples = [tuples[i] for i in range(len(tuples)) if working_rel == tuples[i][:len(working_rel)]]
            edges_to_add = [[working_rel, el] for el in working_rel_tuples]
            new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    def make_tuple_at(Graph, rel, relational_structure):
        new_edges =[]
        for relation in rel:
            working_tuples = list(relational_structure[relation].index)
            working_attribute = list(relational_structure[relation].columns)
            for attr in working_attribute: 
                edges_to_add = [[attr + ',' + sub_el, sub_el] for sub_el in  working_tuples]
                new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    


    def make_at_valeur(Graph, rel, at):
        new_edges =[]
        for relation in rel: 
            working_relation = relation 
            working_at = [at[i] for i in range(len(at)) if working_relation == at[i][:len(working_relation)]]
            dataset = copy.deepcopy(relational_structure[relation])
            edges_to_add = []
            for el in working_at:
                working_attribute, working_tuple = el.split(',')
                working_el = dataset[working_attribute][working_tuple]
                
                if  str(working_el) != 'nan': # C EST ICI QUE JAI CHANGÉ
                    Graph.add_node(working_el, structure = working_attribute, embedding = [])
                    edges_to_add.append([el, working_el])
                new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    def make_gaifman(Graph, rel, relational_structure):
        new_edges = []
        
        for relation in rel: 
            for el in relational_structure[relation].index:
                working_row = relational_structure[relation].loc[el]
                
                working_row = working_row[pd.notna(working_row)]
                working_row = list(working_row)
                edges_to_add = [
                    (working_row[i], working_row[j])
                    for i in range(len(working_row))
                    for j in range(i+1, len(working_row))
                    if pd.notna(working_row[j]) and pd.notna(working_row[i])
                ]
                
                new_edges += edges_to_add
        # Add edges to the graph
        Graph.add_edges_from(new_edges)
        
        return Graph, new_edges
    

    edge_list = []
    Graph = nx.Graph()

    for el in rel:
        Graph.add_node(el, structure = 'rel', embedding = [])
    for el in at:
        Graph.add_node(el, structure = 'at', embedding = [])
    for el in tuples:
        Graph.add_node(el, structure = 'tuple', embedding = [])
    for el in omega:
        Graph.add_node(el, structure = 'attribut', embedding = [])

    Graph, edges_relation_tuples = make_relation_tuples(Graph, rel, tuples)
    Graph, edges_tuple_at = make_tuple_at(Graph, rel, relational_structure)
    Graph, edges_at_valeur = make_at_valeur(Graph, rel, at)
    Graph, edges_gaifman = make_gaifman(Graph, rel, relational_structure)

    edge_list = edges_relation_tuples + edges_tuple_at + edges_at_valeur + edges_gaifman

    return Graph, edge_list


def make_standard(rel, at, tuples, omega, relational_structure):
    
    def make_relation_tuples(Graph, rel, tuples):
        new_edges =[]
        for relation in rel:
            working_rel = relation
            working_rel_tuples = [tuples[i] for i in range(len(tuples)) if working_rel == tuples[i][:len(working_rel)]]
            edges_to_add = [[working_rel, el] for el in working_rel_tuples]
            new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    def make_tuple_at(Graph, rel, relational_structure):
        new_edges =[]
        for relation in rel:
            working_tuples = list(relational_structure[relation].index)
            working_attribute = list(relational_structure[relation].columns)
            for attr in working_attribute: 
                edges_to_add = [[attr + ',' + sub_el, sub_el] for sub_el in  working_tuples]
                new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    def make_at_attribute(Graph, at, omega):
        new_edges = []
        for att in omega:
            working_attribute = att
            working_at = [at[i] for i in range(len(at)) if working_attribute == at[i][:len(working_attribute)]]
            edges_to_add = [[working_attribute, el] for el in working_at]
            new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges

    def make_at_valeur(Graph, rel, at):
        new_edges =[]
        for relation in rel: 
            working_relation = relation 
            working_at = [at[i] for i in range(len(at)) if working_relation == at[i][:len(working_relation)]]
            dataset = copy.deepcopy(relational_structure[relation])
            edges_to_add = []
            for el in working_at:
                working_attribute, working_tuple = el.split(',')
                working_el = dataset[working_attribute][working_tuple]
                
                if  str(working_el) != 'nan': # C EST ICI QUE JAI CHANGÉ
                    Graph.add_node(working_el, structure = working_attribute, embedding = [])
                    edges_to_add.append([el, working_el])
                new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    edge_list = []

    Graph = nx.Graph()
    for el in rel:
        Graph.add_node(el, structure = 'rel', embedding = [])
    for el in at:
        Graph.add_node(el, structure = 'at', embedding = [])
    for el in tuples:
        Graph.add_node(el, structure = 'tuple', embedding = [])
    for el in omega:
        Graph.add_node(el, structure = 'attribut', embedding = [])

    Graph, edges_relation_tuples = make_relation_tuples(Graph, rel, tuples)
    Graph, edges_tuple_at = make_tuple_at(Graph, rel, relational_structure)
    Graph, edges_at_valeur = make_at_valeur(Graph, rel, at)
    Graph, edges_at_attributes = make_at_attribute(Graph, at, omega)

    edge_list = edges_relation_tuples + edges_tuple_at + edges_at_valeur + edges_at_attributes

    return Graph, edge_list


def gaifman(relational_structure):

    def make_gaifman(relational_structure):
        edge_list = []
        for relation in relational_structure:
            for el in relational_structure[relation].index:
                working_row = relational_structure[relation].loc[el]
                working_row = working_row.dropna(inplace=False)
                index = list(working_row.index)
                working_row = list(working_row)
                new_edges =[(working_row[i], working_row[j]) for i in range(len(working_row)) for j in range(i+1, len(working_row))]
                for idx, subel in zip(index, working_row):
                    Graph.add_node(subel, structure = idx, embedding = [])
                Graph.add_edges_from(new_edges)
        return Graph, edge_list
   
    Graph = nx.Graph()
    Graph, edge_list = make_gaifman(relational_structure)
    return Graph, edge_list




def make_standard_without_A(rel, at, tuples, relational_structure):
    
    def make_relation_tuples(Graph, rel, tuples):
        new_edges =[]
        for relation in rel:
            working_rel = relation
            working_rel_tuples = [tuples[i] for i in range(len(tuples)) if working_rel == tuples[i][:len(working_rel)]]
            edges_to_add = [[working_rel, el] for el in working_rel_tuples]
            new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    def make_tuple_at(Graph, rel, relational_structure):
        new_edges =[]
        for relation in rel:
            working_tuples = list(relational_structure[relation].index)
            working_attribute = list(relational_structure[relation].columns)
            for attr in working_attribute: 
                edges_to_add = [[attr + ',' + sub_el, sub_el] for sub_el in  working_tuples]
                new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    

    def make_at_valeur(Graph, rel, at):
        new_edges =[]
        for relation in rel: 
            working_relation = relation 
            working_at = [at[i] for i in range(len(at)) if working_relation == at[i][:len(working_relation)]]
            dataset = copy.deepcopy(relational_structure[relation])
            edges_to_add = []
            for el in working_at:
                working_attribute, working_tuple = el.split(',')
                working_el = dataset[working_attribute][working_tuple]
                
                if  str(working_el) != 'nan': # C EST ICI QUE JAI CHANGÉ
                    Graph.add_node(working_el, structure = working_attribute, embedding = [])
                    edges_to_add.append([el, working_el])
                new_edges += edges_to_add
        Graph.add_edges_from(new_edges)
        return Graph, new_edges
    
    edge_list = []

    Graph = nx.Graph()
    for el in rel:
        Graph.add_node(el, structure = 'rel', embedding = [])
    for el in at:
        Graph.add_node(el, structure = 'at', embedding = [])
    for el in tuples:
        Graph.add_node(el, structure = 'tuple', embedding = [])

    Graph, edges_relation_tuples = make_relation_tuples(Graph, rel, tuples)
    Graph, edges_tuple_at = make_tuple_at(Graph, rel, relational_structure)
    Graph, edges_at_valeur = make_at_valeur(Graph, rel, at)

    edge_list = edges_relation_tuples + edges_tuple_at + edges_at_valeur

    return Graph, edge_list


def masked_databases(df, miss_rate = 0.1):
    num_nan = int(miss_rate * df.size)
    nan_indices = np.random.choice(df.size, num_nan, replace=False)
    a = df.to_numpy()
    a = np.reshape(a, a.size)
    if len(nan_indices)>0:
        a[nan_indices] = np.nan
    a = np.reshape(a, df.shape)
    new_df = pd.DataFrame(a, columns = df.columns, index = df.index)
    return new_df



def initialization(Graph, dim):
    shape = (len(Graph), dim)
    X = nn.init.xavier_uniform_(torch.empty(shape))
    return X

def initialization_for_perturbed_graph(X_init, G1, G2):
    node_g2 = G2.nodes()
    node_to_index1 = dict(zip(G1.nodes(), list(range(len(G1)))))
    slicing = []
    for node in node_g2:
            slicing.append(node_to_index1[node])
    return X_init[slicing,:]



def  new_shape(S1, S2):
    n1 = S1.shape[0]
    n2 = S2.shape[0]

    if n1 > n2:
        new_array = np.zeros((n1,n1))
        for i in range(n2):
            new_array[i,:n2] += S2[i,:]
        S2 = new_array
        
    elif n1 < n2:
        new_array = np.zeros((n2,n2))
        for i in range(n1):
            new_array[i,:n1] += S1[i,:]
        S1 = new_array
    return S1, S2
  
def permutation_matrix(G1, G2):
    nodes_small_graph = G2.nodes()
    nodes_big_graph = G1.nodes()

    n = len(nodes_small_graph)
    N = len(nodes_big_graph)

    assert(N>= n)

    pseudo_map = {node : {'big_graph': i, 'small_graph': None} for i,node in enumerate(nodes_big_graph)}
    for i, node in enumerate(nodes_small_graph):
        pseudo_map[node]['small_graph'] = i

    to_permutation = list(range(N))
    for i, node in enumerate(pseudo_map):
        index_big = pseudo_map[node]['big_graph']
        index_small = pseudo_map[node]['small_graph']

        if index_small is None: 
            to_permutation[i] =n
            n+=1 

        elif index_big != index_small:
            to_permutation[i] = index_small

    P = np.zeros((N, N))
    for i in range(N):
        P[i,to_permutation[i]] = 1
    return P


def special_masked_dataset(df, elements, miss_rate):
    working_array = df.to_numpy()
    working_array = np.reshape(working_array, working_array.size).astype(object)
    index = []
    index2 = []
    for i in range(working_array.size):
        if working_array[i] in elements:
            index.append(i)
    num_nan = int(miss_rate * len(index))
    nan_indices = np.random.choice(len(index), num_nan, replace=False).astype(int)

    for  i in range(len(index)):
        if i in nan_indices:
            index2.append(index[i])  
    if len(index2)>0:
        working_array[index2] = np.nan
    working_array = np.reshape(working_array, df.shape)
    out =  pd.DataFrame(working_array, columns = df.columns, index = df.index)
    return out

def perturbation_missing_value( relational_structure,elements, rate):

    relational_structure_perturbed = {}
    rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed = [], [], [], []

    for relation in relational_structure:
        dataset = copy.deepcopy(relational_structure[relation])
        new_dataset = special_masked_dataset(dataset, elements, rate)
        rel_perturbed.append(relation)
        tuples_perturbed += list(new_dataset.index)
        omega_perturbed += list(new_dataset.columns)
        inter_at = []
        for col in list(new_dataset.columns):
            inter_at += [col + ',' + el for el in list(new_dataset.index)]
        at_perturbed += inter_at
        relational_structure_perturbed[relation] = new_dataset
    return rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed


def upper_bound(GNN, G1, G2, x, Fl, L, shift_operator = 'normalized_adj'):

    def compute_shift_operators(G1,G2):
        S1 = np.array(nx.adjacency_matrix(G1).todense())
        S2 = np.array(nx.adjacency_matrix(G2).todense())
        if shift_operator == 'normalized_adj':
            degree_S1 = np.array(list(dict(G1.degree()).values()))
            degree_S1[np.where(degree_S1 == 0)] = 1
            degree_S2 = np.array(list(dict(G2.degree()).values()))
            degree_S2[np.where(degree_S2 == 0)] = 1

            S1 = np.divide(S1.T, degree_S1).T
            S2 = np.divide(S2.T, degree_S2).T
        return S1, S2

    def compute_opt_norme(S1,S2):
        S1, S2 = new_shape(S1, S2)
        P = permutation_matrix(G1, G2)

        E = S1 - np.dot(P, np.dot(S2, P.T))
        opNorm = np.linalg.norm(E, ord = 2)
        return opNorm
    
    def compute_H_inf(GNN, S1):
        eigenvalues, eigenvectors = np.linalg.eigh(S1)
        eigenvalues = eigenvalues.real

        eigenvalues = torch.tensor(eigenvalues).unsqueeze(1).to(torch.float)
        eig1, eig2 = eigenvalues[0], eigenvalues[-1]
        parameters = list(GNN.parameters())

        # first a (of aX +b)
        all_a = parameters[0].detach()
        all_a = all_a.reshape(1, all_a.numel())

        # first b (of aX+b)
        all_b = parameters[1].detach()
        all_b = all_b.reshape(1, all_b.numel())

        # all_a is a tensor of all the a in the nn same for all_b
        for i, param in enumerate(parameters[2:]):
            if i%2 == 0:
                working_param = param
                working_param= working_param.reshape(1, working_param.numel()).detach()
                all_a = torch.cat((all_a,working_param),dim =1)
            else:
                working_param = param
                working_param= working_param.reshape(1, working_param.numel()).detach()
                all_b = torch.cat((all_b,working_param),dim =1)

        # tensor of lenght 2*number of parameters, 
        # the first half of the tensor is the lowest eig value of S
        # and the second half the highest 
        working_eig = torch.tensor([eig1]* all_a.numel() + [eig2]* all_a.numel())

        all_a = torch.cat((all_a,all_a),dim =1)[0]
        all_b = torch.cat((all_b,all_b),dim =1)[0]
        H_inf = np.max(torch.abs(all_a*working_eig + all_b).numpy())
        return H_inf
    
    def compute_max_a(GNN):
        max_a = 0
        for i,param in enumerate(GNN.parameters()):
            if i%2 == 0:
                working_param = param.detach().numpy()
                max_a = np.max(np.abs(working_param))  if max_a < np.max(np.abs(working_param)) else max_a
        return max_a

    #1) Definition shift operators
    S1, S2 = compute_shift_operators(G1,G2)

    # 2) Compute ||S - \hat{S}||
    opNorm = compute_opt_norme(S1,S2)

    # 3) Compute H_inf
    H_inf = compute_H_inf(GNN, S1)

    # 4) Compute max_a 
    max_a = compute_max_a(GNN)
    return L* max_a*np.power(((Fl* H_inf)), L-1)*opNorm*np.linalg.norm(x)



def refined_upper_bound(GNN, G1, G2, x, d_F, L, shift_operator = 'normalized_adj'):
    def compute_shift_operators(G1,G2):
        S1 = np.array(nx.adjacency_matrix(G1).todense())
        S2 = np.array(nx.adjacency_matrix(G2).todense())
        if shift_operator == 'normalized_adj':
            degree_S1 = np.array(list(dict(G1.degree()).values()))
            degree_S1[np.where(degree_S1 == 0)] = 1
            degree_S2 = np.array(list(dict(G2.degree()).values()))
            degree_S2[np.where(degree_S2 == 0)] = 1

            S1 = np.divide(S1.T, degree_S1).T
            S2 = np.divide(S2.T, degree_S2).T
        return S1, S2

    def compute_opt_norme(S1,S2):
        S1, S2 = new_shape(S1, S2)
        P = permutation_matrix(G1, G2)

        E = S1 - np.dot(P, np.dot(S2, P.T))
        opNorm = np.linalg.norm(E, ord = 2)
        return opNorm

    def compute_all_H_inf_i_and_max_a(GNN, S1):

        parameters = list(GNN.parameters())
        eigenvalues, eigenvectors = np.linalg.eigh(S1)
        eigenvalues = eigenvalues.real
        eigenvalues = torch.tensor(eigenvalues).unsqueeze(1).to(torch.float)
        eig1, eig2 = eigenvalues[0], eigenvalues[-1]
        all_H_inf_i = []
        all_A_i = []
        for i in range(L):
            a,b = parameters[i*2],  parameters[i*2+1]

            a = a.detach()
            a = a.reshape(1, b.numel())[0]
            all_a = torch.cat((a,a))
            b =b.detach()
            b = b.reshape(1, b.numel())[0]
            all_b = torch.cat((b,b))

            working_eig = torch.tensor([eig1]* len(a) + [eig2]* len(a))
            H = torch.abs(all_a*working_eig + all_b).numpy()
            
            H_inf = np.max(H)
            max_a = torch.max(all_a)
            all_H_inf_i.append(H_inf)
            all_A_i.append(max_a)
        return all_H_inf_i, all_A_i
    
    def compute_Borne(all_H_inf_i, all_A_i, pi_Fi):
        B = 0
        for i in range(1,L):
            if  len(all_H_inf_i[:~i]) ==0:
                H_j_m = 1
            else : 
                H_j_m = reduce(lambda x, y: x*y, all_H_inf_i[:~i])

            if  len( all_H_inf_i[:i]) ==0:
                H_j_M = 1
            else : 
                H_j_M = reduce(lambda x, y: x*y,  all_H_inf_i[L-i:])

            B += H_j_m *   all_A_i[~i] * H_j_M
        
        B = B*pi_Fi
        return B

    #1) shift operators
    S1, S2 = compute_shift_operators(G1,G2)

    # 2) compute ||S - \hat{S}||
    opNorm = compute_opt_norme(S1,S2)

    # 3) Calcul de \Pi Fi
    pi_Fi = reduce(lambda x, y: x*y, d_F)

    # 4) compute  A_i and  H_inf_i
    all_H_inf_i, all_A_i = compute_all_H_inf_i_and_max_a(GNN, S1)

    # compute B
    B = compute_Borne(all_H_inf_i, all_A_i, pi_Fi)
    return opNorm* np.linalg.norm(x)*B



def remove_tuples(relational_structure, removing_rate):
   
    new_rs = {}
    new_tuples = []
    new_omega = []
    new_at= []
    new_rel= []
    for key in relational_structure.keys():
        df_relation = copy.deepcopy(relational_structure[key])
        boolean_list = np.random.choice([True, False], size=df_relation.shape[0], p=[1-removing_rate,removing_rate])
        boolean_list = boolean_list.tolist()
        df_relation = df_relation[boolean_list]
        new_rs[key] = df_relation
        new_tuples += list(new_rs[key].index)
        new_omega += list(new_rs[key].columns)
        at_relation = []
        for attribut in list(new_rs[key].columns):
            at_relation += [attribut + ',' + el for el in list(new_rs[key].index)]
        new_at += at_relation
        new_rel.append(key)
    return new_rel, new_at, new_tuples, new_omega, new_rs




def perturbed_on_tuples(rel, at, tuples, omega, relational_structure,perturbation, removing_rate=0.2):
    ''' 
    relational_structure = dict of pandas array of each relation of the relationnal strucutre, t
    perturbation = "remove_tuple", "add_tuples"
    '''
    if perturbation == "remove_tuple":
        rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed = remove_tuples(relational_structure, removing_rate)
        return (rel, at, tuples, omega, relational_structure),(rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed)
    else: 
        rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed = rel, at, tuples, omega, relational_structure
        rel, at, tuples, omega, relational_structure = remove_tuples(relational_structure, removing_rate)
        return (rel, at, tuples, omega, relational_structure),(rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed)
    

def incorporate_embeddings(embeddings, Graph):
    for i, data in enumerate(Graph.nodes(data=True)):
        node, meta = data
        struct, embed = meta
        Graph.add_node(node, structure = meta[struct], embedding = embeddings[i,:])
    return Graph




def equidepth_without_redundancy(relational_structure, nbr_bin):
    values_occurences = occurence(relational_structure)
    data = list(values_occurences.values())
    q = np.linspace(0,100, nbr_bin).astype(int)
    persentil = np.percentile(data, q)
    i = 0

    while sum(collections.Counter(persentil).values()) != len(collections.Counter(persentil))  and i  < 100000:
        for key in collections.Counter(persentil):
            if collections.Counter(persentil)[key] > 1:
                data.remove(key)
                persentil = np.percentile(data, q).astype(int)
        i +=1
    low_high_bound = [(int(np.ceil(persentil[i])), int(np.floor(persentil[i+1]))-1) for i in range(len(persentil) -1)] #ici je vais rajouter un -1 
    return low_high_bound


def occurence(relational_structure):
    out_dict = {}
    for relation in relational_structure:
        working_rel = copy.deepcopy(relational_structure[relation])
        working_rel = working_rel.dropna()
        working_rel = working_rel.to_numpy()
        working_rel = np.reshape(working_rel, working_rel.size)
        working_rel = list(np.reshape(working_rel, working_rel.size))
        #
        working_dict = collections.Counter(working_rel)
        for key in list(working_dict.keys()):
            if key in list(out_dict.keys()):
                out_dict[key] += working_dict[key]
            else : 
                out_dict[key] = working_dict[key]
    return out_dict 



def chosen_elements(values_occurences, low_bound, high_bound):
    candidates = []
    for key in values_occurences.keys():
        occ = values_occurences[key]
        if occ >= low_bound and occ <= high_bound:
            candidates.append(key)
    return candidates 

def expe_tuple_removal(working_path, database_name):
    rel, at, tuples, omega, relational_structure = preprocessing(working_path, database_name)

    dim, d_F =  30, [3,4,6,3]
    features, nbr_layers = max(d_F), len(d_F) +1
    in_dim, hidden_context, out_dim, num_layers_context=dim,dim,dim, 5
    removing_rate = np.linspace(0.1,0.50,10)  

    type_of_graph = ['Standard','Standard_with_gaifman', 'Standard_without_A','Bipartite','Tripartite','Tuple_click']
    up_bound_per_tog = ['refined_up_bound' + tog for tog in type_of_graph]

    keys = type_of_graph + up_bound_per_tog
    results = {key : [] for key in keys}

    for tog in type_of_graph: 
        if tog== 'Standard': 
            G1, edge_list = make_standard(rel, at, tuples, omega, relational_structure)
        elif tog== 'Standard_with_gaifman': 
            G1, edge_list = make_standard_with_Gaifman(rel, at, tuples, omega, relational_structure)
        elif tog == 'Standard_without_A':
            G1, edge_list = make_standard_without_A(rel, at, tuples, relational_structure)
        elif tog == 'Gaifman':
            G1, edge_list = gaifman(relational_structure)
        elif tog == 'Tuple_click':
            G1, edge_list = make_tuple_click(relational_structure, tuples)


        names = G1.nodes
        index = list(range(len(names)))
        mapping = dict(zip(names, index))
        structure = []
        for node in G1.nodes(data= True):
            a,b = node 
            structure.append(b['structure'])

        node_to_structure = dict(zip(names,structure ))
        X = initialization(G1, dim)
        X = X.to(torch.float32)
        edge_index = torch_geometric.utils.from_networkx(G1).edge_index
        data= Data(x=X, edge_index=edge_index)
        data.G = G1
        nodeGNN = Node_GNN(d_F, nbr_layers)

        X_out = nodeGNN(data.x,data.edge_index)
        X_out = X_out.detach().numpy()

        G1 = incorporate_embeddings(X_out, G1)

        for rate in removing_rate:
            perturbation, poison_count, value_type = 0,0,0
            perturbation = 'remove_tuple'
            type_of_perturbations = 'removing_tuples'
            if type_of_perturbations == 'removing_tuples':
                non_perturbed, perturbed = perturbed_on_tuples(rel, at, tuples, omega, relational_structure, perturbation, rate)

            rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed = perturbed

            if tog== 'Standard': 
                G2, edge_list_p = make_standard(rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed)
            elif tog== 'Standard_with_gaifman':
                G2, edge_list_p = make_standard_with_Gaifman(rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed)
            elif tog == 'Standard_without_A':
                G2, edge_list_p = make_standard_without_A(rel_perturbed, at_perturbed, tuples_perturbed, relational_structure_perturbed)
            elif tog == 'Gaifman':
                G2, edge_list_p = gaifman(relational_structure_perturbed)
            elif tog == 'Tuple_click':
                G2, edge_list_p = make_tuple_click(relational_structure_perturbed, tuples_perturbed)


            X_p = initialization_for_perturbed_graph(X, G1, G2)
            X_p = X_p.to(torch.float32)
            edge_index_p = torch_geometric.utils.from_networkx(G2).edge_index
            data_p= Data(x=X_p, edge_index=edge_index_p)
            data_p.G = G2
            
            X_out_p = nodeGNN(data_p.x, data_p.edge_index)
            X_out_p = X_out_p.detach().numpy()

            G2 = incorporate_embeddings(X_out_p, G2)

            m = 0
            k = 0
            for re in rel_perturbed:
                for tu in tuples_perturbed: 
                    if tu[:3] == re: 
                        m += np.linalg.norm( G1.nodes()[tu]['embedding'] -  G2.nodes()[tu]['embedding'])
                        k +=1

            if m/k >= 0.0000001:
                results[tog].append(m/k)
            else :
                results[tog].append(0.0000001)

            up_bound= np.sqrt(dim)*upper_bound(nodeGNN, G1, G2, X[:,0],features, nbr_layers)
            refined_up_bound = refined_upper_bound(nodeGNN, G1, G2, X[:,0],d_F, nbr_layers)
            if up_bound > 0.00001:
                results['refined_up_bound' + tog].append(refined_up_bound)
            else: 
                results['refined_up_bound'+ tog].append(0.0000001)
    for key in results: 
        if type(results[key][0]) == torch.Tensor : 
            for i in range(len(results[key])):
                results[key][i] = results[key][i].numpy()
                results[key][i] = results[key][i].item()
    return results

def expe_value_removal(working_path, database_name):
    rel, at, tuples, omega, relational_structure = preprocessing(working_path, database_name)

    type_of_graph = ['Standard','Standard_with_gaifman', 'Standard_without_A','Bipartite','Tripartite','Tuple_click'] 

    dim, d_F = 30, [3,4,6,3]
    features, nbr_layers = max(d_F), len(d_F) +1

    low_high_bound = equidepth_without_redundancy(relational_structure, nbr_bin = 5)  
    values_occurences = occurence(relational_structure)
    bounds = []
    for bound in low_high_bound:
        low, high = bound
        bounds.append(f"nbr occurrences: {low} - {high}")

    results = {bound : {tog : [] for tog in type_of_graph} for bound in bounds}

    miss_rate = np.linspace(0.1,0.6,10)    

    for tog in tqdm(type_of_graph): 
        if tog== 'Standard': 
            G1, edge_list = make_standard(rel, at, tuples, omega, relational_structure)
        elif tog== 'Standard_with_gaifman': 
            G1, edge_list = make_standard_with_Gaifman(rel, at, tuples, omega, relational_structure)
        elif tog == 'Standard_without_A':
            G1, edge_list = make_standard_without_A(rel, at, tuples, relational_structure)
        elif tog == 'Gaifman':
            G1, edge_list = gaifman(relational_structure)
        elif tog == 'Tuple_click':
            G1, edge_list = make_tuple_click(relational_structure, tuples)

        names = G1.nodes
        index = list(range(len(names)))
        mapping = dict(zip(names, index))
        structure = []
        for node in G1.nodes(data= True):
            a,b = node 
            structure.append(b['structure'])

        node_to_structure = dict(zip(names,structure ))
        X = initialization(G1, dim)
        
        X = X.to(torch.float32)
        edge_index = torch_geometric.utils.from_networkx(G1).edge_index
        data= Data(x=X, edge_index=edge_index)
        data.G = G1

        nodeGNN = Node_GNN(d_F, nbr_layers)  

        X_out = nodeGNN(data.x, data.edge_index)
        X_out = X_out.detach().numpy()

        G1 = incorporate_embeddings(X_out, G1)

        for i, bound in enumerate(low_high_bound): 
            low_bound, high_bound = bound
            elements = chosen_elements(values_occurences, low_bound, high_bound)

            name = bounds[i]
            for rate in miss_rate:
                rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed = perturbation_missing_value( relational_structure,elements, rate)

                if tog== 'Standard': 
                    G2, edge_list_p = make_standard(rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed)
                elif tog== 'Standard_with_gaifman':
                    G2, edge_list_p = make_standard_with_Gaifman(rel_perturbed, at_perturbed, tuples_perturbed, omega_perturbed, relational_structure_perturbed)
                elif tog == 'Standard_without_A':
                    G2, edge_list_p = make_standard_without_A(rel_perturbed, at_perturbed, tuples_perturbed, relational_structure_perturbed)
                elif tog == 'Gaifman':
                    G2, edge_list_p = gaifman(relational_structure_perturbed)
                elif tog == 'Tuple_click':
                    G2, edge_list_p = make_tuple_click(relational_structure_perturbed, tuples_perturbed)

                X_p = initialization_for_perturbed_graph(X, G1, G2)
                X_p = X_p.to(torch.float32)
                edge_index_p = torch_geometric.utils.from_networkx(G2).edge_index
                data_p= Data(x=X_p, edge_index=edge_index_p)
                data_p.G = G2
                X_out_p = nodeGNN(data_p.x, data_p.edge_index)
                X_out_p = X_out_p.detach().numpy()
                G2 = incorporate_embeddings(X_out_p, G2)
                m = 0
                k = 0
                for re in rel_perturbed:
                    for tu in tuples_perturbed: 
                        if tu[:3] == re: 
                            m += np.linalg.norm( G1.nodes()[tu]['embedding'] -  G2.nodes()[tu]['embedding'])
                            k +=1
                if m/k >= 0.0000001:
                    results[name][tog].append(m/k)

                else :
                    results[name][tog].append(0.0000001)
    for key in results: 
        if type(results[key][0]) == torch.Tensor : 
            for i in range(len(results[key])):
                results[key][i] = results[key][i].numpy()
                results[key][i] = results[key][i].item()
    return results
