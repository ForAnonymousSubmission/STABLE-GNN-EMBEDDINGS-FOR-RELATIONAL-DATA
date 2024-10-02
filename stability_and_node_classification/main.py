
import argparse
import utils
import pandas as pd

if __name__ == '__main__':
    result_list = []
    for name in ['CiteSeer', 'PubMed', 'cora']:
        inter = utils.make_expe(name) 
        inter = pd.DataFrame.from_dict(inter)
        print(inter)
        result_list.append(inter)
    for df,name in zip(result_list,['CiteSeer', 'PubMed', 'cora']):
        df.to_csv(name + '_result'+'.csv')
    