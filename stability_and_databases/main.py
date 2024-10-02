
import argparse
import utils
import os 
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--database_name',
        default = 'mondial',
        choices=["tpce", "genes", "hepatitis", "mondial", "mutagenesis", "world"],
        type=str
    )
    parser.add_argument(
        '--path',
        type=str
    )
    args = parser.parse_args() 
    if args.type_of_expe == 'tuple_removal':
        results = utils.expe_tuple_removal(args.path, args.database_name)
    elif args.type_of_expe == 'value_removal':
        results = utils.expe_value_removal(args.path, args.database_name)

    with open(args.type_of_expe + '.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    