from twarc_csv import CSVConverter
import logging
import glob


DATA_PATH = '../../twitter_data/'


def main():
    logging.basicConfig(filename='csv_converter.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(message)s')

    for file in glob.glob(DATA_PATH + '**/*.jsonl', recursive=True):
        file_name = file.split('/')[-1].split('.')[0]
        logging.info(f'Converting {file_name}')
        if 'test' not in file_name:
            with open(file, 'r') as infile:
                with open(f'./out/{file_name}.csv', 'w') as outfile:
                    converter = CSVConverter(infile=infile, outfile=outfile)
                    converter.process()


if __name__ == "__main__":
    main()
