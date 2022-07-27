import argparse
import os
import posixpath
import pandas as pd
from tqdm import tqdm
from collections import Counter
import win32com.client as win32

def normalize_path(path : str):
    return path.replace(os.sep, posixpath.sep)

def write_meta(path, file_name, output_path):

    abs_path = os.path.abspath(os.path.join(path, file_name[0]))
    print('loading file no.1')
    data = pd.read_excel(abs_path, engine='openpyxl')
    if len(file_name)>1:
        for i in range(1, len(file_name)):
            new_path = os.path.abspath(os.path.join(path, file_name[i]))
            print('loading file no.{}'.format(i+1))
            new_data = pd.read_excel(new_path, engine='openpyxl')
            data = pd.concat([data, new_data], axis=0)

    filename_strip = [i[:4] for i in file_name]
    out_filename = '-'.join(filename_strip) + '.xlsx'
    writer = pd.ExcelWriter(os.path.abspath(os.path.join(output_path, out_filename)), engine='xlsxwriter')

    print('counting column value...')
    for col in tqdm(data.columns):
        col_cnt = Counter(data.loc[:, col])
        cnt_df = pd.DataFrame(col_cnt, index=['count']).T
        cnt_df = cnt_df.sort_values(by='count', ascending=False)
        cnt_df.to_excel(writer, sheet_name=col)

    writer.save()
    return out_filename, len(data.columns)

def normalize_xlsx(input_path, filename, column_cnt, output_path):

    exc = win32.Dispatch('Excel.Application')
    wb = exc.Workbooks.Open(os.path.abspath(os.path.join(input_path, filename)))
    for i in range(1, column_cnt+1):
        exc.Worksheets(i).Activate()
        exc.ActiveSheet.Columns.AutoFit()

    wb.SaveAs(os.path.abspath(os.path.join(output_path, filename)))
    wb.Close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', help='excel data path')
    parser.add_argument('--file_names', type=str, default='', help='excel file name')
    parser.add_argument('--output_path', type=str, default='', help='excel output path')
    parser.add_argument('--normalize', type=bool, default=True, help='column cell width normalize')

    args = parser.parse_args()
    path = normalize_path(args.input_path)
    file_names = args.file_names
    file_names = [i for i in file_names.split(',')]
    output_path = normalize_path(args.output_path)
    normalize = args.normalize

    if len(file_names)<1:
        raise AssertionError('need 1 excel file at least')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename, column_cnt = write_meta(path, file_names, output_path)
    if normalize:
        normalize_xlsx(output_path, filename, column_cnt, output_path)
    print('finished')

if __name__ == '__main__':
    main()
