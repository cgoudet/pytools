import itertools
import re
import pandas as pd


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable

    :param list seq: sequence from which to extract subsequences
    :param int n: number of elements to extract at each iteration

    If sequence is smaller than required window width, return empty string

    :Example:

    ```
    window(s) # -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    ```

    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def multi_replace(string, replacements):
    """Given a string and a replacement map, it returns the replaced string.

    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}

    :return: modified string
    :rtype: str

    This code is an adaptation of : https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729
    """
    if not len(replacements):
        return string

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


def write_df(df, filename, names=[]):
    """
    Write a pandas dataframe in csv or excel.

    :param DataFrame df: DataFrame to save
    :param str filename: output file name

    The format of the ouput file is determined by the extension :
    - xlsx, xls : excel
    - rest csv
    """
    if type(df).__name__ == 'DataFrame':
        df = [df]

    is_excel = filename.suffix in ['.xlsx', '.xls']
    writer = pd.ExcelWriter(filename) if is_excel else None

    for i, f, n in enumerate(itertools.zip_longest(df, names, fillvalues='')):
        n = n or 'sheet{}'.format(i)
        if is_excel:
            f.to_excel(writer, n)
        else:
            f.to_csv(filename.with_suffix('_'+n))

    if writer:
        writer.save()


def spreadsheet(filename, sheet_name=0, *args, **kwargs):
    """
    Return the file content as a pandas DataFrame.
    If not excel, the file is interpreted as csv.

    :param str filename: input file. Excel extensions .xls, .xlsx
    :param str,int sheet_name: Name of number of the sheet to read in case of excel
    """
    data = (
        pd.read_excel(str(filename), sheet_name=sheet_name, *args, **kwargs)
        if filename.suffix in ['.xlsx', '.xls'] else
        pd.read_csv(str(filename), *args, **kwargs))

    return data
