import pandas as pd
import string


def request_from_file(filename, engine, template_args={}):
    """
    return the Dataframe resulting in executing the content of the filename
    """
    with open(filename) as rq:
        query = string.Template(rq.read()).substitute(**template_args)
        frame = pd.read_sql(query, engine)
        return frame
