import sqlite3
import yaml

from db_utils.models import Record


class SumatraDB(object):
    """
    Used to connect to the sumatra database.
    """
    def __init__(self,
                 db=".smt/records",
                 record_table="django_store_record",
                 param_table="django_store_parameterset"):

        self.db = db
        self.record_table = record_table
        self.param_table = param_table
        self.con = sqlite3.connect(db)
        # returns a dictonary per queried row
        self.con.row_factory = self._dict_factory

    def get_cursor(self):
        return self.con.cursor()

    def _dict_factory(self, cursor, row):
        """
        Source: https://stackoverflow.com/questions/3300464/how-can-i-get-dict-from-sqlite-query
        """
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def get_params_dic(self, params_id):
        """
        Query config file content corresponding to 'params_id'.

        Arg:
            - params_id: identifies a row in the parameter table
              of sumatra

        Return:
            - dictionary corresponding to the yaml config file
              which 'params_id' points to
        """
        c = self.get_cursor()
        t = (self.param_table, params_id)
        c.execute("select content from {} where id={}".format(*t))

        return yaml.load(c.fetchone()["content"])

    def query(self, query):
        """
        Arg:
            - query: query string

        Return:
            - rows (as dictionaries) fetched from the sumatra
              database
        """
        c = self.get_cursor()
        c.execute(query)

        return c.fetchall()

    def get_all_records(self, columns):
        """
        Queries all the records from the database.

        Arg:
            - columns: list of column names that are selected

        Return:
            - list of Record objects correponding to all the
              records in the database
        """
        c = self.get_cursor()
        col_s = ",".join(columns)

        q = "select {} from {}".format(col_s, self.record_table)
        c.execute(q)

        res = c.fetchall()

        return list(map(lambda x: Record(x), res))

    def get_filtered_by_label(self, columns, record_label):
        """
        Queries all the records whose label starts with
        'records_label'.

        Arg:
            - columns: list of column names that are selected

        Return:
            - list of Record objects
        """
        c = self.get_cursor()
        col_s = ",".join(columns)

        q = "select {} from {} where label like '{}%'".format(
            col_s, self.record_table, record_label
        )
        c.execute(q)

        res = c.fetchall()

        return list(map(lambda x: Record(x), res))
