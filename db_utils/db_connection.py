import sqlite3
import yaml

from db_utils.models import Record


class SumatraDB(object):
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
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def get_params_dic(self, params_id):
        c = self.get_cursor()
        t = (self.param_table, params_id)
        c.execute("select content from {} where id={}".format(*t))

        return yaml.load(c.fetchone()["content"])

    def query(self, query):
        c = self.get_cursor()
        c.execute(query)

        return c.fetchall()

    def get_all_records(self, columns):
        c = self.get_cursor()
        col_s = ",".join(columns)

        q = "select {} from {}".format(col_s, self.record_table)
        c.execute(q)

        res = c.fetchall()

        return list(map(lambda x: Record(x), res))
