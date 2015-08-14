import unicodedata

import MySQLdb

from util.metamap import run as mmrun

class MMExpand:
    def run(self, args):
        qconcepts = mmrun(self.tokquestions)

        db = MySQLdb.connect(user="root",
                             passwd="lollipop11",
                             db="umls")
        self.cur = db.cursor()

        queries = {}
        for qid, qtxt in self.tokquestions.iteritems():
            qtxt = unicodedata.normalize('NFKD',
                                         qtxt).encode('ascii', 'ignore')
            qterms = [qtxt]

            for cdata in qconcepts[qid]["concepts"]:
                newterms = self.expand_concept(cdata)
                if newterms is not None:
                    qterms.extend(newterms)
                    self.added[qid].extend(newterms)

            queries[qid] = " ".join(qterms)

        self.write_queries(queries)

    def check_args(self, args):
        self.ttys = ['SY']

        ttygroups = {"syns": ('AUN', 'EQ', 'SYN', 'MTH'),
                     "chemicals": ('CCN', 'CSN'),
                     "drugs": ('BD', 'BN', 'CD', 'DP', 'FBD', 'GN', 'OCD'),
                     "diseases": ('DI', ), "findings": ('FI', ),
                     "hierarchy": ('HS', 'HT', 'HX'), "related": ('RT', ),
                     "preferred": ('PTN', 'PT')}

        if len(args) > 3:
            self.ttys = []

            for tty in args[3:]:
                if tty in ttygroups:
                    self.ttys.extend(ttygroups[tty])
                else:
                    self.ttys.append(tty)

    def expand_concept(self, cdata):
        # TODO check semtype to see if it should be expanded
        return self.concept_synonyms(cdata['cid'])

    def concept_synonyms(self, cui):
        termtypes = ("and (TTY=" +
                     " OR TTY=".join(["'%s'" % x for x in self.ttys]) + ")")

        self.cur.execute("select STR from MRCONSO where " +
                         "CUI = '%s' and LAT = 'ENG' and ISPREF = 'Y'" % cui +
                         termtypes + " and SAB != 'CHV'")

        syns = set(filter(lambda y: y.replace(" ", "").isalpha(),
                          [x.lower() for x, in self.cur.fetchall()]))

        return syns

if __name__ == "__main__":
    import sys
    m = Method(sys.argv)
