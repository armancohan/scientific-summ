import MySQLdb
import json
with open('.mysqlconf', 'rb') as f:
    mysql_conf = json.load(f)


class UMLS_Query():

    def __init__(self):
        self.db = MySQLdb.connect(host=mysql_conf['host'],
                                  port=mysql_conf['port'],
                                  user=mysql_conf['user'],
                                  passwd=mysql_conf['passwd'],
                                  db=mysql_conf['db'])
        self.cur = self.db.cursor()

    def getCUI(self, field, value):
        """
        Get CUI based on a designated field
        :field can be one of AUI, SUI, LUI,
             CUI, STR (case sensitive)
        :value value of the corresponding field
        return a list of CUIs.
        """
        q = 'select distinct CUI from MRCONSO where %s="%s"'\
            % (field, value)
        self.cur.execute(q)
        # result = self.cur.fetchall()
        syns = set([x for x, in self.cur.fetchall()])
        result = list(syns)
        return result

    def getAUI(self, field, value):
        """
        Get AUI (Atom Unique Identifier)
         based on a designated field
        :field can be one of AUI, SUI, LUI,
             CUI, STR (case sensitive)
        :value value of the corresponding field
        return a list of AUIs.
        """
        q = 'select distinct AUI from MRCONSO where %s="%s"'\
            % (field, value)
        self.cur.execute(q)
        # result = self.cur.fetchall()
        syns = set([x for x, in self.cur.fetchall()])
        result = list(syns)
        return result

    def getStr(self, field, value, only_prefered=True):
        """
        Get String of a concept/atom/string
         based on a designated field
        :field can be one of AUI, SUI, LUI,
             CUI, STR (case sensitive)
        :value value of the corresponding field
        return a list of Strings.
        """
        # TS (Term status) = P (Preferred LUI of the CUI)
        # STT   (String Type) = PF (Preferred form of term)
        if field == 'STR':
            q = 'select distinct STR from MRCONSO where STR like "%' +\
                value + '%"'
        else:
            q = 'select distinct STR from MRCONSO where %s="%s"'\
                % (field, value)
        if only_prefered:
            q += ' and TS="P" and STT="PF" and ISPREF="Y"'
        q += ' and LAT="ENG"'
        self.cur.execute(q)
        # result = self.cur.fetchall()
        syns = set([x for x, in self.cur.fetchall()])
        result = list(syns)
        return result

    def getSrcAbb(self, field, value):
        """
        Get Source Abbreviation of a concept/atom/string
        the Metathesaurus has "versionless" or "root" Source
             Abbreviations (SABs) in the data files.
        e.g. MeSH -> MSH
        :field can be one of AUI, SUI, LUI,
             CUI, STR (case sensitive)
        :value value of the corresponding field
        return a list of SABs.
        """
        # TS (Term status) = P (Preferred LUI of the CUI)
        # STT   (String Type) = PF (Preferred form of term)
        q = 'select distinct SAB from MRCONSO where %s="%s"'\
            % (field, value)
        self.cur.execute(q)
        # result = self.cur.fetchall()
        syns = set([x for x, in self.cur.fetchall()])
        result = list(syns)
        return result

    def getParents(self, field, value, SAB=False, relation=False):
        """
        Gets parents for a CUI or AUI
        SAB is source abbreviations, can be:
        'MSH',  'SNOMEDCT_US', etc
        relation can be: isa, etc
        Returns a tuple of size 2
        first element is AUIs of all the parents
        second element is the immediate parent
        e.g.
        (['A0398286','A0130731','A0487444'],'A0487444')
        """
        if (field != 'CUI' and field != 'AUI'):
            print 'ERROR (nothing returned), Field should be CUI or AUI'
            return None
        q = 'select distinct PTR, PAUI from MRHIER where %s="%s"'\
            % (field, value)
        if SAB:
            q += ' and SAB="%s"' % SAB
        if relation:
            q += ' and RELA="%s"' % relation
        self.cur.execute(q)
        # result = self.cur.fetchall()
        syns = self.cur.fetchall()
        if len(syns) > 0:
            all_parents = syns[0][0].split('.')
            return (all_parents, syns[0][1])
        else:
            print syns
            return syns

    def getChildren(self, field, value,  SAB=False, relation=False):
        """
        Gets children for a CUI or AUI
        :Field recommended=AUI (Much faster)
        :SAB is source abbreviations, can be:
        'MSH',  'SNOMEDCT_US', etc
        :relation can be: isa, etc
        ...
        Returns a tuple of size 2
        first element is AUIs of all the parents
        second element is the immediate parent
        e.g.
        (['A0398286','A0130731','A0487444'],'A0487444')
        """
        if (field == 'CUI'):
            q = 'select distinct m2.CUI from MRHIER, MRCONSO'\
                + ' as m1, MRCONSO as m2 '\
                + ' where MRHIER.PAUI = m1.AUI and m1.CUI'\
                + ' = "%s" and MRHIER.AUI = m2.AUI' \
                % value
            if SAB:
                q += ' and MRHIER.SAB="%s"' % SAB
            if relation:
                q += ' and MRHIER.RELA="%s"' % relation
        elif (field == 'AUI'):
            q = 'select distinct AUI from MRHIER where PAUI="%s"'\
                % value
            if SAB:
                q += ' and SAB="%s"' % SAB
            if relation:
                q += ' and RELA="%s"' % relation
        else:
            print 'ERROR (nothing returned), Field should be CUI or AUI'
            return None

        self.cur.execute(q)
        # result = self.cur.fetchall()
        syns = self.cur.fetchall()
        return syns

    def test(self):
        q = 'SELECT * FROM MRHIER WHERE CUI = "C0032344"'
        self.cur.execute(q)
        q = 'Show columns from MRHIER'
        syns = self.cur.fetchall()

        self.cur.execute(q)
        s = self.cur.fetchall()
        print s
        return syns

if __name__ == '__main__':
    umls = UMLS_Query()
#     print umls.getAUI('CUI', 'C0006826')
    # print umls.getStr('CUI', 'C0006826')
    # print umls.getSrcAbb('CUI', 'C0006826')
# a = umls.getParents('CUI', 'C0006826', SAB='SNOMEDCT_US')
    print umls.getStr('STR', 'iKras')
# for i in a[0]:
#   print "for AUI: %s " %i +  str(umls.getEverything('AUI', i))
    # print umls.getChildren('CUI', 'C0006826', SAB='SNOMEDCT_US', relation="isa")
    # print umls.test()
