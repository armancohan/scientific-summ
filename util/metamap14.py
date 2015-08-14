import codecs
from hashlib import md5
import json
import os
import subprocess
import sys
import tempfile

from cache import CachingDecorator, cache_file
from common import hash_file, prep_for_json, hash_obj
CACHEDIR = "cache/"
MMJAR_PATH = "/home/util/MetamapConcepts.jar"


@CachingDecorator
def _run(infn, mmjar, cache_file=None, no_cache=None):
#     tmp = tempfile.NamedTemporaryFile()
#     outfn = tmp.name

    cmd = ["java", "-Xmx2g", "-jar", mmjar, infn]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate()  

    if err is not None:
        print >> sys.stderr, "MetaMap error:", err
        print >> sys.stderr, "\tcommand run:", cmd
        sys.exit(1)

#     f = codecs.open(outfn, "r", encoding="utf-8")
#     lines = "\n".join(f.readlines())
#     f.close()
#     tmp.close()

    # add dict brackets and removing trailing ','
    js = json.loads(out)
    return js


def run(mm_input,
        mmjar=MMJAR_PATH,
        no_cache=False,
        long_concepts=True):
    """
        Run metamap on mm_input

        Parameters
        ----------
        mm_input : str/dict
            Object to process. It could be either a string
            (both utf-8 and ASCII are supported) or a dictionary
            with the following format:
            {   'key_1': 'text_1',
                ...
                'key_n': 'text_n',
            }

        mmjar : str (optional, default=MMJAR_PATH)
            Location of the metamap.jar file

        no_cache : bool (optional, default=False)
            Minimize the amount of cache generated, so
            that if a mm_input is queried twice to
            metamap, the server in inquired twice.

        Returns
        ----------
        resp : dict
            Metamap response. 
            {phrases:<string list of phrases>, concepts:<object list of concepts>}
            note:   if the list of concepts is empty, metamap has
                    found no concepts in mm_input
            note 2: all the string fields are unicode fields!
        e.g. output:
        {u'phrases': [u'Previous results', u'showed', u'that oncogenic Ras',
        u'elevates', u'ribonucleotide synthesis largely through the nonoxidative branch of the PPP'],
        u'concepts': [{u'ConceptId': u'C0205156', u'PreferredName': u'Previous',
        u'SemanticTypes': [u'tmco'], u'ConceptName': u'Previous',
        u'MatchedWords': [u'previous']}, {u'ConceptId': u'C1274040',
        u'PreferredName': u'Result', u'SemanticTypes': [u'ftcn'],
        u'ConceptName': u'Result', u'MatchedWords': [u'result']},
        {u'ConceptId': u'C1547282', u'PreferredName': u'Show', u'SemanticTypes': [u'inpr'],
        u'ConceptName': u'Show', u'MatchedWords': [u'show']}, {u'ConceptId': u'C0007090',
        u'PreferredName': u'Carcinogens', u'SemanticTypes': [u'hops'],
        u'ConceptName': u'Oncogens', u'MatchedWords': [u'oncogens']},
        {u'ConceptId': u'C0034678', u'PreferredName': u'ras Oncogene',
        u'SemanticTypes': [u'gngm'], u'ConceptName': u'RAS',
        u'MatchedWords': [u'ras']}, {u'ConceptId': u'C0205250',
        u'PreferredName': u'High', u'SemanticTypes': [u'qlco'],
        u'ConceptName': u'elevate', u'MatchedWords': [u'elevate']},
        {u'ConceptId': u'C1157684', u'PreferredName': u'ribonucleotide biosynthetic process',
        u'SemanticTypes': [u'moft'], u'ConceptName': u'ribonucleotide synthesis',
        u'MatchedWords': [u'ribonucleotide', u'synthesis']}, {u'ConceptId': u'C1253959',
        u'PreferredName': u'Branch of', u'SemanticTypes': [u'bpoc'],
        u'ConceptName': u'Branch of', u'MatchedWords': [u'branch', u'of']}, 
        {u'ConceptId': u'C0031106', u'PreferredName': u'Periodontitis,
        Juvenile', u'SemanticTypes': [u'dsyn'], u'ConceptName': u'PPP',
        u'MatchedWords': [u'ppp']}]}
    """
    resp = None

    if type(mm_input) is dict:
        resp = _qrun(mm_input, mmjar)
    if type(mm_input) in (str, unicode):
        resp = _txtrun(mm_input, mmjar, no_cache, long_concepts)

    if resp:
        return resp
    else:
        print ('[metamap error] input for run should be dict or '
               'str/unicode, not %s') % type(mm_input)
        sys.exit(1)


def process_text(text, mmjar, no_cache=False, long_concepts=False, semantic_types=None):
    '''
    like metamap run, just returns a string of concepts, instead of json
    '''
    if not text:
        print '[metamap info] empty query'
        return {'txt': {'concepts': []}}
    concepts = run(text, mmjar, no_cache, long_concepts)
    strres = ''
    if not semantic_types:
        for concept in concepts['txt']['concepts']:
            strres += str(concept['cname'])
    else:
        for concept in concepts['txt']['concepts']:
            if str(concept['semtype'][0]) in semantic_types.keys():
                found_concept = {}
                found_concept['cname'] = str(concept['cname'])
                for t in concept['semtype']:
                    found_concept['ctype'] = t
                strres += str(concept['cname']) + ' '
    return strres.strip()


def _txtrun(text, mmjar, no_cache, long_concepts):

    if not text:
        print '[metamap info] empty query'
        return {'txt': {'concepts': []}}

#     tmp = tempfile.NamedTemporaryFile()
#     tfn = tmp.name
# 
#     with codecs.open(tfn, 'wb', encoding='utf-8') as f:
#         print >> f, u"txt:{0}".format(text)
    cf = cache_file(CACHEDIR, [hash_obj(text), hash_file(mmjar)], "metamap")

    # for some inesplicable reason, the Java MetaMap API client
    # throws a java.lang.StringIndexOutOfBoundsException exception
    # instead of gracefully returning nothing when a text has no concepts.
    try:
        d = _run(text, mmjar,  no_cache=no_cache, cache_file=cf)
    except ValueError:
        print '[metamap info] no concepts found'
        d = {'txt': {'concepts': []}}

    return d
#     if long_concepts:
#         return longest_concepts(d)
#     else:
#         return d


def _qrun(queries, mmjar):
    out = {}

    # run one query at a time for more efficient caching
    for k, v in queries.iteritems():
        q = {k: v}
        tmp = tempfile.NamedTemporaryFile()
        qfn = tmp.name

        f = codecs.open(qfn, "w", encoding="utf-8")
        for qid, qtxt in q.iteritems():
            print >> f, "%s:%s" % (qid, qtxt)
        f.close()

        d = _run(qfn, mmjar, cache_file=cache_file(CACHEDIR,
                                                   [hash_file(qfn),
                                                    hash_file(mmjar)],
                                                   "metamap"))
        d = longest_concepts(d)

        for kk, vv in d.iteritems():
            if kk in out:
                print >> sys.stderr, "error: duplicate key:", kk
                print >> sys.stderr, out
                sys.exit(1)
            else:
                out[kk] = vv

        tmp.close()

    return out


def batch(data, fields=None, cachedir='../cache',
          mmjar=MMJAR_PATH, no_cache=False,
          long_concepts=True):
    """ batch process all the elements in data and cache
        them in a single file (reduces IO time).
        data is an list of dictionaries. If fields=None, then
        all the fields in every dictionary are cached;
        fields should be a list/tuple containing the relevant
        fields to consider.
    """

    if not no_cache:
        digest_data = ('batch_mm_{0}.cache'
                       ''.format(md5(json.dumps([fields, data])).hexdigest()))
        if not os.path.exists(cachedir):
            print >> sys.stderr, '[cache error] %s does not exists' % cachedir
            sys.exit(1)
        cache_path = os.path.join(cachedir, digest_data)
        if os.path.exists(cache_path):
            with codecs.open(cache_path, mode='rb', encoding='utf-8') as f:
                return json.load(f)

    out = []
    for elem in data:
        # if fields is not specified, then elem_fields == elem.keys()
        # otherwise list comprehension acts like a filter function
        elem_fields = [k for k in elem.keys()
                       if (not(fields) or (k in fields))]

        result = {fl: run(elem[fl],
                          no_cache=True,
                          mmjar=mmjar,
                          long_concepts=long_concepts)
                  for fl in elem_fields}
        out.append(result)

    if not no_cache:
        out = prep_for_json(out)
        with codecs.open(cache_path, mode='wb', encoding='utf-8') as f:
            json.dump(out, f)

    return out


def batch_filtered(data, semtypes, fields=None, cachedir='../cache',
                   mmjar=MMJAR_PATH, no_cache=False,
                   long_concepts=True):
    """ exactly like batch with the following difference:
        It returns only concepts and their semantic types
        and only for semantic types that are indicated in semtypes
        the output format is a list of concepts with only concept name
         and its semantic type
        default semtypes path is: data/semantic_types.json
    """

    if not no_cache:
        digest_data = ('batch_mm_filtered_{0}.cache'
                       ''.format(md5(json.dumps([fields, data])).hexdigest()))
        if not os.path.exists(cachedir):
            print >> sys.stderr, '[cache error] %s does not exists' % cachedir
            sys.exit(1)
        cache_path = os.path.join(cachedir, digest_data)
        if os.path.exists(cache_path):
            with codecs.open(cache_path, mode='rb', encoding='utf-8') as f:
                return json.load(f)

    out = []
    for elem in data:
        # if fields is not specified, then elem_fields == elem.keys()
        # otherwise list comprehension acts like a filter function
        elem_fields = [k for k in elem.keys()
                       if (not(fields) or (k in fields))]

        result = {fl: run(elem[fl],
                          no_cache=True,
                          mmjar=mmjar,
                          long_concepts=long_concepts)
                  for fl in elem_fields}

        found_concepts = {}
        for fl in elem_fields:
            found_concepts[fl] = []
            for concept in result[fl]['txt']['concepts']:
                if str(concept['semtype'][0]) in semtypes:
                    found_concept = {}
                    found_concept['cname'] = str(concept['cname'])
                    for t in concept['semtype']:
                        found_concept['ctype'] = t
                    found_concepts[fl].append(found_concept)

        out.append(found_concepts)

    if not no_cache:
        out = prep_for_json(out)
        print out
        with codecs.open(cache_path, mode='wb', encoding='utf-8') as f:
            json.dump(out, f)

    return out


def longest_concepts(qconcepts):
    ''' Returns a qconcepts without overlapping concepts '''
    longqc = {}

    for qid in qconcepts.keys():
        for con in qconcepts[qid]["concepts"]:
            con['positions'] = [set(range(start, start + length)) for (start, length) in
                                [eval(p) for p in con['positions']]]

        longqc[qid] = {"concepts": []}
        for i, con1 in enumerate(qconcepts[qid]["concepts"]):
            keep = True

            for j, con2 in enumerate(qconcepts[qid]["concepts"]):
                if i == j:
                    continue

                if _contains(little=con1, big=con2):
                    keep = False
                    break

            if keep:
                longqc[qid]["concepts"].append(con1)

    return longqc


def _contains(little, big):
    ''' Returns true if every instance of little is contained within an instance of big '''
    for littlepos in little['positions']:
        for bigpos in big['positions']:
            if not littlepos.issubset(bigpos):
                return False

    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "usage: %s Str" % sys.argv[0]
        sys.exit(1)

#     js = run(json.load(open(sys.argv[1]))["questions"])
    js = run(sys.argv[1])
    print "metamap done; received:"
    print js
