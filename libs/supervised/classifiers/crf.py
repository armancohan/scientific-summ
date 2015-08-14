import MBSP
from util.cache import (simple_caching, CacheOperator)
from util.common import (hash_obj, VerbosePrinter)
from libs.supervised.supervised_interface import SupervisedInterface
from tempfile import NamedTemporaryFile
import numpy as np
import subprocess
import codecs
import os
import re
from util.tokenizer import sent_tokenize

# Not configured for multi-threading, yet.
# MBSP.config.threading = True

# Optimize MBSP for biomedical literature
MBSP.tokenizer.BIOMEDICAL = True
GENIAPATH = 'geniatagger-3.0.1/'


class Supervised(SupervisedInterface):
    """docstring for SupervisedInterface"""

    supervised_opts = {'geniapath': {'default': GENIAPATH},
                       'nlp-tool': {'default': 'MBSP',
                                    'choices': ['MBSP', 'genia']}}

    def __init__(self, args, opts):
        super(Supervised, self).__init__(args, opts)
        self.cachedir = self.opts.cachedir
        self.printer = VerbosePrinter(self.opts.verbose)

        if hasattr(self.opts, 'crfpp_template'):
            self.template_path = self.opts.crfpp_template
        else:
            self.template_path = 'crfpp_templates/test.crfpp'
            self.printer('[warning] no crf template in opts. using %s' %
                         self.template_path)

        self.tmpdir = getattr(self.opts, 'tmpdir', None)

    @simple_caching()
    def _chunk_MBSP(self, txt):
        chunked = MBSP.chunk(txt)
        return unicode(chunked)

    @simple_caching()
    def _parse_MBSP(self, txt):
        parsed = MBSP.parse(txt)
        return unicode(parsed)

    @simple_caching()
    def _tokenize_MBSP(self, txt):
        tokenized = MBSP.tokenize(txt)
        return unicode(tokenized)

    @simple_caching()
    def _lemmatize_MBSP(self, txt):
        lemmatized = MBSP.lemmatize(txt)
        return unicode(lemmatized)

    def _process_geniatagger(self, li_txt):
        ''' Process data using geniatagger;
            unlike MBSP, geniatagger is best used with
            as many sentences as possible, all at once.
        '''
        if not hasattr(self, 'co_gt'):
            self.co_gt = CacheOperator(self.cachedir,
                                       prefix='geniatagger',
                                       mute=True)

        # after being processed by geniatagger, sentences are
        # cached individually. Therefore, as a first step,
        # each sentence is hashed and retrieved from the cache
        # folder if possible.
        cached = {}
        to_cache = []
        for s in li_txt:
            c = self.co_gt.get_cache(s)
            if c:
                # data is found, no need to query it to tagger
                cached[s] = c
            else:
                # data meeds to be tagged
                to_cache.append(s)

        self.printer('[geniatagger] %s cached, %s to generate' %
                     (len(cached), len(to_cache)))

        if len(to_cache) > 0:
            # save previous path, move to geniatagger folder
            pwd = os.getcwd()
            try:
                os.chdir(self.opts.geniapath)
            except OSError:
                os.chdir(os.path.join(pwd, self.opts.geniapath))

            with NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            with codecs.open(temp_path, 'wb', 'utf-8') as train_file:
                train_file.write('\n'.join([re.sub(r'\n+', ' ', s.strip())
                                            for s in to_cache]))

            proc = subprocess.Popen(['./geniatagger', temp_path],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            msg_out, msg_err = proc.communicate()
            msg_err = '\n'.join([e for e in msg_err.split('\n')
                                 if e.find('loading') < 0])
            if msg_err:
                err_msg = ('subprocess exited with error \n"%s"' %
                           '\n'.join(['\t%s' % l
                                      for l in msg_err.split('\n')]))

                ####
                with codecs.open(temp_path, encoding='utf-8') as f:
                    d = f.readlines()
                el = [int(v) for v in re.findall(r'\d+', msg_err)]
                for pos in el:
                    print pos
                    print type(d), len(d)
                    print d[el]
                ####
                raise OSError(err_msg)

            # set the path back to correct folder
            os.chdir(pwd)

            # fix spacing issue in the genia-tagged data
            msg_out = re.sub(r'\n\n\n+', '\n\n', msg_out)

            # append new data to cached files
            for i, c in enumerate(msg_out.strip().split('\n\n')):
                cached[to_cache[i]] = unicode(c, "UTF-8")

            # cache new data
            for e in to_cache:
                self.co_gt.set_cache(e, cached[e])
            self.printer('[geniatagger] tagging complete')
        else:
            self.printer('[geniatagger] no need to invoke tagger')

        # return data in the correct order
        data_out = u''
        for e in li_txt:
            c = cached[e]
            data_out += (c + u'\n\n')
        data_out = data_out.strip()

        return data_out

    # def _split_sentences

    def _genia_process(self, raw_X, Y=None):
        X = []
        Y = Y if Y is not None else ['' for j in range(len(raw_X))]
        sents_cnt = []
        for x in raw_X:
            lt = sent_tokenize(x)
            sents = [s.strip() for s in lt['sentences']]
            X.extend(sents)
            sents_cnt.append(len(sents))

        tagged_data = self._process_geniatagger(X)

        ann_data = []
        i = 0
        for l in tagged_data.split('\n'):
            # this counter is used to reference the right
            # element of Y: when an empty string is found,
            if l == '':
                if sents_cnt[0] == 1:
                    i += 1
                    ann_data.append('')
                    sents_cnt.pop(0)
                else:
                    sents_cnt[0] -= 1
                continue
            ann_data.append(l.split('\t')[1:] + [str(Y[i])])

        return ann_data

    def _print_crfpp(self, msg_out):
        prefix = '[CRF++] '
        ps = ' ' * len(prefix)
        msg_out = msg_out.strip().split('\n')
        crf_iter_cnt = sum([1 for m in msg_out if m.find('iter=') == 0])
        msg_out = [ps + m for m in msg_out if m.find('iter=') < 0][3:]
        msg_out.insert(-1, ps + '# iterations: %s' % crf_iter_cnt)
        msg_out.insert(0, '')
        msg_out.insert(-1, '')
        msg_out.insert(0, prefix + 'run output:')
        self.printer('\n'.join(msg_out))

    def _MBSP_process(self, X, Y=None):
        ann_data = []

        self.printer('[info] Preprocesing %s elems with MBSP...' %
                     len(X))
        cnt_elem = 0
        for txt, y in zip(X, (Y if Y is not None
                              else ['' for i in range(len(X))])):
            if cnt_elem % 1000 == 0 and cnt_elem > 0:
                self.printer('[info] %s preprocessed...' % cnt_elem)
            cnt_elem += 1

            # retrieve lemma and chunk info
            cc = hash_obj(txt)
            chunked = self._chunk_MBSP(txt, cache_comment=cc)
            lemmatized = self._lemmatize_MBSP(txt, cache_comment=cc)

            # append each token in sentence in a format
            # compatible with CRF++
            ann_data.extend([[l] + c.split('/')[1:] + [str(y)] for l, c in
                             zip(lemmatized.split(), chunked.split())])

            # separate each sentence with a blank line.
            ann_data.append([''])
        self.printer('[info] preprocessing completed '
                     '(%s processed).' % cnt_elem)
        return ann_data

    def train(self, X_train, y_train):
        # get data in the right format for being processed by crf++
        if self.opts.nlp_tool == 'MBSP':
            train_data = self._MBSP_process(X_train, y_train)
        else:
            train_data = self._genia_process(X_train, y_train)

        # create temp file, open it as unicode file and train data on it
        with NamedTemporaryFile(delete=False, dir=self.tmpdir) as train_file:
            train_path = train_file.name
        with codecs.open(train_path, 'wb', 'utf-8') as train_file:
            for l in train_data:
                train_file.write('%s\n' % ' '.join(l))

        # create a temp file that contains the model
        # "self" is needed since the file needs to be referenced in run
        with NamedTemporaryFile(delete=False, dir=self.tmpdir) as model_file:
            self.model_path = model_file.name

        # run crf++ in a subprocess and collect stdout, stderr
        # raise error if strerr is not empty
        proc = subprocess.Popen(['crf_learn', self.template_path,
                                train_path, self.model_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        msg_out, msg_err = proc.communicate()
        if msg_err:
            err_msg = ('subprocess exited with error \n"%s"' %
                       '\n'.join(['\t%s' % l for l in msg_err.split('\n')]))
            raise OSError(err_msg)

        # print crf++ output (only if opts.verbose == 1)
        self._print_crfpp(msg_out)

    def run(self, X_test):
        if self.opts.nlp_tool == 'MBSP':
            test_data = self._MBSP_process(X_test)
        else:
            test_data = self._genia_process(X_test)

        # create temp file, open it as unicode file and train data on it
        with NamedTemporaryFile(delete=False, dir=self.tmpdir) as test_file:
            test_path = test_file.name
        with codecs.open(test_path, 'wb', 'utf-8') as test_file:
            for l in test_data:
                test_file.write('%s\n' %
                                ' '.join([e for e in l if len(e) > 0]))

        # run crf++ in a subprocess and collect stdout, stderr
        # raise error if strerr is not empty
        proc = subprocess.Popen(['crf_test', '-m',
                                self.model_path, test_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        msg_out, msg_err = proc.communicate()

        if msg_err:
            err_msg = ('subprocess exited with error \n"%s"' %
                       '\n'.join(['\t%s' % l for l in msg_err.split('\n')]))
            raise OSError(err_msg)

        # extract predictions from returned data, split them by
        # sentence and format them as required by test/evaluation script
        pred_raw = [(int(l[-1]) if len(l) > 0 else '')
                    for l in msg_out.split('\n')][:-1]
        predictions = []
        prev = 0
        for i in range(len(pred_raw)):
            if pred_raw[i] == '':
                predictions.append(round(np.average(pred_raw[prev:i])))
                prev = i + 1

        return predictions
