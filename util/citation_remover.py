import re
from string import punctuation

PUNC = set(punctuation)

class CitationRemover(object):
    def __init__(self):
        # see method _remove_citations to detail description of what
        # there RegExes do.
        self.dashes_rep = re.compile(u'\u2013\s?').sub
        self.brackets_cit = re.compile(r'\[(\d+(-|,|,\s)?)+\]').sub
        self.multi_pound = re.compile(r'#[^a-zA-Z]+#').sub
        self.extra_comma_space = re.compile(r'\s(,\s)').sub
        self.extra_stop_space = re.compile(r'\s(\.\s)').sub
        self.multi_space = re.compile(r'\s\s+').sub
        self.et_al_cit = re.compile(r'[a-zA-Z]+(\sand\s[a-zA-Z]+|\set\sal\.)?'
                                    r',?\s\d{4}(;\s|\sand\s)?').sub
        self.unecessary_space = re.compile(r'\s').sub

    def remove_citations(self, s):
        # convert long dashes (and \emph) to minus
        # signs (easy to deal with b/c ASCII)
        s = self.dashes_rep('-', s)

        # replaces citations in square brackets
        # (e.g. [1]) with <CITATION> for (not too)
        # easy removal
        s = self.brackets_cit('<CITATION>', s)

        # replaces citations like "Author1 et al., 2000",
        # "Author1 and Author2, 2000" and "Author1, 2000"
        # with <CITATION>.
        # Note: this method is not 100% proof (e.g. it ignores
        # authors with spaces or dashes in names), but, given
        # the fact that such citations are often in parenthesis,
        # that is enough :)
        s = self.et_al_cit('<CITATION>', s)

        # create list out of input text, initialize
        # counters and boolean variables
        s = [ch for ch in s]
        in_cit = False
        last_par = -1
        i = 0

        while i < len(s):
            # look for opening parenthesis; if found
            # set last_par to its location
            if s[i] == '(':
                last_par = i

            # look for closing parenthesis; if found,
            # look for '#' between i and open parenthesis;
            # if any '#' is found, replace all characters
            # between last '#' and i (i included) with '#'
            if s[i] == ')':
                last_pound = -1
                for j in range(i - 1, last_pound - 1, -1):
                    if s[j] == '#':
                        last_pound = j
                        break
                if last_pound > -1:
                    for j in range(i, last_pound, -1):
                        s[j] = '#'
                last_par = -1

            # set in_cit to True if a opening angled
            # parenthesis is found.
            # if that is the case and s[i] is inside a
            # parenthesis, replace every charcter from
            # i (excluded, see next if statement)
            # back to last_par (included) with '#'
            if s[i] == '<':
                in_cit = True
                if last_par > -1:
                    for j in range(i-1, last_par-1, -1):
                        popped = s.pop(j)
                        s.insert(j, '#')

            # if in_cit replace character with '#'.
            # if the replaced char is '>', exit from
            # in_cit
            if in_cit:
                popped = s.pop(i)
                s.insert(i, '#')
                if popped == '>':
                    in_cit = False

            i += 1
        # join string together, eliminate '#' and
        # sequences of '#' with non-letter chars in between
        s = ''.join(s)
        s = self.multi_pound('', s)

        # remove space before commas/full stops
        s = self.extra_comma_space(', ', s)
        s = self.extra_stop_space('. ', s)

        # condense multiple sequential spaces into one,
        # strips spaces at the end of sentence s
        s = self.multi_space(' ', s)

        # replace spaces other than ' ' with ' '.
        s = self.unecessary_space(' ', s).strip()

        # add full stop at end if not present
        if s[-1] not in PUNC:
            s = s + '.'

        return s
