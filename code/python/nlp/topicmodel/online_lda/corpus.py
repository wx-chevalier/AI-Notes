import os
from itertools import izip
import re
# read and organize data

#3 2:3 4:5 5:3 --- document info (word: count)
class document:
    ''' the class for a single document '''
    def __init__(self):
        self.words = []
        self.counts = []
        self.length = 0
        self.total = 0

class corpus:
    ''' the class for the whole corpus'''
    def __init__(self):
        self.size_vocab = 0
        self.docs = []
        self.num_docs = 0

    def read_data(self, filename):
        if not os.path.exists(filename):
            print 'no data file, please check it'
            return
        print 'reading data from %s.' % filename

        for line in file(filename): 
            ss = line.strip().split()
            if len(ss) == 0: continue
            doc = document()
            doc.length = int(ss[0])

            doc.words = [0 for w in range(doc.length)]
            doc.counts = [0 for w in range(doc.length)]
            for w, pair in enumerate(re.finditer(r"(\d+):(\d+)", line)):
                doc.words[w] = int(pair.group(1))
                doc.counts[w] = int(pair.group(2))

            doc.total = sum(doc.counts) 
            self.docs.append(doc)

            if doc.length > 0:
                max_word = max(doc.words)
                if max_word >= self.size_vocab:
                    self.size_vocab = max_word + 1

            if (len(self.docs) >= 10000):
                break
        self.num_docs = len(self.docs)
        print "finished reading %d docs." % self.num_docs

# def read_data(filename):
#     c = corpus()
#     c.read_data(filename)
#     return c

def read_stream_data(f, num_docs):
  c = corpus()
  splitexp = re.compile(r'[ :]')
  for i in range(num_docs):
    line = f.readline()
    line = line.strip()
    if len(line) == 0:
      break
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    c.docs.append(d)

  c.num_docs = len(c.docs)
  return c

# This version is about 33% faster
def read_data(filename):
    c = corpus()
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        d = document()
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        d.words = wordids
        d.counts = wordcts
        d.total = sum(d.counts)
        d.length = len(d.words)
        c.docs.append(d)

        if d.length > 0:
            max_word = max(d.words)
            if max_word >= c.size_vocab:
                c.size_vocab = max_word + 1

    c.num_docs = len(c.docs)
    return c

def count_tokens(filename):
    num_tokens = 0
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        splitline = [int(i) for i in splitexp.split(line)]
        wordcts = splitline[2::2]
        num_tokens += sum(wordcts)

    return num_tokens

splitexp = re.compile(r'[ :]')
def parse_line(line):
    line = line.strip()
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    return d
