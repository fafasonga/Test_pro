{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from os.path import join\n",
    "from glob import glob\n",
    "import re\n",
    "stemmer = nltk.snowball.EnglishStemmer()\n",
    "from collections import defaultdict\n",
    "import numpy as np\n",
    "import random\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "stop = [stemmer.stem(word) for word in stopwords.words('english')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "files = glob(join('./data/LinkedIn/parsed/', '*.txt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def preprocessing(content):\n",
    "    content = content.lower() #to lower case\n",
    "    content = re.sub('\\n', ' ', content) #replacing endlines with spaces\n",
    "    content = re.sub(r'http[^ ]*','LINK', content) #replacing links with LINK\n",
    "    content = re.sub('[^a-zA-Z0-9 ]', ' ', content) #placing everything with spaces except numbers and letters\n",
    "    content = re.sub('[12][0-9]{3}', 'YEAR', content) #replacing 4-numbers starting with 1 or 2 with YEAR\n",
    "    content = re.sub('[0-9]+', 'NUMBER', content) #replacing all other numbers with NUMBER\n",
    "    content = re.sub(' +', ' ', content) #replacing multiple spaces with one\n",
    "    return content"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "sum_files = []\n",
    "edu_files = []\n",
    "exp_files = []\n",
    "for txt in files:\n",
    "    with open(txt) as f:\n",
    "        content = f.read() + ' '\n",
    "    content = preprocessing(content)\n",
    "    if '_sum.txt' in txt:\n",
    "        sum_files.append(content)\n",
    "    if '_edu.txt' in txt:\n",
    "        edu_files.append(content)\n",
    "    if '_exp.txt' in txt:\n",
    "        exp_files.append(content)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$\\hat{s} = \n",
    "\\arg\\!\\max_s (p (s|x)) = \n",
    "\\arg\\!\\max_s \\left( \\cfrac{ \\prod_{i=1}^{N} p(s,x_i) }{\\prod_{i=1}^{N} p(x_i)} \\right) = \n",
    "\\arg\\!\\max_s \\left(  \\sum_{i=1}^{N} \\log{p(s,x_i)}  - \\sum_{i=1}^{N} \\log{p(x_i)} \\right) = $\n",
    "$=\n",
    "\\arg\\!\\max_s \\left( \\sum_{i=1}^{N} (\\log{p(x_i | s)} + \\log{p(s)}) - \\sum_{i=1}^{N} \\log{p(x_i)} \\right) = \\arg\\!\\max_s \\left(  \\sum_{i=1}^{N} \\log{p(x_i | s)} + N\\log{p(s)} \\right)$ "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def getprobs(datasets):\n",
    "    ps = []\n",
    "    cond_count = defaultdict(lambda: [0.1] * len(datasets))\n",
    "    sum_count = [0.0] * len(datasets)\n",
    "    total_samples = sum([len(dataset) for dataset in datasets]) \n",
    "    for i,dataset in enumerate(datasets):\n",
    "        ps.append(len(dataset) / float(total_samples))\n",
    "        for item in dataset:\n",
    "            for word in item.split():\n",
    "                w = stemmer.stem(word)\n",
    "                if w not in stop:\n",
    "                    cond_count[w][i] += 1\n",
    "                    sum_count[i] += 1.0\n",
    "    for word in cond_count:\n",
    "        for i in range(len(datasets)):\n",
    "            cond_count[word][i] /= (sum_count[i] + 0.1 * len(cond_count))\n",
    "    return ps, cond_count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#get train and test sets\n",
    "test_ratio = 0.2\n",
    "\n",
    "random.shuffle(sum_files)\n",
    "sum_files_train = sum_files[int(len(sum_files) * test_ratio):]\n",
    "sum_files_test = sum_files[:int(len(sum_files) * test_ratio)]\n",
    "\n",
    "random.shuffle(edu_files)\n",
    "edu_files_train = edu_files[int(len(edu_files) * test_ratio):]\n",
    "edu_files_test = edu_files[:int(len(edu_files) * test_ratio)]\n",
    "\n",
    "random.shuffle(exp_files)\n",
    "exp_files_train = exp_files[int(len(exp_files) * test_ratio):]\n",
    "exp_files_test = exp_files[:int(len(exp_files) * test_ratio)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "ps, pxs = getprobs([sum_files_train, edu_files_train, exp_files_train])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def predict(item, ps, pxs):\n",
    "    scores = []\n",
    "    for i in range(len(ps)):\n",
    "        score = 0\n",
    "        for word in item.split():\n",
    "            w = stemmer.stem(word)\n",
    "            if w not in stop:\n",
    "                cond_prob = pxs[stemmer.stem(w)][i]\n",
    "                if cond_prob:\n",
    "                    score += np.log(cond_prob)\n",
    "        score += len(ps) * np.log(ps[i])\n",
    "        scores.append(score)\n",
    "    return max(enumerate(scores), key=lambda x: x[1])[0] #argmax calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "predicts = []\n",
    "true = []\n",
    "for item in sum_files_test:\n",
    "    predicts.append(predict(item, ps, pxs))\n",
    "    true.append(0)\n",
    "for item in edu_files_test:\n",
    "    predicts.append(predict(item, ps, pxs))\n",
    "    true.append(1)\n",
    "for item in exp_files_test:\n",
    "    predicts.append(predict(item, ps, pxs))\n",
    "    true.append(2)\n",
    "predicts = np.asarray(predicts)\n",
    "true = np.asarray(true)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8958333333333334"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(np.where(predicts == true)[0]) / float(len(predicts))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Other:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "dataset_directory = './data/other/'\n",
    "files = glob(join(dataset_directory, '*.pdf'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "import PyPDF2\n",
    "other_res = []\n",
    "for the_file in files:\n",
    "    pdfFileObj = open(the_file, 'rb')\n",
    "    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)\n",
    "    file_content = \"\"\n",
    "    for page_ind in range(pdfReader.numPages):\n",
    "        pageObj = pdfReader.getPage(page_ind)\n",
    "        page_content = pageObj.extractText()\n",
    "        file_content += page_content\n",
    "    pred = predict(preprocessing(file_content), ps,pxs)\n",
    "    other_res.append(pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[2,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 1,\n",
       " 1,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 0,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 1,\n",
       " 1,\n",
       " 1,\n",
       " 1,\n",
       " 2,\n",
       " 0,\n",
       " 2,\n",
       " 2,\n",
       " 1,\n",
       " 1,\n",
       " 2,\n",
       " 1,\n",
       " 2,\n",
       " 1,\n",
       " 2,\n",
       " 2,\n",
       " 2,\n",
       " 2]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "other_res"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One more labelling algorithm:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "featscores = dict()\n",
    "for key in pxs:\n",
    "    feat_score = np.sort(pxs[key])[-1] /  np.sort(pxs[key])[-2]\n",
    "    featscores[key] = (feat_score, np.argsort(pxs[key])[-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "bachelor: (290.76859331839262, 1)\n",
      "bs: (151.75572001876264, 1)\n",
      "ph: (117.00250169385518, 1)\n",
      "june: (107.36340843561604, 2)\n",
      "electr: (105.41809558555269, 1)\n",
      "societi: (100.19580966279254, 1)\n",
      "may: (92.057016194285836, 2)\n",
      "mathemat: (70.664877260645213, 1)\n",
      "problem: (69.92158185990013, 0)\n",
      "ba: (59.080471152342717, 1)\n",
      "march: (54.884349322483963, 2)\n",
      "octob: (52.697721859436783, 2)\n",
      "communiti: (52.697721859436783, 2)\n",
      "love: (52.65699374634454, 0)\n",
      "best: (52.65699374634454, 0)\n",
      "alway: (52.65699374634454, 0)\n",
      "wisconsin: (47.496065044040215, 1)\n",
      "telecommun: (47.496065044040215, 1)\n",
      "psi: (47.496065044040215, 1)\n",
      "madison: (47.496065044040215, 1)\n",
      "linkedi: (47.496065044040215, 1)\n",
      "class: (47.496065044040215, 1)\n",
      "challeng: (44.024699689566745, 0)\n",
      "believ: (44.024699689566745, 0)\n",
      "weber: (35.911658935737734, 1)\n",
      "washington: (35.911658935737734, 1)\n",
      "univ: (35.911658935737734, 1)\n",
      "tong: (35.911658935737734, 1)\n",
      "sigma: (35.911658935737734, 1)\n",
      "shanghai: (35.911658935737734, 1)\n",
      "queen: (35.911658935737734, 1)\n",
      "peke: (35.911658935737734, 1)\n",
      "jiao: (35.911658935737734, 1)\n",
      "indian: (35.911658935737734, 1)\n",
      "forens: (35.911658935737734, 1)\n",
      "diploma: (35.911658935737734, 1)\n",
      "contest: (35.911658935737734, 1)\n",
      "columbia: (35.911658935737734, 1)\n",
      "berkeley: (35.911658935737734, 1)\n",
      "academi: (35.911658935737734, 1)\n",
      "thing: (35.39240563278895, 0)\n",
      "strong: (35.39240563278895, 0)\n",
      "step: (35.39240563278895, 0)\n",
      "cdn: (35.39240563278895, 0)\n",
      "background: (35.39240563278895, 0)\n",
      "devic: (35.204702155059429, 2)\n",
      "test: (33.018074692012256, 2)\n",
      "featur: (33.018074692012256, 2)\n",
      "specialti: (32.658793359427726, 0)\n",
      "code: (30.831447228965086, 2)\n",
      "tsinghua: (29.518267577005265, 1)\n",
      "integr: (28.644819765917919, 2)\n",
      "today: (26.760111576011159, 0)\n",
      "stop: (26.760111576011159, 0)\n",
      "someth: (26.760111576011159, 0)\n",
      "recent: (26.760111576011159, 0)\n",
      "partnership: (26.760111576011159, 0)\n",
      "look: (26.760111576011159, 0)\n",
      "hard: (26.760111576011159, 0)\n",
      "happi: (26.760111576011159, 0)\n",
      "hand: (26.760111576011159, 0)\n",
      "great: (26.760111576011159, 0)\n",
      "ggc: (26.760111576011159, 0)\n",
      "divers: (26.760111576011159, 0)\n",
      "assur: (26.760111576011159, 0)\n",
      "wrote: (26.458192302870749, 2)\n",
      "locat: (26.458192302870749, 2)\n",
      "conduct: (26.458192302870749, 2)\n",
      "honor: (25.36076510137072, 1)\n",
      "york: (24.327252827435235, 1)\n",
      "yat: (24.327252827435235, 1)\n",
      "xiang: (24.327252827435235, 1)\n",
      "winner: (24.327252827435235, 1)\n",
      "vega: (24.327252827435235, 1)\n",
      "universidad: (24.327252827435235, 1)\n",
      "undergradu: (24.327252827435235, 1)\n",
      "smith: (24.327252827435235, 1)\n",
      "shandong: (24.327252827435235, 1)\n",
      "sen: (24.327252827435235, 1)\n",
      "sc: (24.327252827435235, 1)\n",
      "riversid: (24.327252827435235, 1)\n",
      "religi: (24.327252827435235, 1)\n",
      "pune: (24.327252827435235, 1)\n",
      "princeton: (24.327252827435235, 1)\n",
      "primergi: (24.327252827435235, 1)\n",
      "philosophi: (24.327252827435235, 1)\n",
      "omega: (24.327252827435235, 1)\n",
      "nvq: (24.327252827435235, 1)\n",
      "nevada: (24.327252827435235, 1)\n",
      "ming: (24.327252827435235, 1)\n",
      "meritori: (24.327252827435235, 1)\n",
      "mari: (24.327252827435235, 1)\n",
      "liu: (24.327252827435235, 1)\n",
      "laud: (24.327252827435235, 1)\n",
      "las: (24.327252827435235, 1)\n",
      "induct: (24.327252827435235, 1)\n",
      "guild: (24.327252827435235, 1)\n",
      "fudan: (24.327252827435235, 1)\n",
      "eec: (24.327252827435235, 1)\n",
      "delhi: (24.327252827435235, 1)\n",
      "dean: (24.327252827435235, 1)\n",
      "cum: (24.327252827435235, 1)\n",
      "chao: (24.327252827435235, 1)\n",
      "bsc: (24.327252827435235, 1)\n",
      "brigham: (24.327252827435235, 1)\n",
      "beta: (24.327252827435235, 1)\n",
      "beij: (24.327252827435235, 1)\n",
      "video: (24.271564839823579, 2)\n",
      "ibm: (22.084937376776413, 2)\n",
      "dbnumber: (22.084937376776413, 2)\n",
      "get: (21.892158185990017, 0)\n",
      "administr: (21.203262625736176, 1)\n",
      "survey: (19.898309913729243, 2)\n",
      "next: (19.898309913729243, 2)\n",
      "app: (19.898309913729243, 2)\n",
      "affair: (19.898309913729243, 2)\n",
      "univers: (19.199728812284956, 1)\n",
      "profession: (18.303279794844112, 0)\n",
      "consum: (18.303279794844112, 0)\n",
      "accomplish: (18.303279794844112, 0)\n",
      "wpa: (18.127817519233368, 0)\n",
      "wifi: (18.127817519233368, 0)\n",
      "whenev: (18.127817519233368, 0)\n",
      "want: (18.127817519233368, 0)\n",
      "volum: (18.127817519233368, 0)\n",
      "troubleshoot: (18.127817519233368, 0)\n",
      "theori: (18.127817519233368, 0)\n",
      "tcp: (18.127817519233368, 0)\n",
      "supplic: (18.127817519233368, 0)\n",
      "strategist: (18.127817519233368, 0)\n",
      "sinc: (18.127817519233368, 0)\n",
      "ship: (18.127817519233368, 0)\n",
      "right: (18.127817519233368, 0)\n",
      "realiti: (18.127817519233368, 0)\n",
      "proven: (18.127817519233368, 0)\n",
      "possibl: (18.127817519233368, 0)\n",
      "molecular: (18.127817519233368, 0)\n",
      "methodolog: (18.127817519233368, 0)\n",
      "logic: (18.127817519233368, 0)\n",
      "life: (18.127817519233368, 0)\n",
      "liac: (18.127817519233368, 0)\n",
      "learner: (18.127817519233368, 0)\n",
      "kind: (18.127817519233368, 0)\n",
      "jasmin: (18.127817519233368, 0)\n",
      "glaze: (18.127817519233368, 0)\n",
      "formal: (18.127817519233368, 0)\n",
      "fast: (18.127817519233368, 0)\n",
      "failur: (18.127817519233368, 0)\n",
      "experienc: (18.127817519233368, 0)\n",
      "excit: (18.127817519233368, 0)\n",
      "enjoy: (18.127817519233368, 0)\n",
      "eda: (18.127817519233368, 0)\n",
      "domain: (18.127817519233368, 0)\n",
      "coupl: (18.127817519233368, 0)\n",
      "aw: (18.127817519233368, 0)\n",
      "aptitud: (18.127817519233368, 0)\n",
      "abl: (18.127817519233368, 0)\n",
      "yahoo: (17.711682450682073, 2)\n",
      "warehous: (17.711682450682073, 2)\n",
      "report: (17.711682450682073, 2)\n",
      "power: (17.711682450682073, 2)\n",
      "numberd: (17.711682450682073, 2)\n",
      "interfac: (17.711682450682073, 2)\n",
      "illustr: (17.711682450682073, 2)\n",
      "coordin: (17.711682450682073, 2)\n",
      "audit: (17.711682450682073, 2)\n",
      "colleg: (17.63968907519228, 1)\n",
      "zhejiang: (17.045760150101628, 1)\n",
      "scholarship: (17.045760150101628, 1)\n",
      "certifi: (17.045760150101628, 1)\n",
      "alpha: (17.045760150101628, 1)\n",
      "week: (15.525054987634903, 2)\n",
      "transform: (15.525054987634903, 2)\n",
      "releas: (15.525054987634903, 2)\n",
      "modul: (15.525054987634903, 2)\n",
      "end: (15.525054987634903, 2)\n",
      "avail: (15.525054987634903, 2)\n",
      "api: (15.525054987634903, 2)\n"
     ]
    }
   ],
   "source": [
    "list_feats = []\n",
    "for key, value in sorted(featscores.iteritems(), key=lambda (k,v): (v[0],k), reverse=True):\n",
    "    if value[0] > 15:\n",
    "        print \"%s: %s\" % (key, value)\n",
    "        list_feats.append(key)\n",
    "    else: \n",
    "        break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def extract_feats(original):\n",
    "    new_lines_num = original.count('\\n') #Number of lines\n",
    "    length = len(original) #Number of symbols\n",
    "    avg_words_in_line = np.mean([len(line.split()) for line in original.split('\\n')]) #Average number of words in line\n",
    "    num_capitals= len(re.findall('[A-Z]', original)) #Number of capital letters\n",
    "    \n",
    "    preproced = preprocessing(original)\n",
    "    \n",
    "    num_words = len(preproced.split()) #Number of words\n",
    "    cap_words_ratio = num_capitals / float(num_words) #percentage of words starting with capital letter\n",
    "    num_years = preproced.count('YEAR') \n",
    "    num_year_years = preproced.count('YEAR YEAR')\n",
    "    num_numbers = preproced.count('NUMBER')\n",
    "    num_links = preproced.count('LINK')\n",
    "    \n",
    "    feats = [new_lines_num, length, avg_words_in_line, num_capitals, num_words, cap_words_ratio, \n",
    "             num_years, num_year_years, num_numbers, num_links]\n",
    "    for word in list_feats:\n",
    "        feats.append(int(word in preproced))\n",
    "    return feats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "X = []\n",
    "Y = []\n",
    "for txt in files:\n",
    "    with open(txt) as f:\n",
    "        content = f.read() + ' '\n",
    "    X.append(extract_feats(content))\n",
    "    if '_sum.txt' in txt:\n",
    "        Y.append(0)\n",
    "    if '_edu.txt' in txt:\n",
    "        Y.append(1)\n",
    "    if '_exp.txt' in txt:\n",
    "        Y.append(2)\n",
    "X = np.asarray(X)\n",
    "Y = np.asarray(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.model_selection import cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.938987527721\n"
     ]
    }
   ],
   "source": [
    "gbc = GradientBoostingClassifier(n_estimators=30)\n",
    "print(np.mean(cross_val_score(gbc, X, Y, cv = 5, scoring='f1_macro')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:py27]",
   "language": "python",
   "name": "conda-env-py27-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
