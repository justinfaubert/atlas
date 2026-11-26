"""Does the "culture of origin" signal in the native pair model come from shared authors?

Refits the pair model from analyse_native.py (pairs_*.csv) with two extra features: same author but a different
book in the same language, and the same across languages. Bootstrap over works.

    python experiments/native/author_check.py runs/native_analysis/pairs_svd.csv ...
"""
import csv,sys,numpy as np
A={'vingtmille':'verne','tourdumonde':'verne','robur':'verne','ilemyst':'verne','mousquetaires':'dumas','tulipe':'dumas',
'werther':'goethe','wahlverw':'goethe','meister':'goethe','karenina':'tolstoy','warpeace':'tolstoy','copperfield':'dickens','oliver':'dickens'}
au=lambda n:A.get(n.split('--')[0],n.split('--')[0])
w=lambda n:n.split('--')[0]
rng=np.random.default_rng(0)
for f in sys.argv[1:]:
    P=list(csv.DictReader(open(f)))
    for p in P: a=int(au(p['x'])==au(p['y']) and p['same_work']=='0'); p['same_author']=a*int(p['same_lang']); p['author_xlang']=a*(1-int(p['same_lang']))
    feats=['same_work','same_lang','same_origin','both_native','same_author','author_xlang','rel']
    def fit(R):
        X=np.array([[1]+[float(r[k]) for k in feats] for r in R]);y=np.array([float(r['rsa']) for r in R])
        return np.linalg.lstsq(X,y,rcond=None)[0][1:]
    b=fit(P);works=sorted({w(p['x']) for p in P}|{w(p['y']) for p in P});B=[]
    for _ in range(300):
        s=rng.choice(works,len(works));c={k:(s==k).sum() for k in works}
        R=[p for p in P for _ in range(c[w(p['x'])]*c[w(p['y'])] if w(p['x'])!=w(p['y']) else c[w(p['x'])])]
        B.append(fit(R))
    lo,hi=np.percentile(B,[2.5,97.5],axis=0)
    # origin effect without any same-author pairs
    Q=[p for p in P if not (p['same_author'] or p['author_xlang'])];feats2=feats[:-3]+['rel'];
    X=np.array([[1]+[float(r[k]) for k in feats2] for r in Q]);y=np.array([float(r['rsa']) for r in Q])
    b2=np.linalg.lstsq(X,y,rcond=None)[0][1:]
    print(f.split('pairs_')[1], ' '.join(f'{k} {b[i]:+.3f}[{lo[i]:+.3f},{hi[i]:+.3f}]' for i,k in enumerate(feats[:6])), '| no-author origin %+.3f'%b2[2], 'n_author_pairs',sum(p['same_author'] for p in P))
