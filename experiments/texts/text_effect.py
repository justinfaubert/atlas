"""What reading a text in order adds, beyond the words it contains (run after concepts.py with neural instruments).

For each model and text: text effect = concept matrix of the text read in order minus the matrix of the same
words shuffled. The untrained model is the null: its text effect should agree across nothing. Also compares
every instrument with the count-based consensus (which has no prior), and lists the concept pairs that
reading in order consistently pulls together or apart.

    python experiments/texts/text_effect.py runs/texts_en
"""
import os
import sys

os.chdir(sys.argv[1])
import numpy as np,json,itertools
from scipy.stats import rankdata
meta=json.load(open('meta.json')); big=meta['big_texts']; kinds=meta['kinds']; P=meta['probes']; texts=meta['texts']
insts={'svd':'svd','ppmi':'ppmi','bert':'hf-bert-base-uncased','random':'hf-random-bert-base-uncased','roberta':'hf-roberta-base','gpt2':'hf-gpt2','mpnet':'hf-sentence-transformers_all-mpnet-base-v2'}
M={}
for k,f in insts.items():
    z=np.load(f+'_matrices.npz'); M[k]={kk:(S.astype(float),o) for kk,S,o in zip(z['keys'],z['S'],z['occ'])}
# common probes: present (>=5) in full, both halves and shuffled of every big text for every instrument
ok=np.ones(len(P),bool)
for k in insts:
    for t in big:
        for h in ['full','half0','half1','shuffled']:
            key=f'{t}::{h}'
            if key in M[k]: ok&=M[k][key][1]>=5
idx=np.where(ok)[0]; iu=np.triu_indices(len(idx),1); print(len(idx),'probes common to everything')
def U(S): return S[np.ix_(idx,idx)][iu]
def r(a,b): return np.corrcoef(rankdata(a),rankdata(b))[0,1]
def z(v): v=rankdata(v); return (v-v.mean())/v.std()
neural=['bert','random','roberta','gpt2','mpnet']
D={}
for k in neural:
    for t in big:
        sh=U(M[k][f'{t}::shuffled'][0])
        for h in ['full','half0','half1']:
            D[(k,t,h)]=U(M[k][f'{t}::{h}'][0])-sh
for k in ['svd','ppmi']:
    for t in big:
        for h in ['full','half0','half1']: D[(k,t,h)]=U(M[k][f'{t}::{h}'][0])
pairs=list(itertools.combinations(big,2))
B=[t for t in big if kinds[t]=='bible']; N=[t for t in big if kinds[t]!='bible']
out={}
for k in insts:
    rel=np.mean([r(D[(k,t,'half0')],D[(k,t,'half1')]) for t in big])
    cross=np.mean([r(D[(k,x,'full')],D[(k,y,'full')]) for x,y in pairs])
    crossN=np.mean([r(D[(k,x,'full')],D[(k,y,'full')]) for x,y in itertools.combinations(N,2)])
    hh=np.mean([r(D[(k,x,'half0')],D[(k,y,'half1')]) for x,y in pairs])
    cB=np.mean([z(D[(k,t,'full')]) for t in B],0); cN=np.mean([z(D[(k,t,'full')]) for t in N],0)
    rng=np.random.default_rng(0); hN=[]
    for _ in range(100):
        p=rng.permutation(N); hN.append(np.corrcoef(np.mean([z(D[(k,u,'full')]) for u in p[:5]],0),np.mean([z(D[(k,u,'full')]) for u in p[5:]],0))[0,1])
    hB=np.corrcoef(np.mean([z(D[(k,u,'full')]) for u in B[:5]],0),np.mean([z(D[(k,u,'full')]) for u in B[5:]],0))[0,1]
    out[k]=(cB,cN)
    print(f"{k:8s} reliability(half/half)={rel:.3f} cross-text={cross:.3f} (non-Bible {crossN:.3f}) half-to-half across={hh:.3f} | Bible-cons vs book-cons={np.corrcoef(cB,cN)[0,1]:.3f} ceilings books={np.mean(hN):.3f} bibles={hB:.3f}")
print('\nconsensus of text effect across instruments (all-text consensus):')
C={k:(out[k][0]+out[k][1])/2 for k in insts}
ks=list(insts)
print('        '+' '.join(f'{k:>7s}' for k in ks))
for a in ks: print(f'{a:8s}'+' '.join(f'{np.corrcoef(C[a],C[b])[0,1]:7.3f}' for b in ks))
# same-text agreement of text effect across instruments vs different-text
print('\nsame text vs different text, text-effect agreement across instruments:')
for a,b in [('bert','roberta'),('bert','gpt2'),('roberta','mpnet'),('bert','svd'),('mpnet','svd'),('gpt2','ppmi'),('random','svd'),('random','bert')]:
    same=np.mean([r(D[(a,t,'full')],D[(b,t,'full')]) for t in big]); diff=np.mean([r(D[(a,x,'full')],D[(b,y,'full')]) for x in big for y in big if x!=y])
    print(f'  {a}-{b}: same text {same:.3f}  different text {diff:.3f}')
np.save('text_effect_consensus.npy',np.stack([C[k] for k in ks])); json.dump({'ks':ks,'idx':idx.tolist()},open('text_effect_probes.json','w'))

meta=json.load(open('meta.json')); big=meta['big_texts']; P=meta['probes']; kinds=meta['kinds']
idx=np.array(json.load(open('text_effect_probes.json'))['idx']); iu=np.triu_indices(len(idx),1)
insts={'svd':'svd','ppmi':'ppmi','bert':'hf-bert-base-uncased','random':'hf-random-bert-base-uncased','roberta':'hf-roberta-base','gpt2':'hf-gpt2','mpnet':'hf-sentence-transformers_all-mpnet-base-v2'}
def z(v): v=rankdata(v); return (v-v.mean())/v.std()
def U(S): return S[np.ix_(idx,idx)][iu]
M={k:{kk:S.astype(float) for kk,S in zip(np.load(f+'_matrices.npz')['keys'],np.load(f+'_matrices.npz')['S'])} for k,f in insts.items()}
count=np.mean([z(U(M[k][f'{t}::full'])) for k in ['svd','ppmi'] for t in big],0)
B=[t for t in big if kinds[t]=='bible']; N=[t for t in big if kinds[t]!='bible']
countN=np.mean([z(U(M[k][f'{t}::full'])) for k in ['svd','ppmi'] for t in N],0)
print('count-based consensus (no prior) vs, for each model:')
for k in ['bert','random','roberta','gpt2','mpnet']:
    pr=np.load(insts[k]+'_prior.npy'); X=pr[idx]-pr[idx].mean(0); X/=np.linalg.norm(X,axis=1,keepdims=True); prior=(X@X.T)[iu]
    full=np.mean([z(U(M[k][f'{t}::full'])) for t in big],0); sh=np.mean([z(U(M[k][f'{t}::shuffled'])) for t in big],0)
    D=np.mean([z(U(M[k][f'{t}::full'])-U(M[k][f'{t}::shuffled'])) for t in big],0)
    c=lambda a,b: np.corrcoef(rankdata(a),rankdata(b))[0,1]
    print(f'  {k:8s} prior {c(count,prior):.3f}  shuffled-text {c(count,sh):.3f}  text-in-order {c(count,full):.3f}  text effect {c(count,D):.3f} | books-only count vs model in-order {c(countN,full):.3f}')
# which pairs does reading in order pull together/apart, consistently across trained models
Ds=[]
for k in ['bert','roberta','gpt2','mpnet']:
    Ds.append(np.mean([z(U(M[k][f'{t}::full'])-U(M[k][f'{t}::shuffled'])) for t in big],0))
Dm=np.mean(Ds,0); pairs=[(P[idx[i]],P[idx[j]]) for i,j in zip(*iu)]
o=np.argsort(-Dm); print('reading in order pulls together:',[pairs[i] for i in o[:15]]); print('pushes apart:',[pairs[i] for i in o[::-1][:12]])
