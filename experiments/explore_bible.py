import sys, time, numpy as np
from atlas.loaders import load_bibles
from atlas.embed import embed_corpus, EmbeddingCache
from atlas.bundle import Bundle
from atlas.nulls import null_distribution, triangle_wilsons
from atlas.spectra import spectrum, decay_fit
D=sys.argv[1]
langs="English French German Spanish Latin Greek Russian Finnish Hungarian Turkish Arabic Hebrew Hindi Indonesian Vietnamese Chinese Japanese Korean".split()
t=time.time()
c=load_bibles(D, langs); print('verses',len(c), time.time()-t)
ch=c.aggregate('chapter'); print('chapters',len(ch))
emb=embed_corpus(ch,'lsa:200',EmbeddingCache('/tmp/claude-0/-home-user/73b3f126-75c9-574b-88e3-4acb0ff33ff2/scratchpad/cache')); print('embedded', time.time()-t)
b=Bundle.from_embeddings(emb,k=32,unit_ids=ch.unit_ids)
cons=b.consensus(); print('shared',round(cons.shared_fraction,3),'iters',cons.iterations)
sh=null_distribution(b, lambda bb: np.array([bb.consensus().shared_fraction]), 'shuffled', reps=5)
print('shared shuffled', sh.ravel().round(3))
print('view residual', {k:round(v,3) for k,v in sorted(cons.view_residual.items(), key=lambda x:x[1])})
tw=np.array([h.wilson for h in b.all_triangles()])
fl=null_distribution(b, triangle_wilsons(b.views), 'flat', reps=10)
sf=null_distribution(b, triangle_wilsons(b.views), 'shuffled', reps=3)
print('tri wilson real mean %.4f min %.4f max %.4f'%(tw.mean(),tw.min(),tw.max()))
print('flat null mean %.4f min %.4f'%(fl.mean(),fl.min()), 'shuffled mean %.4f'%sf.mean())
z=(fl.mean(0)-tw)/fl.std(0); print('frac triangles beyond flat null (z>3):', (z>3).mean())
loop=['Chinese','Japanese','French','English']
h=b.holonomy(loop); print('Hutch loop wilson %.4f maxdeg %.1f meandeg %.2f twice %.4f'%(h.wilson,h.max_angle_deg,h.mean_angle_deg,h.power(2).wilson))
for L in (['English','French','German'],['English','Chinese','Japanese'],['French','Spanish','Latin'],['Chinese','Japanese','Korean'],['English','Arabic','Hebrew']):
    print(L, round(b.holonomy(L).wilson,4), round(b.holonomy(L).max_angle_deg,1))
lam=spectrum(cons.M); f=decay_fit(lam); print('consensus top3 %.3f logR2 %.3f powR2 %.3f'%(f.top3,f.log_r2,f.power_r2))
for L in ['English','Chinese']:
    f=decay_fit(spectrum(b.fibres[L])); print(L,'top3 %.3f'%f.top3)
