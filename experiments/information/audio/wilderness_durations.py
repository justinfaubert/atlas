"""Compare CMU Wilderness read-NT durations (LangList.txt, aligned hours per recording) with our NT bits and characters.

Usage: python wilderness_durations.py <scratch dir holding wild/LangList.txt and runs/info_rate/nt.json>
Recordings with under 10 aligned hours are treated as failed alignments. Translations differ from the corpus ones."""
import json, numpy as np, collections, sys
from scipy.stats import spearmanr, pearsonr
S=sys.argv[1]
iso={'Afrikaans':'afr','Albanian':'als','Amharic':'amh','Arabic':'arb','Bulgarian':'bul','Burmese':'mya','Cebuano':'ceb','Chinese':'cmn','Croatian':'hrv','Czech':'ces','Danish':'dan','Dutch':'nld','English':'eng','Esperanto':'epo','Farsi':'pes','Finnish':'fin','French':'fra','German':'deu','Greek':'ell','Hebrew':'heb','Hindi':'hin','Hungarian':'hun','Icelandic':'isl','Indonesian':'ind','Italian':'ita','Japanese':'jpn','Kannada':'kan','Korean':'kor','Latin':'lat','Lithuanian':'lit','Malagasy':'plt','Malayalam':'mal','Maori':'mri','Marathi':'mar','Nepali':'npi','Norwegian':'nob','Paite':'pck','Polish':'pol','Portuguese':'por','Romanian':'ron','Russian':'rus','Serbian':'srp','Shona':'sna','Slovak':'slk','Slovene':'slv','Somali':'som','Spanish':'spa','Swedish':'swe','Tagalog':'tgl','Telugu':'tel','Thai':'tha','Turkish':'tur','Vietnamese':'vie','Xhosa':'xho','Zarma':'dje','Zulu-NT':'zul','Swahili-NT':'swh','Ewe-NT':'ewe','Kabyle-NT':'kab','Wolof-NT':'wol','Wolaytta-NT':'wal','Latvian-NT':'lav','Gujarati-NT':'guj','Basque-NT':'eus','Ukranian-NT':'ukr','Cherokee-NT':'chr','Aukan-NT':'djk','Cabecar-NT':'cjp','Cakchiquel-NT':'cak','Galela-NT':'gbi','Jakalteko-NT':'jac','Lukpa-NT':'dop','Mam-NT':'mam','Shuar-NT':'jiv','Tachelhit-NT':'shi','Uma-NT':'ppk','Uspanteco-NT':'usp','Amuzgo-NT':'amu','Coptic-NT':'cop','Syriac-NT':'syc','Romani-NT':'rmy','Quichua-NT':'qvi','Nahuatl-NT':'nhe','Chinantec-NT':'chz'}
dur=collections.defaultdict(list); names={}
for line in open(S+'/wild/LangList.txt'):
    f=line.split()
    if len(f)<14: continue
    h,m,s=map(int,f[10].split(':')); dur[f[1]].append((f[0],h*3600+m*60+s, int(f[8])))
nt=json.load(open(S+'/runs/info_rate/nt.json')); nv=nt['verses']
R={r['name']:r for r in nt['results']}
rows=[]
for n in nt['languages']:
    c=iso.get(n)
    if c and c in dur:
        ds=[d for d in dur[c]]
        rows.append((n,c,np.mean([d[1] for d in ds])/3600, len(ds), R[n]['text_orig']['chars']-(R[n]['text_orig']['words']-nv), R[n]['text_orig']['ngram_bits_by_order'][-1], [round(d[1]/3600,1) for d in ds]))
print(len(rows),'matched of',len(nt['languages']))
for r in rows: print(r[0],r[1],round(r[2],1),r[3],r[6])
D=np.log([r[2] for r in rows]); C=np.log([r[4] for r in rows]); B=np.log([r[5] for r in rows])
sd=lambda x: round(100*np.std(x,ddof=1),1)
print('spread %: hours',sd(D),'chars',sd(C),'bits',sd(B))
print('pearson log hours vs log chars',round(pearsonr(D,C)[0],3),' vs log bits',round(pearsonr(D,B)[0],3))
print('spearman', round(spearmanr(D,C)[0],3), round(spearmanr(D,B)[0],3))
# same-language different recordings floor
pairs=[np.log(max(x[1] for x in v)/min(x[1] for x in v))*100 for v in dur.values() if len(v)>=2]
print('within-language recording spread (median abs log diff %)', round(np.median(pairs),1), len(pairs))
allh=np.log([d[1] for v in dur.values() for d in v]); print('all 699 recordings spread', sd(allh), 'median hours', round(np.exp(np.median(allh))/3600,1))
print('--- filtered: recordings with >= 10 h aligned (drops failed alignments)')
rows2=[]
for r in rows:
    ok=[h for h in r[6] if h>=10]
    if ok: rows2.append((r[0],np.mean(ok),r[4],r[5]))
D=np.log([r[1] for r in rows2]); C=np.log([r[2] for r in rows2]); B=np.log([r[3] for r in rows2])
print(len(rows2),'spread %: hours',sd(D),'chars',sd(C),'bits',sd(B))
print('pearson log hours vs log chars',round(pearsonr(D,C)[0],3),' vs log bits',round(pearsonr(D,B)[0],3))
# partial: hours vs bits controlling chars, and vs chars controlling bits
def part(y,x,z):
    X=np.column_stack([np.ones(len(z)),z]); ry=y-X@np.linalg.lstsq(X,y,rcond=None)[0]; rx=x-X@np.linalg.lstsq(X,x,rcond=None)[0]; return round(float(np.corrcoef(ry,rx)[0,1]),3)
print('partial hours~bits | chars', part(D,B,C), ' hours~chars | bits', part(D,C,B))
b=np.polyfit(C,D,1)[0]; print('slope log hours on log chars', round(b,2), ' on log bits', round(np.polyfit(B,D,1)[0],2))
print('rate spreads %: chars per second', sd(C-D), ' bits per second', sd(B-D))
bps=np.exp(B-D)/3600; cps=np.exp(C-D)/3600
print('median bits/s', round(float(np.median(bps)),1), ' median chars/s', round(float(np.median(cps)),1))
for r,b_,c_ in sorted(zip(rows2,bps,cps), key=lambda t:t[1]): print(r[0], round(b_,1), round(c_,1))
json.dump([[r[0],round(r[1],2),r[2],r[3]] for r in rows2], open(S+'/wild/audio_rows.json','w'))
