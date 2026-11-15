"""Retrieve and log bibliographic database searches; no novelty claims."""
import json
import re
from pathlib import Path
from datetime import datetime,timezone
import requests

ROOT=Path(__file__).resolve().parents[1]/'literature_search'
QUERIES=[
 'music AND (recommendation OR recommender) AND (emotion OR mood) AND (uncertainty OR confidence OR abstention OR probabilistic)',
 'music AND (recommendation OR recommender) AND (abstention OR "selective prediction")',
 'music AND emotion AND ("probability distribution" OR "uncertainty estimation")']

def main():
    ROOT.mkdir(exist_ok=True); logs=[]; records=[]
    for db in ('openalex','semantic_scholar'):
        for i,q in enumerate(QUERIES):
            token=None; fetched=0
            while True:
                if db=='openalex':
                    url='https://api.openalex.org/works'; params={'search':q,'filter':'to_publication_date:2026-11-14','per-page':200,'cursor':token or '*'}
                else:
                    url='https://api.semanticscholar.org/graph/v1/paper/search/bulk'
                    # S2 Boolean operators: + AND, | OR, whitespace AND.
                    query=q.replace(' AND ',' + ').replace(' OR ',' | ')
                    params={'query':query,'fields':'title,abstract,year,externalIds,url','publicationDateOrYear':':2026-11-14'}
                    if token:params['token']=token
                entry=dict(database=db,query_index=i+1,url=url,params=params,timestamp_utc=datetime.now(timezone.utc).isoformat())
                try:
                    r=requests.get(url,params=params,timeout=40);entry['http_status']=r.status_code
                    name=f'{db}_q{i+1}_{fetched}.json'
                    (ROOT/name).write_text(r.text,encoding='utf-8');entry['raw_file']=name
                    r.raise_for_status(); data=r.json()
                    items=data.get('results',[]) if db=='openalex' else data.get('data',[])
                    entry['total']=data.get('meta',{}).get('count') if db=='openalex' else data.get('total')
                    for item in items:
                        if db=='openalex':
                            inv=item.get('abstract_inverted_index') or {};words={p:w for w,positions in inv.items() for p in positions}; abstract=' '.join(words[p] for p in sorted(words))
                            records.append(dict(database=db,query=i+1,title=item.get('title'),doi=item.get('doi'),year=item.get('publication_year'),url=item.get('id'),abstract=abstract))
                        else:records.append(dict(database=db,query=i+1,title=item.get('title'),doi=(item.get('externalIds') or {}).get('DOI'),year=item.get('year'),url=item.get('url'),abstract=item.get('abstract') or ''))
                    fetched+=len(items);token=data.get('meta',{}).get('next_cursor') if db=='openalex' else data.get('token')
                    entry['retrieved']=len(items);entry['cap_reached']=fetched>=2000 and bool(token)
                    logs.append(entry)
                    if not token or not items or fetched>=2000:break
                except Exception as exc:
                    entry['error']=str(exc);logs.append(entry);break
            print(db,i+1,fetched,flush=True)
    unique={}
    for r in records:
        key=(r['doi'] or '').lower().replace('https://doi.org/','') or re.sub(r'\W+','',str(r['title']).lower())
        if key in unique:continue
        text=(str(r['title'])+' '+r['abstract']).lower()
        music='music' in text
        uncertain=any(x in text for x in ('uncertain','confidence','absten','probabilis','probability','selective prediction','bayesian'))
        affect=any(x in text for x in ('emotion','mood','affect','recommend'))
        r['screening']='retain for full-text review' if music and uncertain and affect else ('uncertain: abstract absent' if not r['abstract'] else 'exclude: topic criteria not met')
        unique[key]=r
    (ROOT/'search_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    (ROOT/'screening.json').write_text(json.dumps(list(unique.values()),indent=2),encoding='utf-8')
    print('Unique',len(unique),'retained',sum(r['screening'].startswith('retain') for r in unique.values()))

if __name__=='__main__':main()
