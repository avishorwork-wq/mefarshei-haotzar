const ALIAS={"gen":"Genesis","bereishit":"Genesis","bereshit":"Genesis","ex":"Exodus","shemot":"Exodus","lev":"Leviticus","vayikra":"Leviticus","num":"Numbers","bamidbar":"Numbers","deut":"Deuteronomy","devarim":"Deuteronomy","ps":"Psalms","tehillim":"Psalms","isa":"Isaiah","yeshayahu":"Isaiah","jer":"Jeremiah","ber":"Berakhot","berachot":"Berakhot","brachot":"Berakhot","shab":"Shabbat","pes":"Pesachim","bk":"Bava Kamma","bm":"Bava Metzia","bb":"Bava Batra","sanh":"Sanhedrin","kid":"Kiddushin","ket":"Ketubot","yev":"Yevamot","git":"Gittin","chul":"Chullin","chullin":"Chullin","avoda zara":"Avodah Zarah","az":"Avodah Zarah","rosh hashana":"Rosh Hashanah","song of songs":"Song of Songs","shir hashirim":"Song of Songs","kohelet":"Ecclesiastes","mishlei":"Proverbs","iyov":"Job","esther":"Esther"};
const COMM=["Rashi","Tosafot","Ramban","Rashba","Ritva","Meiri","Rosh","Ibn Ezra","Rashbam","Sforno","Radak","Maharsha","Onkelos","Chizkuni","Or HaChaim","Kli Yakar"];
function parseQ(q){let t=' '+q.toLowerCase().replace(/[,]/g,' ')+' ';const out={book:null,sec:null,verse:null,comm:null,topic:''};
 for(const c of COMM){const r=new RegExp('\\b'+c.toLowerCase()+'\\b');if(r.test(t)){out.comm=c;t=t.replace(r,' ');break}}
 const names=[...TANAKH,...TALMUD].map(b=>[b.toLowerCase(),b]).concat(Object.entries(ALIAS)).sort((a,b)=>b[0].length-a[0].length);
 for(const [k,v] of names){const r=new RegExp('(^|\\s)'+k.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'(?=\\s|$)');if(r.test(t)){const t0=t;t=t.replace(r,' ');const hasNum=/\b\d{1,3}[ab]?\b/.test(t);const scoped=new RegExp('\\b(in|on|from|within)\\s*$').test(t0.slice(0,t0.search(r)+1).trim()+' ')||new RegExp('(^|\\s)'+k.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\s*$').test(t0)&&/\b(in|on|from|within)\s+\S+\s*$/.test(t0);if(hasNum||scoped||!t.trim()){out.book=v}else{t=t0}break}}
 if(out.book){const tal=TALMUD.includes(out.book);let m;if(tal&&(m=t.match(/\b(\d{1,3}[ab])\b/))){out.sec=m[1];t=t.replace(m[0],' ')}
  else if(!tal&&(m=t.match(/\b(\d{1,3})\s*[: .]\s*(\d{1,3})\b/))){out.sec=m[1];out.verse=m[2];t=t.replace(m[0],' ')}
  else if(!tal&&(m=t.match(/\b(\d{1,3})\b/))){out.sec=m[1];t=t.replace(m[0],' ')}}
 out.topic=t.replace(/\b(on|in|of|the|at|chapter|perek|daf|verse|pasuk)\b/g,' ').replace(/\s+/g,' ').trim();return out}
async function goRef(book,sec,verse){S.kind=TALMUD.includes(book)?'talmud':'tanakh';S.book=book;S.sec=sec;$('#kind').value=S.kind;await fillPickers();if(![...$('#sec').options].some(o=>o.value===S.sec)){S.sec=$('#sec').options[0].value;$('#sec').value=S.sec}document.body.classList.remove('sr');$('#results').hidden=true;await render();if(verse){const el=document.querySelector(`.v[data-i="${verse}"]`);if(el){el.scrollIntoView({behavior:'smooth',block:'center'});openPane(verse)}}}
const heRe=/[\u0590-\u05FF]/;
async function exactSearch(q,filters){const body={query:q,type:'text',size:400,source_proj:true,field:heRe.test(q)?'naive_lemmatizer':'naive_lemmatizer',slop:0,sort_method:'score',aggs:[],filters:filters||[],filter_fields:filters||[]};
 const r=await fetch(API+'search-wrapper',{method:'POST',headers:{'Content-Type':'text/plain'},body:JSON.stringify({query:q,type:'text',size:400,source_proj:true,field:heRe.test(q)?'naive_lemmatizer':'exact'})});const j=await r.json();return(j.hits?.hits||[]).map(h=>({ref:(h._source?.ref)||h._id.replace(/ \(.*$/,''),he:h._source?.heRef||'',path:h._source?.path||'',hl:(h.highlight?.naive_lemmatizer||h.highlight?.exact||[]).join(' … ')}))}
const AI_ON=()=>{const t=document.getElementById('aiToggle');return !t||t.checked};let TX=null,SH={},HEW=null,SCOPE='all';
async function getTX(){if(!TX){const m=await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');m.env.allowLocalModels=false;TX=await m.pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2')}return TX}
let RR=null;
async function getRR(){if(!RR){const m=await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');m.env.allowLocalModels=false;const id='Xenova/ms-marco-MiniLM-L-6-v2';RR={tok:await m.AutoTokenizer.from_pretrained(id),model:await m.AutoModelForSequenceClassification.from_pretrained(id)}}return RR}
async function rerank(q,res){const r=await getRR();const tx=await Promise.all(res.map(async x=>{try{const t=await fetchText(x[1]);return strip(flat(t.en).join(' ')).slice(0,1200)}catch(e){return ''}}));
 const inp=r.tok(res.map(()=>q),{text_pair:tx,padding:true,truncation:true,max_length:256});const out=await r.model(inp);const sc=Array.from(out.logits.data);
 return res.map((x,i)=>[x,sc[i],tx[i]]).filter(z=>z[2]).sort((a,b)=>b[1]-a[1]).map(z=>z[0])}
async function enVec(q){const p=await getTX();const o=await p(q,{pooling:'mean',normalize:true});return o.data}
async function shard(k){if(SH[k])return SH[k];const [cfg,buf]=await Promise.all([fetch(`sem_${k}.json`).then(r=>r.json()),fetch(`sem_${k}.bin`).then(r=>r.arrayBuffer())]);return SH[k]={refs:cfg.refs,scale:cfg.scale,v:new Int8Array(buf)}}
async function meaning(vec,scope,n=12){const kinds=scope==='all'?['tanakh','talmud']:[scope];let best=[];for(const k of kinds){const sh=await shard(k);const N=sh.refs.length;for(let i=0;i<N;i++){let d=0;const o=i*384;for(let j=0;j<384;j++)d+=sh.v[o+j]*vec[j];d/=sh.scale;best.push([d,sh.refs[i]]);if(best.length>400){best.sort((a,b)=>b[0]-a[0]);best.length=60}}}best.sort((a,b)=>b[0]-a[0]);return best.slice(0,n)}
function erf(x){const s=x<0?-1:1;x=Math.abs(x);const t=1/(1+0.3275911*x);return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x))}
async function heVec(q){
  if(!HEW){const [cfg,e,d]=await Promise.all([fetch('he_nn.json').then(r=>r.json()),fetch('he_nn_emb.bin').then(r=>r.arrayBuffer()),fetch('he_nn_dense.bin').then(r=>r.arrayBuffer())]);
    const f=new Float32Array(d),H=cfg.H;let p=0;const l1w=f.subarray(p,p+=H*H),l1b=f.subarray(p,p+=H),ow=f.subarray(p,p+=384*H),ob=f.subarray(p,p+=384);HEW={cfg,E:new Int8Array(e),l1w,l1b,ow,ob}}
  const {D,H,scale}=HEW.cfg;const enc=new TextEncoder();const t=q.replace(/[\u0591-\u05C7]/g,'').replace(/[^\u05D0-\u05EA ]/g,'');
  const ids=[];for(const w0 of t.split(/\s+/).filter(Boolean)){const w=' '+w0+' ';for(const n of [3,4])for(let i=0;i+n<=w.length;i++)ids.push(crc32(enc.encode(w.slice(i,i+n)))%D)}
  if(!ids.length)ids.push(0);const bag=new Float32Array(H);for(const id of ids){const o=id*H;for(let d=0;d<H;d++)bag[d]+=HEW.E[o+d]}
  const k=1/(scale*Math.sqrt(ids.length));for(let d=0;d<H;d++)bag[d]*=k;let mu=0;for(let d=0;d<H;d++)mu+=bag[d];mu/=H;let va=0;for(let d=0;d<H;d++)va+=(bag[d]-mu)**2;va/=H;const is=1/Math.sqrt(va+1e-5);
  const ln=new Float32Array(H);for(let d=0;d<H;d++)ln[d]=(bag[d]-mu)*is;const h=new Float32Array(H);for(let i=0;i<H;i++){let a=HEW.l1b[i];const o=i*H;for(let j=0;j<H;j++)a+=HEW.l1w[o+j]*ln[j];h[i]=bag[i]+a*0.5*(1+erf(a/Math.SQRT2))}
  const out=new Float32Array(384);let m=0;for(let i=0;i<384;i++){let a=HEW.ob[i];const o=i*H;for(let j=0;j<H;j++)a+=HEW.ow[o+j]*h[j];out[i]=a;m+=a*a}m=Math.sqrt(m);for(let i=0;i<384;i++)out[i]/=m;return out}
const crcT=(()=>{const t=[];for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=c&1?0xEDB88320^(c>>>1):c>>>1;t[n]=c>>>0}return t})();function crc32(b){let c=0xFFFFFFFF;for(const x of b)c=crcT[(c^x)&255]^(c>>>8);return(c^0xFFFFFFFF)>>>0}
async function card(ref,extra){let t={he:[],en:[]};try{t=await fetchText(ref)}catch(e){}const he=strip(flat(t.he).join(' ')),en=strip(flat(t.en).join(' '));return `<div class="card"><span class="ref">${esc(ref)}</span><span class="hr">${esc(extra||'')}</span>${he?`<div class="he">${esc(he)}</div>`:''}${en?`<div class="en">${esc(en)}</div>`:''}<div class="btns"><button data-open="${esc(ref)}">Open with mefarshim ›</button></div></div>`}

const NIQ=/[\u0591-\u05C7]/g;
function hlSafe(h){return esc(String(h||'').replace(/<b>/g,'\u0001').replace(/<\/b>/g,'\u0002')).replace(/\u0001/g,'<mark>').replace(/\u0002/g,'</mark>')}
function coreRank(ref,path){const b=(ref||'').replace(/ \d.*$/,'');const top=(path||'').split('/')[0];
 if(TANAKH.includes(b))return -1;if(TALMUD.includes(b))return 0;
 const m=(ref||'').match(/^(.*?) on (.*)$/);if(m&&COMM.includes(m[1].replace(/^Tosafot.*/,'Tosafot').replace(/^Rashi.*/,'Rashi')))return 1;
 if(/^(Rashi|Tosafot|Ramban|Rashba|Ritva|Meiri|Rosh|Ibn Ezra|Rashbam|Sforno|Radak|Chizkuni|Or HaChaim|Kli Yakar|Maharsha)\b/.test(ref||''))return 1;
 if(/^(Mishnah|Mishneh Torah|Shulchan Arukh|Tosefta|Midrash|Sifra|Sifrei|Mekhilta|Bereishit Rabbah|Shemot Rabbah|Zohar)/.test(ref||''))return 2;
 return 3}
async function kwFetch(q,field,slop){const body={query:q,type:'text',size:120,source_proj:true,field};if(slop)body.slop=slop;let r;for(let a=0;a<3;a++){r=await fetch(API+'search-wrapper',{method:'POST',headers:{'Content-Type':'text/plain'},body:JSON.stringify(body)});if(r.ok)break;await new Promise(z=>setTimeout(z,600*(a+1)))}if(!r.ok)throw new Error('search '+r.status);const j=await r.json();const tot=j.hits?.total;return{total:typeof tot==='object'?tot.value:tot,hits:(j.hits?.hits||[]).map((h,i)=>({ref:(h._source?.ref)||String(h._id).replace(/ \(.*$/,''),he:h._source?.heRef||'',path:h._source?.path||'',ver:h._source?.version||'',lang:h._source?.lang||'',hl:(h.highlight?.[field]||h.highlight?.exact||h.highlight?.naive_lemmatizer||[])[0]||'',i}))}}
let KW={q:'',list:[],shown:0,total:0,phrase:0};
function kwRender(){const L=$('#xl');const part=KW.list.slice(0,KW.shown);
 L.innerHTML=part.map(h=>`<div class="card kw"><span class="ref">${esc(h.ref)}</span><span class="hr">${esc(h.he)}${h.phrase?' · exact phrase':''}</span><div class="${heRe.test(h.hl)?'he':'en'} snip">${hlSafe(h.hl)}</div><button class="srcbtn" data-open="${esc(h.ref)}">Open</button></div>`).join('')+(KW.shown<KW.list.length?'<button class="srcbtn" id="kwMore" style="margin:8px 0">Show more</button>':'');
 const m=$('#kwMore');if(m)m.onclick=()=>{KW.shown+=25;kwRender()};
 $('#st').textContent=KW.list.length?`${KW.total.toLocaleString()} matches in Sefaria${KW.phrase?` (${KW.phrase} exact-phrase)`:''}. Showing ${Math.min(KW.shown,KW.list.length)} of the top ${KW.list.length}, core texts first.`:'No matches. Try fewer words, or turn on AI search.'}
async function keywordSearch(q0){const he=heRe.test(q0);const q=he?q0.replace(/[\u05BE\u05F3\u05F4]/g,' ').replace(NIQ,'').replace(/\s+/g,' ').trim():q0.trim();const field=he?'naive_lemmatizer':'exact';
 $('#st').textContent='Searching…';$('#xh').hidden=true;$('#rl').innerHTML='';$('#xl').innerHTML='';
 const words=q.split(/\s+/).filter(Boolean);
 const jobs=[kwFetch(q,field,0)];if(words.length>1)jobs.push(kwFetch(q,field,10));
 const res=await Promise.all(jobs.map(p=>p.catch(e=>({hits:[],total:0,err:e}))));if(res.every(r=>r.err)){$('#st').textContent='Search unavailable: '+res[0].err.message;return}
 const seen=new Map();let phraseN=0;res[0].hits.forEach(h=>{h.phrase=words.length>1;if(!seen.has(h.ref)||(!seen.get(h.ref).hl&&h.hl))seen.set(h.ref,h)});
 if(res[1])res[1].hits.forEach(h=>{if(!seen.has(h.ref))seen.set(h.ref,Object.assign(h,{phrase:false}))});
 let list=[...seen.values()];list.forEach(h=>{h.cr=coreRank(h.ref,h.path)});
 list.sort((a,b)=>(b.phrase-a.phrase)||(a.cr-b.cr)||(a.i-b.i));
 if(SCOPE==='tanakh')list=list.filter(h=>TANAKH.some(b=>h.ref.startsWith(b+' ')));else if(SCOPE==='talmud')list=list.filter(h=>TALMUD.some(b=>h.ref.startsWith(b+' ')));
 KW={q,list,shown:25,total:Math.max(res[0].total||0,res[1]?.total||0),phrase:list.filter(h=>h.phrase).length};$('#xh').hidden=false;$('#xh').textContent='Word matches';kwRender()}
async function runSearch(q){const p=parseQ(q);if(p.book&&p.sec&&!p.topic&&!p.comm){return goRef(p.book,p.sec,p.verse)}
 document.body.classList.add('sr');const R=$('#results');R.hidden=false;
 R.innerHTML=`<div class="hd"><h3>Results for “${esc(q)}”</h3><button id="rc" class="srcbtn">Close</button></div><div class="seg" id="scope" style="width:max-content;margin:8px 0"><button data-s="all" class="${SCOPE==='all'?'on':''}">All</button><button data-s="tanakh" class="${SCOPE==='tanakh'?'on':''}">Tanakh</button><button data-s="talmud" class="${SCOPE==='talmud'?'on':''}">Gemara</button></div><p class="mut" id="st">Searching by meaning…</p><div id="rl"></div><h4 id="xh" hidden>Exact-word matches (Sefaria)</h4><div id="xl"></div>`;
 $('#rc').onclick=()=>{document.body.classList.remove('sr');R.hidden=true};$('#scope').onclick=e=>{const b=e.target.closest('button');if(b){SCOPE=b.dataset.s;runSearch(q)}};
 if(!AI_ON()){$('#rl').innerHTML='';return keywordSearch(p.topic||q)}
 const topic=p.topic||q;const he=heRe.test(topic);
 try{const vec=he?await heVec(topic):await enVec(topic);let res=await meaning(vec,SCOPE,he?30:60);let reranked=false;
  if(p.book)res=res.filter(r=>r[1].startsWith(p.book));if(p.comm)res=res.filter(r=>r[1].includes(p.comm));if(!he&&res.length>1){$('#st').textContent='Reranking…';try{res=await rerank(topic,res.slice(0,50));reranked=true}catch(e){console.warn('rerank unavailable',e)}}res=res.slice(0,12);
  $('#st').textContent=he?'Closest in meaning (Hebrew query, experimental)':reranked?'Closest in meaning (reranked)':'Closest in meaning';
  $('#rl').innerHTML=(await Promise.all(res.map(r=>card(r[1],reranked?'':Math.round(r[0]*100)+'% match')))).join('')||'<p class="mut">No meaning matches.</p>'}catch(e){$('#st').textContent='Meaning search unavailable: '+e.message}
 let hits=[];try{hits=await exactSearch(topic)}catch(e){}const seen=new Set();hits=hits.filter(h=>!seen.has(h.ref)&&seen.add(h.ref));
 if(p.book)hits=hits.filter(h=>(h.ref+' '+h.path).includes(p.book));if(p.comm)hits=hits.filter(h=>(h.ref+' '+h.path).includes('/'+p.comm+'/')||h.ref.startsWith(p.comm));
 if(hits.length){$('#xh').hidden=false;$('#xl').innerHTML=(await Promise.all(hits.slice(0,10).map(h=>card(h.ref,h.he)))).join('')}}
document.addEventListener('click',e=>{const b=e.target.closest('[data-open]');if(!b)return;const m=b.dataset.open.match(/^(.*?) (\d+[ab]?)(?::(\d+))?/);if(m){let bk=m[1].replace(/^.* on /,'');goRef(bk,m[2],m[3])}});
$('#searchForm').addEventListener('submit',e=>{e.preventDefault();const q=$('#q').value.trim();if(q)runSearch(q)});
(()=>{const t=document.getElementById('aiToggle');if(!t)return;try{if(localStorage.getItem('mh_ai')==='0')t.checked=false}catch(e){}
 t.addEventListener('change',()=>{try{localStorage.setItem('mh_ai',t.checked?'1':'0')}catch(e){}const q=$('#q').value.trim();if(q&&document.body.classList.contains('sr'))runSearch(q)})})();

(async()=>{const t=document.getElementById('aiToggle');if(!t)return;try{const r=await fetch('sem_tanakh.json',{method:'HEAD'});if(!r.ok)throw 0}catch(e){t.checked=false;t.disabled=true;const l=t.closest('.ai');l.title='AI search data is not on this host yet. Exact word search is on.';l.style.opacity='.55';l.querySelector('b').textContent='AI soon'}})();
