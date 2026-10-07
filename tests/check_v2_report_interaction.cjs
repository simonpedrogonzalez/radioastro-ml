// node tests/check_v2_report_interaction.cjs [report-directory] [source]
// Exercises report controls and fixed visibility ranges with a DOM/canvas double.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = process.argv[2] || 'collect/experiments/v2_visible_features_revised';
const sources = process.argv[3] ? [process.argv[3]] : ['0005+383','0012-399','0201-115'].filter(s=>fs.existsSync(path.join(root,`pilot_${s}.html`)));
assert(sources.length>0, 'No generated reports found');
let totalCases=0;
for(const sid of sources){
 const html=fs.readFileSync(path.join(root,`pilot_${sid}.html`),'utf8');
 const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
 let arcs=0,lines=0;
 const finite=(...xs)=>xs.forEach(x=>assert(Number.isFinite(x)));
 const ctx={fillText(){},strokeRect:finite,save(){},restore(){},translate:finite,rotate:finite,
   clearRect:finite,fillRect:finite,beginPath(){},fill(){},stroke(){},moveTo:finite,
   lineTo(...x){finite(...x);lines++;},setLineDash(){},arc(...x){finite(...x);arcs++;}};
 class Element{
  constructor(id){this.id=id;this.options=[];this.handlers={};this._value='';this._html='';this.textContent='';this.width=1100;this.height=id==='tracks'?850:460;
   this.tagName=['tracks','timeseries'].includes(id)?'CANVAS':/^(antenna|baseline)-(raw|log)$/.test(id)?'svg':'IMG';}
  add(o){this.options.push(o);if(this.options.length===1)this._value=o.value;}
  set innerHTML(v){this._html=v;if(this.id==='candidate-select'){assert.equal(v,'');this.options=[];this._value='';}}
  get innerHTML(){return this._html;}
  get value(){return this._value;} set value(v){this._value=String(v);}
  get selectedIndex(){return this.options.findIndex(o=>o.value===this.value);}
  addEventListener(name,fn){this.handlers[name]=fn;}
  getContext(type){assert.equal(type,'2d');return ctx;}
  getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height};}
  showModal(){this.open=true;}close(){this.open=false;}
  toDataURL(){return 'data:image/png;base64,test';}
 }
 const chartIds=['antenna-raw','antenna-log','baseline-raw','baseline-log'];
 const ids=['case-select','candidate-select','residual-panel','power-panel','baseline-panel','case-description','candidate-description','ranking','tracks','timeseries','hover-detail',...chartIds,'rank-truth','plot-zoom','zoom-close','zoom-content'];
 for(const id of ids)assert(html.includes(`id='${id}'`),`Missing DOM node ${id}`);
 const elements=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
 const sandbox=vm.createContext({document:{getElementById:id=>elements[id]},Option:class{constructor(text,value){this.text=text;this.value=String(value);}}});
 vm.runInContext(source,sandbox,{timeout:30000});
 const d=vm.runInContext('D',sandbox),cs=elements['case-select'],ca=elements['candidate-select'];
 const bounds=JSON.stringify(d.visibility_limits);
 for(const [key,fn] of [['amplitude',(r,i)=>Math.hypot(r,i)],['phase',(r,i)=>Math.atan2(i,r)*180/Math.PI]]){
  let lo=Infinity,hi=-Infinity;
  for(const c of d.cases)for(let i=0;i<c.real.length;i++){let x=fn(c.real[i],c.imag[i]);lo=Math.min(lo,x);hi=Math.max(hi,x);}
  assert(Math.abs(lo-d.visibility_limits[key].min)<1e-10);
  assert(Math.abs(hi-d.visibility_limits[key].max)<1e-10);
 }
 for(let i=0;i<cs.options.length;i++){
  cs.value=i;cs.handlers.change();
  assert(elements['case-description'].textContent.includes(cs.options[i].text));
  for(const id of ['residual-panel','power-panel','baseline-panel'])assert(elements[id].src.startsWith('data:image/png;base64,'));
  const c=d.cases[i],gain=['amplitude','phase','mixture'].includes(c.variant);
  for(const kind of ['antenna','baseline']){
   const rows=vm.runInContext(`rankedScores(D.cases[${i}], '${kind}')`,sandbox);
   assert.equal(rows.length,kind==='antenna'?d.antennas.length:d.valid_baselines.length);
   for(let k=1;k<rows.length;k++)assert(rows[k-1].raw>=rows[k].raw,'Bars must descend by raw score');
   const expected=kind==='antenna'?(gain?1:0):d.valid_baselines.filter(b=>c.injected_baseline?d.pairs[b].every((a,k)=>a===c.injected_baseline[k]):gain&&d.pairs[b].includes(d.antenna)).length;
   assert.equal(rows.filter(r=>r.affected).length,expected,'Wrong injected entities');
   for(const mode of ['raw','log']){
    const svg=elements[kind+'-'+mode].innerHTML;
    assert.equal((svg.match(/data-entity=/g)||[]).length,rows.length);
    assert.equal((svg.match(/data-affected="true"/g)||[]).length,expected);
    assert(svg.includes('rank 1 · raw score'));
   }
  }
  if(process.env.REPORT_PREVIEW_DIR&&c.variant==='amplitude'&&i===1){
   fs.mkdirSync(process.env.REPORT_PREVIEW_DIR,{recursive:true});
   for(const id of chartIds)fs.writeFileSync(path.join(process.env.REPORT_PREVIEW_DIR,id+'.svg'),'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 440">'+elements[id].innerHTML+'</svg>');
  }
  for(const id of ['residual-panel','antenna-log','tracks']){
   elements[id].handlers.click();assert(elements['plot-zoom'].open);
   assert(elements['zoom-content'].innerHTML.length>0);
   elements['zoom-close'].handlers.click();assert(!elements['plot-zoom'].open);
  }
  for(const prefix of ['a:','b:','all']){
   ca.value=ca.options.find(o=>o.value.startsWith(prefix)).value;ca.handlers.change();
   assert(elements['candidate-description'].textContent.includes(ca.options[ca.selectedIndex].text));
   assert.equal(JSON.stringify(d.visibility_limits),bounds,'Selection changed global limits');
   const points=vm.runInContext('points',sandbox);assert(points.length>0);
   for(let k=0;k<points.length;k+=2){
    const p=points[k],range=d.visibility_limits.amplitude;
    assert(Math.abs(p.y-(55+310-(p.amp-range.axis_min)/(range.axis_max-range.axis_min)*310))<1e-9);
    const phase=points[k+1],pr=d.visibility_limits.phase;
    assert(Math.abs(phase.y-(55+310-(phase.phase-pr.axis_min)/(pr.axis_max-pr.axis_min)*310))<1e-9);
   }
   const point=points[0];elements.timeseries.handlers.mousemove({target:elements.timeseries,clientX:point.x,clientY:point.y});
   assert(elements['hover-detail'].textContent.includes('ln(1 + baseline score B)'));
   assert(!elements['hover-detail'].textContent.includes('NaN'));
  }
  totalCases++;
 }
 assert(arcs>0&&lines>0);
 console.log(`${sid}: ${cs.options.length} cases; four ranked views and injection markers; plot enlargement; fixed visibility axes and selections passed`);
}
console.log(`Passed ${totalCases} cases (DOM/canvas double; not browser layout).`);
