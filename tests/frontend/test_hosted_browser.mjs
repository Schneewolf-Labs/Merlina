// Run with PLAYWRIGHT_MODULE=/path/to/playwright node tests/frontend/test_hosted_browser.mjs
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {createRequire} from 'node:module';
const {chromium}=createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || 'playwright');
const server=createServer(async(req,res)=>{
  const name=req.url==='/'?'index.html':req.url.replace('/hosted-static/','');
  if(!['index.html','app.mjs','client.mjs','style.css'].includes(name)){res.writeHead(404).end();return;}
  res.setHeader('Content-Type',name.endsWith('mjs')?'application/javascript':name.endsWith('css')?'text/css':'text/html');
  res.end(await readFile(new URL(`../../frontend/hosted/${name}`,import.meta.url)));
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : chromium.executablePath()),args:['--no-sandbox','--disable-dev-shm-usage']});
try {
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const project={id:'project-a',name:'Moonlight',goal:'Classify support tickets'};let created=false,job=null,approved=0,asked=false;
  await page.route('**/api/hosted/**',async route=>{
    const req=route.request(),path=new URL(req.url()).pathname.replace('/api/hosted',''),method=req.method();let data={};
    if(path==='/session')data=method==='POST'?{token:'pilot-session'}:{id:'alice',email:'alice@example.com'};
    else if(path==='/billing/balance')data={available_minor:2000,reserved_minor:0,currency:'usd'};
    else if(path==='/projects'){if(method==='POST'){created=true;data=project;}else data=created?[project]:[];}
    else if(path==='/projects/project-a')data=project;
    else if(path.endsWith('/datasets'))data=method==='POST'?{id:'dataset-a',name:'examples.jsonl'}:[{id:'dataset-a',name:'examples.jsonl'}];
    else if(path.endsWith('/advisor/messages')){asked=true;data={message:'Compare held-out accuracy.'};}
    else if(path.endsWith('/advisor'))data={turns:asked?[{user_message:'How should I evaluate?',status:'complete',response:{message:'Compare held-out accuracy. <img src=x onerror=alert(1)>'}}]:[],plans:[],evaluations:[]};
    else if(path.endsWith('/jobs')){if(method==='POST'){job={id:'job-a',status:'draft',config:req.postDataJSON().config};data=job;}else data=job?[job]:[];}
    else if(path.endsWith('/quote'))data={quote_id:'q-a',amount_minor:900,currency:'usd',expires_at:Math.floor(Date.now()/1000)+300,stages:[{name:'training',reserved_minor:900}]};
    else if(path.endsWith('/approve')){approved++;assert.equal(req.postDataJSON().quote_id,'q-a');job.status='running';data={job_id:'job-a',state:'approved'};}
    else if(path.endsWith('/cancel')){job.status='cancelled';data={state:'cancellation_requested'};}
    else if(path.endsWith('/events'))data=[{kind:'progress',payload:{step:2,total:10}}];
    else if(path.endsWith('/artifacts'))data=[];
    else if(path.endsWith('/jobs/job-a'))data=job;
    else throw new Error(`Unexpected request ${method} ${path}`);
    await route.fulfill({json:data});
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByLabel('Email',{exact:true}).fill('alice@example.com');await page.getByLabel('Password',{exact:true}).fill('password');await page.getByRole('button',{name:'Open my studio'}).click();
  await page.getByLabel('Project name',{exact:true}).fill('Moonlight');await page.getByLabel('What should your model accomplish?').fill('Classify support tickets');await page.getByRole('button',{name:'Create project',exact:true}).click();
  await page.getByLabel('Training examples').setInputFiles({name:'examples.jsonl',mimeType:'application/jsonl',buffer:Buffer.from('{"prompt":"Hello","chosen":"Hi"}')});await page.getByRole('button',{name:'Upload examples'}).click();
  await page.getByLabel('Base model',{exact:true}).fill('example/model');await page.getByRole('button',{name:'Prepare an experiment'}).click();
  await page.getByRole('button',{name:'Get current quote'}).click();await page.getByText('Spending limit: $9.00',{exact:true}).waitFor();
  assert.equal(await page.getByRole('button',{name:'Reserve credits & run'}).isDisabled(),true);assert.equal(approved,0);
  await page.getByLabel('I approve this experiment').check();await page.getByRole('button',{name:'Reserve credits & run'}).click();await page.getByText('Experiment job-a · running',{exact:true}).waitFor();assert.equal(approved,1);
  assert.equal(job.config.base_model,'example/model');assert.equal(job.config.dataset.source.source_type,'upload');
  await page.getByLabel('Ask Merlina',{exact:true}).fill('How should I evaluate?');await page.getByRole('button',{name:'Ask Merlina'}).click();await page.getByText('Compare held-out accuracy. <img src=x onerror=alert(1)>',{exact:true}).waitFor();assert.equal(await page.locator('#conversation img').count(),0);
  await page.getByRole('button',{name:'Cancel experiment'}).click();await page.getByText('Experiment job-a · cancelled',{exact:true}).waitFor();
  await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.getByRole('button',{name:'Sign out'}).click();await page.locator('#studio').waitFor({state:'hidden'});assert.equal(await page.locator('#project-panel').isVisible(),false);
  assert.deepEqual(errors,[]);console.log('Hosted browser flow passed: login → project → upload → draft → quote → explicit approval → progress → advisor → cancel; mobile and text rendering checked.');
} finally {await browser.close();await new Promise(resolve=>server.close(resolve));}
