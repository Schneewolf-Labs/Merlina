import {HostedClient, money, approvalAllowed, expiryTime, safeCheckout, projectPath} from './client.mjs';
const $ = id => document.getElementById(id);
const api = new HostedClient(sessionStorage.getItem('merlina-session') || '');
const busy = new Set();
let project = null, job = null, dataset = null, quote = null, generation = 0;
let savedDatasets=[];
function notice(message = '', error = false) { $('notice').textContent = message; $('notice').classList.toggle('error', error); }
function resetQuote() { quote = null; $('quote-panel').hidden = true; $('consent').checked = false; $('approve').disabled = true; }
function signedOut() { api.token = ''; sessionStorage.removeItem('merlina-session'); generation++; project = job = dataset = null; resetQuote(); $('project-panel').hidden = true; clearProjectView(); $('project-select').replaceChildren(new Option('Choose a project','')); $('balance').textContent='Balance unavailable'; for(const id of ['project-form','dataset-form','draft-form','advisor-form','evaluation-form'])$(id)?.reset(); $('studio').hidden = true; $('logout').hidden = true; $('login-panel').hidden = false; }
async function action(button, fn) {
  busy.add(button.id); button.disabled = true;
  try { await fn(); } catch (error) { if (error.status === 401) signedOut(); notice(error.message, true); }
  finally { busy.delete(button.id); button.disabled = false; updateControls(); }
}
function updateControls() {
  if($('inspect-dataset'))$('inspect-dataset').disabled = !dataset || busy.has('inspect-dataset');
  $('prepare').disabled = busy.has('prepare') || !dataset;
  $('quote').disabled = busy.has('quote') || !job || job.status !== 'draft';
  $('approve').disabled = busy.has('approve') || !approvalAllowed(quote, $('consent').checked);
  $('cancel').disabled = busy.has('cancel') || !job || ['draft','completed','failed','cancelled','stopped'].includes(job.status);
}
async function balance() { const b = await api.request('/billing/balance'); $('balance').textContent = `${money(b.available_minor,b.currency)} available · ${money(b.reserved_minor,b.currency)} reserved`; }
function options(select, records, placeholder, label) {
  select.replaceChildren(new Option(placeholder,''));
  for (const record of records) select.add(new Option(label(record),record.id));
}
async function projects() { const rows = await api.request('/projects'); options($('project-select'),rows,'Choose a project',p=>p.name); if (project) $('project-select').value=project.id; }
async function enter() { await api.request('/session'); $('login-panel').hidden=true; $('studio').hidden=false; $('logout').hidden=false; await projects(); await balance().catch(e=>notice(e.message,true)); }
async function selectProject(id) {
  const ownGeneration=++generation; project=job=dataset=null; clearProjectView(); resetQuote(); $('project-panel').hidden=true; updateControls();
  if (!id) return;
  const selected=await api.request(projectPath(id)); if(ownGeneration!==generation) return;
  project=selected; $('project-title').textContent=project.name; $('project-goal').textContent=project.goal; $('project-panel').hidden=false; $('dataset-status').textContent=''; $('dataset-inspection').hidden=true; $('configuration').textContent='Create an experiment to inspect its configuration.';
  $('job-status').textContent='No experiment selected.'; $('events').textContent='Your job will keep running when you close this page.'; $('artifacts').replaceChildren();
  await datasets(); await jobs(); document.dispatchEvent(new CustomEvent('merlina-project',{detail:project}));
}
async function jobs() { if(!project)return; const id=project.id; const rows=await api.request(`${projectPath(id)}/jobs`); if(project?.id!==id)return; options($('job-select'),rows,'Choose an experiment',j=>`${j.id.slice(0,8)} · ${j.status}`); if(job)$('job-select').value=job.id; }
async function refreshJob() {
  if(!project||!job)return; const pid=project.id,jid=job.id;
  const path=projectPath(pid,jid);
  const [current,events,artifacts]=await Promise.all([api.request(path),api.request(`${path}/events`),api.request(`${path}/artifacts`)]);
  if(project?.id!==pid||job?.id!==jid)return;
  job=current; $('configuration').textContent=JSON.stringify(job.config,null,2); $('job-status').textContent=`Experiment ${job.id.slice(0,8)} · ${job.status}`;
  $('events').textContent=events.length?events.map(e=>`${e.created_at || ''} ${e.kind || e.type || ''}\n${JSON.stringify(e.payload || e,null,2)}`).join('\n\n'):'Waiting for the next update…';
  $('artifacts').replaceChildren();
  if(!artifacts.length) { const p=document.createElement('p'); p.textContent='Completed artifacts will appear here.'; p.className='muted'; $('artifacts').append(p); }
  for(const artifact of artifacts) {
    const row=document.createElement('div'); row.className='artifact'; const button=document.createElement('button'); button.textContent=`Download ${artifact.name || artifact.id}`;
    button.addEventListener('click',()=>action(button,async()=>{
      const blob=await api.download(`${path}/artifacts/${encodeURIComponent(artifact.id)}/download`);
      const url=URL.createObjectURL(blob); const a=document.createElement('a');a.href=url;a.download=artifact.name || artifact.id;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
    })); row.append(button); $('artifacts').append(row);
  }
  updateControls(); document.dispatchEvent(new CustomEvent('merlina-job',{detail:job}));
}
$('login-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{const session=await api.request('/session','POST',{email:$('email').value,password:$('password').value});api.token=session.token;sessionStorage.setItem('merlina-session',session.token);$('password').value='';notice();await enter();});});
$('logout').addEventListener('click',()=>action($('logout'),async()=>{try{await api.request('/session','DELETE');}finally{signedOut();notice('You have signed out.');}}));
$('project-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{const p=await api.request('/projects','POST',{name:$('project-name').value,goal:$('goal').value});await selectProject(p.id);await projects();notice('A new wish is ready to take shape.');});});
$('project-select').addEventListener('change',()=>action($('project-select'),()=>selectProject($('project-select').value)));
$('credit-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{const amount=Number($('credits').value)*100;const result=await api.request('/billing/checkout','POST',{amount_minor:amount});window.location.assign(safeCheckout(result.checkout_url));});});
$('dataset-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{
  if(!project)throw new Error('Choose a project first.'); const pid=project.id;const file=$('dataset-file').files[0];if(!file)throw new Error('Choose a JSONL file.');if(file.size>10*1024*1024)throw new Error('Pilot datasets must be 10 MB or smaller.');
  const uploaded=await api.request(`${projectPath(pid)}/datasets`,'POST',{name:file.name,content:await file.text()});if(project?.id!==pid)return;dataset=uploaded;resetQuote();await datasets();$('dataset-status').textContent=`${uploaded.name || file.name} is ready.`;
});});
$('draft-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{if(!project||!dataset)return;const pid=project.id;const draft=await api.request(`${projectPath(pid)}/jobs`,'POST',{config:{base_model:$('model').value,training_mode:'sft',num_epochs:Number($('epochs').value),dataset:{source:{source_type:'upload',dataset_id:dataset.id}}}});if(project?.id!==pid)return;job=draft;resetQuote();await jobs();await refreshJob();notice('Experiment prepared. Review the configuration, then request a quote.');});});
$('job-select').addEventListener('change',()=>action($('job-select'),async()=>{resetQuote();job=$('job-select').value?{id:$('job-select').value}:null;await refreshJob();}));
$('quote').addEventListener('click',()=>action($('quote'),async()=>{const pid=project.id,jid=job.id;const next=await api.request(`${projectPath(pid,jid)}/quote`,'POST',{});if(project?.id!==pid||job?.id!==jid)return;quote=next;$('consent').checked=false;$('quote-total').textContent=`Spending limit: ${money(quote.amount_minor,quote.currency)}`;$('quote-expiry').textContent=`Valid until ${new Date(expiryTime(quote.expires_at)).toLocaleString()}. Unused reserved credits are released after settlement.`;$('quote-stages').textContent=JSON.stringify(quote.stages,null,2);$('quote-panel').hidden=false;}));
$('consent').addEventListener('change',updateControls);
$('approve').addEventListener('click',()=>action($('approve'),async()=>{if(!approvalAllowed(quote,$('consent').checked))throw new Error('Request a fresh quote and approve its budget.');const path=projectPath(project.id,job.id);const quoteId=quote.quote_id;resetQuote();await api.request(`${path}/approve`,'POST',{quote_id:quoteId});notice('Credits reserved. Your experiment is ready for execution.');await refreshJob();await balance();}));
$('cancel').addEventListener('click',()=>action($('cancel'),async()=>{await api.request(`${projectPath(project.id,job.id)}/cancel`,'POST',{});notice('Cancellation requested. Resources may continue billing until shutdown is confirmed.');await refreshJob();}));
let polling=false;
setInterval(async()=>{updateControls();if(polling||document.hidden||!api.token||!job)return;polling=true;try{await Promise.all([refreshJob(),balance()]);}catch(e){if(e.status===401)signedOut();notice(`Progress refresh paused: ${e.message}`,true);}finally{polling=false;}},5000);
if(api.token)enter().catch(e=>{signedOut();notice(e.message,true);});

async function advisorHistory() {
  if(!project)return; const pid=project.id;
  const history=await api.request(`${projectPath(pid)}/advisor`);if(project?.id!==pid)return;
  $('conversation').replaceChildren(); $('advisor-plans').replaceChildren();
  for(const turn of history.turns) {
    for(const [speaker,message] of [['You',turn.user_message],['Merlina',turn.response?.message || (turn.status==='pending'?'This request is pending. If it was interrupted, send a new message.':'The advisor could not complete this request.')]]) {
      const block=document.createElement('div');block.className='turn';const label=document.createElement('strong');label.textContent=speaker;const text=document.createElement('p');text.textContent=message;block.append(label,text);$('conversation').append(block);
    }
  }
  for(const proposed of history.plans) {
    const block=document.createElement('div');block.className='proposed-plan';const title=document.createElement('h3');title.textContent=proposed.plan.summary;const pre=document.createElement('pre');pre.textContent=JSON.stringify(proposed.plan,null,2);block.append(title,pre);
    if(proposed.status==='proposed') {
      const button=document.createElement('button');button.textContent=proposed.plan.config?'Accept plan & create draft':'Accept evaluation plan';
      button.addEventListener('click',()=>action(button,async()=>{const result=await api.request(`${projectPath(pid)}/advisor/plans/${encodeURIComponent(proposed.id)}/approve`,'POST',{});if(project?.id!==pid)return;if(result.job_id){job={id:result.job_id};resetQuote();await jobs();await refreshJob();}await advisorHistory();notice('Plan accepted. Paid execution still requires your approval of a current quote.');}));block.append(button);
    } else { const p=document.createElement('p');p.textContent='Plan accepted';block.append(p); }
    $('advisor-plans').append(block);
  }
}
document.addEventListener('merlina-project',()=>{advisorHistory().catch(e=>notice(e.message,true));});
$('advisor-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{
  if(!project)return;const pid=project.id;
  const body={message:$('advisor-message').value,request_key:crypto.randomUUID()};if(job)body.job_id=job.id;if(dataset)body.dataset_id=dataset.id;
  await api.request(`${projectPath(pid)}/advisor/messages`,'POST',body);if(project?.id!==pid)return;$('advisor-message').value='';await advisorHistory();
});});
$('evaluation-form').addEventListener('submit',e=>{e.preventDefault();action(e.submitter,async()=>{if(!job||!project)throw new Error('Select an experiment to record its evaluation.');await api.request(`${projectPath(project.id)}/advisor/evaluations`,'POST',{job_id:job.id,criteria:$('evaluation-criteria').value,observations:$('evaluation-observations').value});notice('Evaluation saved as your reported observations. Ask Merlina to review them alongside the recorded results.');});});

async function datasets() {
  if(!project)return;const pid=project.id;const rows=await api.request(`${projectPath(pid)}/datasets`);if(project?.id!==pid)return;savedDatasets=rows;options($('dataset-select'),rows,'Choose a dataset',d=>d.name);if(dataset)$('dataset-select').value=dataset.id;
}
$('dataset-select').addEventListener('change',()=>{dataset=savedDatasets.find(d=>d.id===$('dataset-select').value)||null;$('dataset-inspection').hidden=true;resetQuote();updateControls();});
$('inspect-dataset').addEventListener('click',()=>action($('inspect-dataset'),async()=>{if(!project||!dataset)return;const pid=project.id,did=dataset.id;const report=await api.request(`${projectPath(pid)}/datasets/${encodeURIComponent(did)}/inspect`);if(project?.id!==pid||dataset?.id!==did)return;$('dataset-inspection').textContent=JSON.stringify(report,null,2);$('dataset-inspection').hidden=false;}));

function clearProjectView() {
  for(const id of ['conversation','advisor-plans','artifacts'])$(id)?.replaceChildren();
  if($('job-select'))$('job-select').replaceChildren(new Option('Choose an experiment',''));
  if($('dataset-select'))$('dataset-select').replaceChildren(new Option('Choose a dataset',''));
  savedDatasets=[];
}
