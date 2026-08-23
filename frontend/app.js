const $ = selector => document.querySelector(selector);
const state = {config:null,calibration:null,mode:"cooperative",ship:"pommern",limitType:"rounds",running:false,paused:false,runId:null,startedAt:0,completed:0,current:0};
const labels={cooperative:"联合作战",asymmetric:"非对称作战"};
const statusLabels={idle:"待机",starting:"启动中",launching_game:"启动游戏",entering_game:"进入游戏",preparing:"准备战斗",battle:"战斗中",paused:"已暂停",collecting_rewards:"结算统计",requeueing:"进入下一局",returning:"返回港口",completed:"计划完成",stopped:"已停止",failed:"运行异常"};
const movementLabels={idle:"等待任务",autopilot_route:"游戏自动航行",route_planning:"规划航线",opening:"驶向中央点",route_transit:"按航线推进",search:"驶向中央点",approach:"点内接敌",hold_range:"维持副炮距离",hold_capture:"点内留守",reverse_range:"倒船拉距",separate:"主动拉开",avoid_island:"绕岛修正",disengage:"脱离",evade:"规避鱼雷",manual_pause:"用户接管暂停"};
const routeLabels={unplanned:"未规划",departure:"离开出生点",transit:"驶向中央点",final_approach:"进入点位",station:"点内作战"};

async function api(path,options={}){const response=await fetch(path,{headers:{"Content-Type":"application/json"},...options});const body=await response.json();if(!response.ok)throw new Error(body.error||"请求失败");return body;}
function number(value){return new Intl.NumberFormat("zh-CN").format(Number(value||0));}
function duration(seconds){const s=Math.max(0,Math.floor(Number(seconds)||0));return[Math.floor(s/3600),Math.floor(s%3600/60),s%60].map(v=>String(v).padStart(2,"0")).join(":");}
function toast(message,error=false){const el=$("#toast");el.textContent=message;el.className=`toast show${error?" error":""}`;clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.className="toast",2600);}

function renderConfig(){
  $("#modeGrid").innerHTML=Object.entries(state.config.modes).map(([key,name])=>`<button class="choice ${key===state.mode?"selected":""}" data-mode="${key}"><strong>${labels[key]||name}</strong><small>${key==="cooperative"?"稳定循环":"高收益挑战"}</small></button>`).join("");
  const ships=Object.entries(state.config.ships);$("#shipCount").textContent=`${ships.length} 艘`;
  $("#shipList").innerHTML=ships.map(([key,ship])=>`<button class="ship-choice ${key===state.ship?"selected":""}" data-ship="${key}"><span><b>${ship.name}</b><small>${ship.nation.toUpperCase()} · ${ship.type}</small></span><em>${ship.secondary_range} km 副炮</em></button>`).join("");
  $("#modeGrid").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.mode=button.dataset.mode;renderConfig();});
  $("#shipList").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.ship=button.dataset.ship;renderConfig();});
}
function renderCalibration(){const c=state.calibration||{valid:false,reason:"启动时自动检查"};const card=$("#calibrationCard");card.classList.toggle("ready",Boolean(c.valid));$("#calibrationTitle").textContent=c.valid?"系统就绪":"启动时自动检查";$("#calibrationMessage").textContent=c.valid?c.reason:"画面、港口和输入均由系统确认";$("#startBtn").disabled=false;}
function updateLimit(){const continuous=state.limitType==="continuous";const value=Math.max(1,Number($("#limitValue").value||1));$("#numberControl").classList.toggle("disabled",continuous);$("#limitValue").disabled=continuous;$("#minusBtn").disabled=continuous;$("#plusBtn").disabled=continuous;$("#limitUnit").textContent=state.limitType==="rounds"?"局":state.limitType==="duration"?"分钟":"";$("#limitHint").textContent=continuous?"持续完整对局，直到手动安全停止":state.limitType==="rounds"?`完整结束 ${value} 局后返回港口并停止`:`达到 ${value} 分钟后完成当前局，再返回港口停止`;document.querySelectorAll(".limit-tab").forEach(button=>button.classList.toggle("active",button.dataset.limit===state.limitType));}
async function startRun(){try{const value=state.limitType==="continuous"?0:Number($("#limitValue").value);const result=await api("/api/run/start",{method:"POST",body:JSON.stringify({ship:state.ship,mode:state.mode,limit_type:state.limitType,limit_value:value})});state.runId=result.run_id;state.startedAt=Date.now()/1000;toast("自动作战已启动");await pollStatus();await loadDashboard();}catch(error){toast(error.message,true);}}
async function stopRun(){try{await api("/api/run/stop",{method:"POST",body:"{}"});toast("将在安全释放控制后停止");}catch(error){toast(error.message,true);}}
async function pauseAutomation(){try{const result=await api("/api/run/pause",{method:"POST",body:"{}"});if(!result.ok)throw new Error("当前没有运行中的任务");state.paused=true;toast("已暂停下发新指令，舰船保持原状态");await pollStatus();}catch(error){toast(error.message,true);}}
async function resumeAutomation(){try{const result=await api("/api/run/resume",{method:"POST",body:"{}"});if(!result.ok)throw new Error("当前没有运行中的任务");state.paused=false;toast("正在识别当前状态并继续原操作");await pollStatus();}catch(error){toast(error.message,true);}}
async function primaryAction(){if(!state.running)return startRun();return state.paused?resumeAutomation():pauseAutomation();}
function renderPrimaryAction(){const button=$("#startBtn"),label=button.querySelector("span"),icon=button.querySelector("b");label.textContent=!state.running?"开始自动作战":state.paused?"继续自动作战":"暂停自动作战";icon.textContent=!state.running?"→":state.paused?"▶":"Ⅱ";button.classList.toggle("paused",state.paused);}

async function pollStatus(){
  try{
    const data=await api("/api/status");state.running=data.running;state.paused=Boolean(data.paused_by_user||data.manual_intervention_latched);state.runId=data.run_id||state.runId;state.startedAt=data.started_at||state.startedAt;state.completed=Number(data.completed_rounds||0);state.current=Number(data.current_round||0);renderPrimaryAction();
    $("#systemStatus").textContent=statusLabels[data.state]||data.state||"待机";$("#systemChip").classList.toggle("live",data.running);$("#liveBadge").textContent=data.running?"LIVE":"OFFLINE";$("#liveBadge").classList.toggle("live",data.running);
    if(data.calibration)state.calibration=data.calibration;renderCalibration();$("#stopBtn").disabled=!data.running;$("#progressMessage").textContent=data.message||"等待任务启动";$("#runId").textContent=data.run_id?`#${data.run_id}`:"—";
    const ship=state.config?.ships[data.ship]?.name||data.ship||"—";$("#activeSetup").textContent=`${ship} · ${labels[data.mode]||data.mode||"—"}`;
    const elapsed=data.elapsed_seconds!=null?Number(data.elapsed_seconds):(state.startedAt?Date.now()/1000-state.startedAt:0);$("#elapsedTime").textContent=duration(elapsed);
    const max=Number(data.max_rounds||0),durationLimit=Number(data.duration_minutes||0)*60;const progress=max?Math.min(100,state.completed/max*100):durationLimit?Math.min(100,elapsed/durationLimit*100):0;
    $("#roundDisplay").textContent=max?`${state.completed} / ${max} 局`:`${state.completed} 局完成`;$("#progressPercent").textContent=`${Math.round(progress)}%`;$("#progressBar").style.width=`${progress}%`;
    $("#movementMode").textContent=movementLabels[data.movement_mode]||data.movement_mode||"等待任务";$("#movementReason").textContent=data.movement_reason||"配置预设后即可开始";
    const routeProgress=Math.max(0,Math.min(100,Number(data.route_progress||0)*100));$("#routePhase").textContent=routeLabels[data.route_phase]||data.route_phase||"未规划";$("#routeProgress").textContent=`${Math.round(routeProgress)}%`;$("#routeLine").style.width=`${routeProgress}%`;$("#routeVisual").style.setProperty("--route-x",`${22+routeProgress*.48}%`);$("#routeVisual").style.setProperty("--route-y",`${68-routeProgress*.39}%`);
    $("#captureDistance").textContent=data.capture_point_distance_km==null?"—":`${Number(data.capture_point_distance_km).toFixed(1)} km`;
    const source=data.distance_source==="ocr"?"OCR":data.distance_source==="minimap_grid"?"小地图":"";$("#targetDistance").textContent=data.target_distance_km==null?"—":`${Number(data.target_distance_km).toFixed(1)} km${source?` · ${source}`:""}`;
    const provider=data.ocr_provider==="CUDAExecutionProvider"?"GPU":data.ocr_provider==="CPUExecutionProvider"?"CPU":data.ocr_provider||"—";$("#visionStatus").textContent=`${provider} · ${data.frame_status||"—"}`;$("#feedbackStatus").textContent=data.movement_verified?"位移已确认":data.safety_state==="tripped"?"安全熔断":"等待反馈";$("#roundState").textContent=data.stop_after_current?"本局结束后停止":data.state==="battle"?`第 ${state.current} 局进行中`:statusLabels[data.state]||"未开始";
    const intervention=data.state==="failed";$("#interventionCard").hidden=!intervention;$("#interventionMessage").textContent=intervention?`${data.message||"自动流程失败"}${data.error?`：${data.error}`:""}`:"";$("#logOutput").textContent=(data.log||[]).join("\n")||"[SYSTEM] 暂无运行日志";
    const manualHold=Boolean(data.manual_intervention_latched);$("#manualResumeCard").hidden=!manualHold;
  }catch(error){$("#systemStatus").textContent="连接中断";$("#systemChip").classList.remove("live");}
}
function renderTaskTotals(run){
  $("#creditsTotal").textContent=number(run?.credits);
  $("#shipXpTotal").textContent=number(run?.ship_xp);
  $("#freeXpTotal").textContent=number(run?.free_xp);
  const badge=$("#taskBadge");
  badge.textContent=run?(statusLabels[run.status]||run.status):"暂无任务";
  badge.className=`task-badge ${run?.status||"idle"}`;
  $("#taskNote").textContent=run
    ? `${state.config?.ships[run.ship]?.name||run.ship} · ${labels[run.mode]||run.mode}；本组完成后保留，开始新任务时归零。`
    : "开始任务后从零累计，完成后保留；下一次开始时自动重置。";
}
async function loadDashboard(){const data=await api("/api/dashboard");const runs=data.runs||[];const current=(state.runId&&runs.find(run=>run.id===state.runId))||runs[0]||null;renderTaskTotals(current);}
async function init(){state.config=await api("/api/config");state.calibration=state.config.calibration;if(!state.config.ships[state.ship])state.ship=Object.keys(state.config.ships)[0];renderConfig();renderCalibration();updateLimit();renderPrimaryAction();await pollStatus();await loadDashboard();$("#startBtn").onclick=primaryAction;$("#retryBtn").onclick=startRun;$("#manualResumeBtn").onclick=resumeAutomation;$("#stopBtn").onclick=stopRun;$("#limitValue").oninput=updateLimit;$("#minusBtn").onclick=()=>{$("#limitValue").stepDown();updateLimit();};$("#plusBtn").onclick=()=>{$("#limitValue").stepUp();updateLimit();};document.querySelectorAll(".limit-tab").forEach(button=>button.onclick=()=>{state.limitType=button.dataset.limit;$("#limitValue").max=state.limitType==="rounds"?100:1440;updateLimit();});setInterval(pollStatus,1000);setInterval(loadDashboard,5000);setInterval(()=>$("#clock").textContent=new Date().toLocaleTimeString("zh-CN",{hour12:false}),1000);}
init().catch(error=>toast(error.message,true));
