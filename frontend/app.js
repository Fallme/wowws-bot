const $ = selector => document.querySelector(selector);
const state = {config:null,calibration:null,mode:"cooperative",ship:"pommern",userSelectedShip:false,limitType:"rounds",running:false,paused:false,runId:null,startedAt:0,completed:0,current:0,rewardsStatus:"pending",rewardsRound:0};
const customStorage={name:"wowws.customShipName",range:"wowws.customSecondaryRange",selected:"wowws.selectedShip"};
const labels={cooperative:"联合作战",asymmetric:"非对称作战"};
const statusLabels={idle:"待机",starting:"启动中",launching_game:"启动游戏",entering_game:"进入游戏",preparing:"准备战斗",battle:"战斗中",paused:"已暂停",collecting_rewards:"结算统计",recovering:"状态恢复",requeueing:"进入下一局",returning:"返回港口",completed:"计划完成",stopped:"已停止",failed:"运行异常"};
const movementLabels={idle:"等待任务",autopilot_route:"游戏自动航行",route_planning:"规划航线",opening:"驶向中央点",route_transit:"按航线推进",search:"驶向中央点",approach:"点内接敌",hold_range:"维持副炮距离",hold_capture:"点内留守",reverse_range:"倒船拉距",separate:"主动拉开",avoid_island:"绕岛修正",disengage:"脱离",evade:"规避鱼雷",manual_pause:"用户接管暂停"};
const routeLabels={unplanned:"未规划",departure:"离开出生点",transit:"驶向中央点",final_approach:"进入点位",station:"点内作战"};

async function api(path,options={}){const response=await fetch(path,{headers:{"Content-Type":"application/json"},...options});const body=await response.json();if(!response.ok)throw new Error(body.error||"请求失败");return body;}
function number(value){return new Intl.NumberFormat("zh-CN").format(Number(value||0));}
function duration(seconds){const s=Math.max(0,Math.floor(Number(seconds)||0));return[Math.floor(s/3600),Math.floor(s%3600/60),s%60].map(v=>String(v).padStart(2,"0")).join(":");}
function toast(message,error=false){const el=$("#toast");el.textContent=message;el.className=`toast show${error?" error":""}`;clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.className="toast",2600);}
function readSaved(key,fallback=""){try{return localStorage.getItem(key)||fallback;}catch{return fallback;}}
function saveCustomShip(){const name=$("#customShipName").value,range=$("#customSecondaryRange").value;try{localStorage.setItem(customStorage.name,name);localStorage.setItem(customStorage.range,range);}catch{}clearTimeout(saveCustomShip.timer);saveCustomShip.timer=setTimeout(()=>api("/api/custom-ship",{method:"POST",body:JSON.stringify({custom_ship_name:name,custom_secondary_range:Number(range)})}).catch(()=>{}),300);}

function renderConfig(){
  $("#modeGrid").innerHTML=Object.entries(state.config.modes).map(([key,name])=>`<button class="choice ${key===state.mode?"selected":""}" data-mode="${key}"><strong>${labels[key]||name}</strong><small>${key==="cooperative"?"稳定循环":"高收益挑战"}</small></button>`).join("");
  const ships=Object.entries(state.config.ships);$("#shipCount").textContent=`${ships.length} 预设 + 自定义`;
  $("#shipList").innerHTML=ships.map(([key,ship])=>`<button class="ship-choice ${key===state.ship?"selected":""}" data-ship="${key}"><span><b>${ship.name}</b><small>${ship.nation.toUpperCase()} · ${ship.type}</small></span><em>${ship.secondary_range} km 副炮</em></button>`).join("")+`<button class="ship-choice custom ${state.ship==="custom"?"selected":""}" data-ship="custom"><span><b>自定义舰船</b><small>按港口完整舰名 OCR 搜索</small></span><em>自定射程</em></button>`;
  $("#customShipFields").hidden=state.ship!=="custom";
  $("#modeGrid").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.mode=button.dataset.mode;renderConfig();});
  $("#shipList").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.ship=button.dataset.ship;state.userSelectedShip=true;try{localStorage.setItem(customStorage.selected,state.ship);}catch{}renderConfig();});
}
function renderCalibration(){const c=state.calibration||{valid:false,reason:"启动时自动检查"};const card=$("#calibrationCard");card.classList.toggle("ready",Boolean(c.valid));$("#calibrationTitle").textContent=c.valid?"系统就绪":"启动时自动检查";$("#calibrationMessage").textContent=c.valid?c.reason:"画面、港口和输入均由系统确认";$("#startBtn").disabled=false;}
function updateLimit(){const continuous=state.limitType==="continuous";const value=Math.max(1,Number($("#limitValue").value||1));$("#numberControl").classList.toggle("disabled",continuous);$("#limitValue").disabled=continuous;$("#minusBtn").disabled=continuous;$("#plusBtn").disabled=continuous;$("#limitUnit").textContent=state.limitType==="rounds"?"局":state.limitType==="duration"?"分钟":"";const quick=Boolean($("#quickBattle")?.checked);$("#limitHint").textContent=quick?"快速测试不统计收益；建议使用运行时长控制总时长":continuous?"持续完整对局，直到手动安全停止":state.limitType==="rounds"?`完整结束 ${value} 局后返回港口并停止`:`达到 ${value} 分钟后完成当前局，再返回港口停止`;document.querySelectorAll(".limit-tab").forEach(button=>button.classList.toggle("active",button.dataset.limit===state.limitType));}
async function startRun(){try{const value=state.limitType==="continuous"?0:Number($("#limitValue").value);const payload={ship:state.ship,mode:state.mode,limit_type:state.limitType,limit_value:value,quick_battle:Boolean($("#quickBattle")?.checked)};if(state.ship==="custom"){const name=$("#customShipName").value.trim(),range=Number($("#customSecondaryRange").value);if(!name)throw new Error("请输入与港口完全一致的舰船名称");if(!Number.isFinite(range)||range<1||range>30)throw new Error("副炮射程必须在 1.0 到 30.0 km 之间");saveCustomShip();payload.custom_ship_name=name;payload.custom_secondary_range=range;}const result=await api("/api/run/start",{method:"POST",body:JSON.stringify(payload)});state.runId=result.run_id;state.startedAt=Date.now()/1000;toast("自动作战已启动");await pollStatus();await loadDashboard();}catch(error){toast(error.message,true);}}
async function stopRun(){try{await api("/api/run/stop",{method:"POST",body:"{}"});toast("将在安全释放控制后停止");}catch(error){toast(error.message,true);}}
async function pauseAutomation(){try{const result=await api("/api/run/pause",{method:"POST",body:"{}"});if(!result.ok)throw new Error("当前没有运行中的任务");state.paused=true;toast("已暂停下发新指令，舰船保持原状态");await pollStatus();}catch(error){toast(error.message,true);}}
async function resumeAutomation(){try{const result=await api("/api/run/resume",{method:"POST",body:"{}"});if(!result.ok)throw new Error("当前没有运行中的任务");state.paused=false;toast("正在识别当前状态并继续原操作");await pollStatus();}catch(error){toast(error.message,true);}}
async function primaryAction(){if(!state.running)return startRun();return state.paused?resumeAutomation():pauseAutomation();}
function renderPrimaryAction(){const button=$("#startBtn"),label=button.querySelector("span"),icon=button.querySelector("b");label.textContent=!state.running?"开始自动作战":state.paused?"继续自动作战":"暂停自动作战";icon.textContent=!state.running?"→":state.paused?"▶":"Ⅱ";button.classList.toggle("paused",state.paused);}
function selectLiveTab(name){document.querySelectorAll(".live-tab").forEach(button=>{const active=button.dataset.liveTab===name;button.classList.toggle("active",active);button.setAttribute("aria-selected",String(active));});document.querySelectorAll("[data-live-pane]").forEach(pane=>pane.hidden=pane.dataset.livePane!==name);}
function normalizedPoint(value){return Array.isArray(value)&&value.length>=2&&value.every(Number.isFinite)?[Math.max(0,Math.min(1,Number(value[0]))),Math.max(0,Math.min(1,Number(value[1])))]:null;}
function normalizedVector(value){return Array.isArray(value)&&value.length>=2&&value.every(Number.isFinite)?[Number(value[0]),Number(value[1])]:null;}
function placeMapElement(element,point){if(!point){element.hidden=true;return;}element.hidden=false;element.style.left=`${point[0]*100}%`;element.style.top=`${point[1]*100}%`;}
function radarNode(className,text,point,radius=0){const node=document.createElement("span");node.className=className;node.textContent=text;node.style.left=`${point[0]*100}%`;node.style.top=`${point[1]*100}%`;if(radius){node.style.width=`${radius*200}%`;node.style.height=`${radius*200}%`;}return node;}
function radarIsland(shape){const points=Array.isArray(shape?.points)?shape.points.map(normalizedPoint).filter(Boolean):[];if(points.length<3)return null;const xs=points.map(point=>point[0]),ys=points.map(point=>point[1]),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys),width=Math.max(.006,right-left),height=Math.max(.006,bottom-top);const node=document.createElement("i");node.className="radar-island";node.style.left=`${left*100}%`;node.style.top=`${top*100}%`;node.style.width=`${width*100}%`;node.style.height=`${height*100}%`;node.style.clipPath=`polygon(${points.map(point=>`${((point[0]-left)/width*100).toFixed(1)}% ${((point[1]-top)/height*100).toFixed(1)}%`).join(",")})`;return node;}
function renderMinimapTelemetry(data){
  const player=normalizedPoint(data.minimap_player),target=normalizedPoint(data.navigation_target),zone=normalizedPoint(data.capture_zone_center),enemy=normalizedPoint(data.nearest_enemy);
  placeMapElement($("#mapPlayer"),player);placeMapElement($("#mapGoal"),target);placeMapElement($("#mapEnemy"),enemy);placeMapElement($("#mapObjective"),zone);
  const heading=normalizedVector(data.minimap_heading);if(player&&heading){const degrees=Math.atan2(heading[1],heading[0])*180/Math.PI+90;$("#mapPlayer").style.setProperty("--heading-deg",`${degrees}deg`);}else $("#mapPlayer").style.setProperty("--heading-deg","0deg");
  const objective=$("#mapObjective"),radius=Math.max(0,Math.min(.25,Number(data.capture_zone_radius)||0));if(zone&&radius){objective.style.width=`${radius*200}%`;objective.style.height=`${radius*200}%`;}$("#mapObjectiveLabel").textContent=data.capture_zone_label||"点";
  const line=$("#routeLine");if(player&&target){const size=$("#routeVisual").clientWidth||210,dx=(target[0]-player[0])*size,dy=(target[1]-player[1])*size;line.hidden=false;line.style.left=`${player[0]*100}%`;line.style.top=`${player[1]*100}%`;line.style.width=`${Math.hypot(dx,dy)}px`;line.style.transform=`rotate(${Math.atan2(dy,dx)*180/Math.PI}deg)`;}else line.hidden=true;
  const zones=Array.isArray(data.capture_zones)?data.capture_zones:[],zoneLayer=$("#mapZones");zoneLayer.replaceChildren(...zones.map(item=>{const point=normalizedPoint(item.position);if(!point)return null;const zoneState=["friendly","hostile","neutral"].includes(item.state)?item.state:"unknown";return radarNode(`radar-zone ${zoneState}`,String(item.label||"点").slice(0,2),point,Math.max(.025,Math.min(.25,Number(item.radius)||.05)));}).filter(Boolean));
  const islands=Array.isArray(data.minimap_islands)?data.minimap_islands:[],islandLayer=$("#mapIslands");islandLayer.replaceChildren(...islands.map(radarIsland).filter(Boolean));
  const contacts=Array.isArray(data.minimap_contacts)?data.minimap_contacts:[],contactLayer=$("#mapContacts");contactLayer.replaceChildren(...contacts.map(item=>{const point=normalizedPoint(item.position);return point?radarNode("radar-contact",item.kind==="enemy"?"◆":"•",point):null;}).filter(Boolean));
  const inside=data.inside_capture_point&&data.capture_zone_label?` · 已进入${data.capture_zone_label}点`:"";$("#routePhase").title=`小地图敌舰 ${Number(data.minimap_enemy_count||0)} 艘${inside}`;
}

async function pollStatus(){
  try{
    const data=await api("/api/status");state.running=data.running;state.paused=Boolean(data.paused_by_user||data.manual_intervention_latched||data.manual_intervention_active);state.runId=data.run_id||state.runId;state.startedAt=data.started_at||state.startedAt;state.completed=Number(data.completed_rounds||0);state.current=Number(data.current_round||0);state.rewardsStatus=data.rewards_status||"pending";state.rewardsRound=Number(data.rewards_round||0);const reportedShip=data.ship;if(!state.userSelectedShip&&reportedShip&&(reportedShip==="custom"||state.config?.ships[reportedShip])&&reportedShip!==state.ship){state.ship=reportedShip;renderConfig();}renderPrimaryAction();
    $("#systemStatus").textContent=statusLabels[data.state]||data.state||"待机";$("#systemChip").classList.toggle("live",data.running);$("#liveBadge").textContent=data.running?"LIVE":"OFFLINE";$("#liveBadge").classList.toggle("live",data.running);
    if(data.calibration)state.calibration=data.calibration;renderCalibration();$("#stopBtn").disabled=!data.running;$("#progressMessage").textContent=data.message||"等待任务启动";$("#runId").textContent=data.run_id?`#${data.run_id}`:"—";
    const ship=data.ship_display_name||state.config?.ships[data.ship]?.name||data.ship||"—";$("#activeSetup").textContent=`${ship} · ${labels[data.mode]||data.mode||"—"}`;
    const elapsed=data.elapsed_seconds!=null?Number(data.elapsed_seconds):(state.startedAt?Date.now()/1000-state.startedAt:0);$("#elapsedTime").textContent=duration(elapsed);
    const max=Number(data.max_rounds||0),durationLimit=Number(data.duration_minutes||0)*60;const progress=max?Math.min(100,state.completed/max*100):durationLimit?Math.min(100,elapsed/durationLimit*100):0;
    $("#roundDisplay").textContent=max?`${state.completed} / ${max} 局`:`${state.completed} 局完成`;$("#progressPercent").textContent=`${Math.round(progress)}%`;$("#progressBar").style.width=`${progress}%`;
    $("#movementMode").textContent=movementLabels[data.movement_mode]||data.movement_mode||"等待任务";$("#movementReason").textContent=data.movement_reason||"配置预设后即可开始";
    const routeProgress=Math.max(0,Math.min(100,Number(data.route_progress||0)*100));$("#routePhase").textContent=(routeLabels[data.route_phase]||data.route_phase||"未规划")+(data.inside_capture_point&&data.capture_zone_label?` · ${data.capture_zone_label}点内`:"");$("#routeProgress").textContent=`${Math.round(routeProgress)}%`;renderMinimapTelemetry(data);
    $("#captureDistance").textContent=data.capture_point_distance_km==null?"—":`${Number(data.capture_point_distance_km).toFixed(1)} km`;
    const source=data.distance_source==="ocr"?"OCR":data.distance_source==="minimap_grid"?"小地图":"";$("#targetDistance").textContent=data.target_distance_km==null?"—":`${Number(data.target_distance_km).toFixed(1)} km${source?` · ${source}`:""}`;
    const provider=data.ocr_provider==="CUDAExecutionProvider"?"GPU":data.ocr_provider==="CPUExecutionProvider"?"CPU":data.ocr_provider||"—";$("#visionStatus").textContent=`${provider} · ${data.frame_status||"—"}`;$("#feedbackStatus").textContent=data.movement_verified?"位移已确认":data.safety_state==="tripped"?"安全熔断":"等待反馈";$("#roundState").textContent=data.stop_after_current?"本局结束后停止":data.state==="battle"?`第 ${state.current} 局进行中`:statusLabels[data.state]||"未开始";
    $("#autopilotStatus").textContent=data.autopilot_enabled?"已开启 · Q/E互锁":"未开启 · 小地图接管";
    const observedRudder=data.rudder_indicator||"neutral",commanded=Number(data.commanded_rudder||0);$("#rudderStatus").textContent=observedRudder==="Q"?"Q 左舵":observedRudder==="E"?"E 右舵":observedRudder==="ambiguous"?"舵位识别中":Math.abs(commanded)>=.1?(commanded<0?"系统左舵":"系统右舵"):"中舵";
    const island=data.island_distance==null?"山体安全":`前方山体 ${Number(data.island_distance).toFixed(2)}`,enemyCount=Number(data.minimap_enemy_count||0);$("#mapSituation").textContent=`敌舰 ${enemyCount} · ${island}`;
    const navLabels={native_autopilot:"原生自动航线",minimap_capture_zone:"点位远端",minimap_capture_zone_fallback:"点位远端接管",minimap_center:"地图中央"};$("#navigationStatus").textContent=navLabels[data.navigation_source]||data.navigation_source||"等待小地图";
    const hp=data.health_percent==null?"生命 —":`生命 ${Number(data.health_percent).toFixed(0)}%`;const hazards=[data.on_fire?"着火":"",data.flooding?"漏水":""].filter(Boolean);$("#shipStatus").textContent=`${hp}${hazards.length?` · ${hazards.join("/")}`:" · 正常"}`;$("#speedStatus").textContent=data.speed_knots==null?"—":`${Number(data.speed_knots).toFixed(1)} kt`;
    const headingVector=normalizedVector(data.minimap_heading);$("#headingStatus").textContent=headingVector?`${Math.round((Math.atan2(headingVector[1],headingVector[0])*180/Math.PI+450)%360)}°`:"—";$("#consumableStatus").textContent=`R ${data.damage_control_ready?"可用":"冷却"} · T ${data.heal_ready?"可用":"冷却"}`;
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
    ? state.runId===run.id&&state.rewardsStatus==="unrecognized"
      ? `第 ${state.rewardsRound||state.completed} 局结算已确认，但数字未可靠识别；本次未写入错误资源，已保留诊断截图。`
      : `${state.config?.ships[run.ship]?.name||run.ship} · ${labels[run.mode]||run.mode}；本组完成后保留，开始新任务时归零。`
    : "开始任务后从零累计，完成后保留；下一次开始时自动重置。";
}
function renderTaskHistory(current,runs){
  const allRuns=Array.isArray(runs)?runs:[];
  const previous=state.running&&current?allRuns.filter(item=>item.id!==current.id):allRuns;
  const entries=previous.slice(0,4),status={completed:"完成",stopped:"已停止",failed:"异常",running:"进行中"};
  $("#historySummary").textContent=entries.length?`最近 ${entries.length} 组`:"暂无历史任务";
  $("#taskHistory").replaceChildren(...(entries.length?entries.map(item=>{
    const row=document.createElement("article"),ship=state.config?.ships[item.ship]?.name||item.ship||"未知舰船";
    const started=Number(item.started_at||0)?new Date(Number(item.started_at)*1000).toLocaleDateString("zh-CN",{month:"2-digit",day:"2-digit"}):"—";
    const rounds=Number(item.completed_rounds||0),wins=Number(item.victories||0),losses=Number(item.defeats||0);
    row.className=`history-row task ${item.status||"unknown"}`;
    row.innerHTML=`<div><strong>${ship} · ${labels[item.mode]||item.mode||"—"}</strong><small>${started} · ${rounds} 局 · 胜 ${wins} / 负 ${losses}</small></div><b>${status[item.status]||"已记录"}</b>`;
    return row;
  }):[Object.assign(document.createElement("p"),{className:"history-empty",textContent:"完成过的自动作战任务会显示在这里。"})]));
}
async function loadDashboard(){const data=await api("/api/dashboard");const runs=data.runs||[];const current=(state.runId&&runs.find(run=>run.id===state.runId))||runs[0]||null;renderTaskTotals(current);renderTaskHistory(current,runs);}
async function init(){state.config=await api("/api/config");state.calibration=state.config.calibration;const savedShip=readSaved(customStorage.selected);if(savedShip==="custom"||state.config.ships[savedShip])state.ship=savedShip;if(state.ship!=="custom"&&!state.config.ships[state.ship])state.ship=Object.keys(state.config.ships)[0];const savedCustom=state.config.custom_ship||{};$("#customShipName").value=readSaved(customStorage.name,savedCustom.name||"");$("#customSecondaryRange").value=readSaved(customStorage.range,String(savedCustom.secondary_range||10));$("#customShipName").oninput=saveCustomShip;$("#customSecondaryRange").oninput=saveCustomShip;renderConfig();renderCalibration();updateLimit();renderPrimaryAction();await pollStatus();await loadDashboard();$("#startBtn").onclick=primaryAction;$("#retryBtn").onclick=startRun;$("#manualResumeBtn").onclick=resumeAutomation;$("#stopBtn").onclick=stopRun;$("#limitValue").oninput=updateLimit;$("#quickBattle").onchange=updateLimit;$("#minusBtn").onclick=()=>{$("#limitValue").stepDown();updateLimit();};$("#plusBtn").onclick=()=>{$("#limitValue").stepUp();updateLimit();};document.querySelectorAll(".limit-tab").forEach(button=>button.onclick=()=>{state.limitType=button.dataset.limit;$("#limitValue").max=state.limitType==="rounds"?100:1440;updateLimit();});document.querySelectorAll(".live-tab").forEach(button=>button.onclick=()=>selectLiveTab(button.dataset.liveTab));setInterval(pollStatus,1000);setInterval(loadDashboard,5000);setInterval(()=>$("#clock").textContent=new Date().toLocaleTimeString("zh-CN",{hour12:false}),1000);}
init().catch(error=>toast(error.message,true));
