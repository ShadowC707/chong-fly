/* Local assets only. This file observes recorded physics; no control law here. */
const $=id=>document.getElementById(id);
const clientId=crypto.randomUUID();
let mode='live', recording=null, frames=[], manifest={}, state=null, cursor=0, playing=false;
let received=0, liveSession=null, requestChain=Promise.resolve(), loading=false, pollBusy=false, sourceEpoch=0;
let droneVisualTemplate=null;
let sceneDirty=true;
const keys=new Set(), tofCells=[];
const fmt=(v,n=2)=>v===null||v===undefined||!Number.isFinite(v)?'—':v.toFixed(n);
const stages=[['network_pwm','Network','#a391d6'],['coordinated_pwm','Coordinated','#d698cb'],
 ['supervised_pwm','Supervised','#e0ca7c'],['issued_pwm','Issued','#619bdc'],
 ['accepted_pwm','Accepted','#93c698'],['manual_pwm','Manual','#f5ac68'],['plant_input_pwm','Plant (reference)','#62d6dd'],
 ['fc_rc_pwm','SITL RC','#e28f5f']];
const enabled=new Set(['network_pwm','coordinated_pwm','accepted_pwm','plant_input_pwm']);
for(const [key,label,color] of stages){const l=document.createElement('label'); l.style.color=color;
 const input=document.createElement('input');input.type='checkbox';input.checked=enabled.has(key);
 input.onchange=()=>{input.checked?enabled.add(key):enabled.delete(key);draw();};
 l.append(input,document.createTextNode(label));$('stageToggles').append(l);}
for(let i=0;i<64;i++){const cell=document.createElement('div');$('tof').append(cell);tofCells.push(cell);}
function error(e){let message=e?String(e.message||e):'';
 if(message.startsWith('Another browser owns'))message='Інше вікно керує польотом. Зупиніть його або зачекайте після закриття.';
 if(message.startsWith('Live duration must exceed'))message='Тривалість має бути більшою за час, що вже минув. Для завершеного прогону потрібен новий старт.';
 if(message==='Start intersects scene geometry')message='Стартова позиція торкається перешкоди. Змініть координати старту або геометрію.';
 if(message==='Initial scene intersects collision shell')message='Оболонка дрона перетинає геометрію на старті. Змініть координати або сцену.';
 $('error').hidden=!message;$('error').textContent=message;}
async function api(path,data){const response=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_id:clientId,...data})}:{});
 const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);return result;}
function send(event){requestChain=requestChain.then(async()=>{if(mode!=='live'||!state)return;
 const result=await api('/api/event',{session_id:state.session_id,revision:state.revision,event});state=result.state;draw();}).catch(error);return requestChain;}
function findFrame(t){let lo=0,hi=frames.length;while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].time_s<=t+1e-9)lo=mid+1;else hi=mid;}return frames[Math.max(0,lo-1)];}
function selectMode(next){if(next===mode)return;sourceEpoch++; if(mode==='live'&&state)send({type:'focus_lost'});
 // Wait for revocation before switching the visible source.
 requestChain=requestChain.then(async()=>{mode=next;playing=false;keys.clear();frames=[];recording=null;cursor=0;received=0;
 $('replayTab').classList.toggle('selected',mode==='replay');$('liveTab').classList.toggle('selected',mode==='live');
 $('replayTools').hidden=mode!=='replay';$('liveTools').hidden=mode!=='live';$('authorityTools').hidden=mode!=='live';
 $('replayTransport').hidden=mode!=='replay';$('flightHelp').hidden=mode!=='live';$('export').hidden=mode!=='live';
 $('seek').disabled=mode==='live';$('speed').disabled=mode==='live';$('step').disabled=mode==='live';
 $('modeBadge').classList.toggle('live',mode==='live');
 if(mode==='replay')loadSelected();else{manifest=state?await api('/api/manifest'):{};showManifest();if(state)await refreshLive();}draw();});return requestChain;}
$('replayTab').onclick=()=>selectMode('replay');$('liveTab').onclick=()=>selectMode('live');
function validate(data){if(data.schema!=='chong-fly-demo-v1'||!Array.isArray(data.frames)||!data.frames.length||!data.manifest)throw Error('Непідтримуваний або порожній запис');
 let previous=-1;for(const f of data.frames){if(!Number.isFinite(f.time_s)||f.time_s<=previous||!Array.isArray(f.position_m)||f.position_m.length!==3||!f.position_m.every(Number.isFinite)||!f.trace)throw Error('Пошкоджені час або координати запису');previous=f.time_s;
 for(const [s]of stages){const pwm=f[s]??f.trace[s];if(pwm!==null&&pwm!==undefined&&(!Array.isArray(pwm)||pwm.length!==4||pwm.some(x=>!Number.isFinite(x)||x<1000||x>2000)))throw Error('Пошкоджені PWM у записі');}
 if(f.sensors)for(const [ch,n]of [['tof',64],['flow',2],['altitude',3],['attitude',2]]){const m=f.sensors[ch];if(!m||!Array.isArray(m.value)||m.value.length!==n||!m.value.every(Number.isFinite)||typeof m.valid!=='boolean'||!Number.isFinite(m.timestamp_s)||m.timestamp_s<0||m.timestamp_s>f.time_s+1e-8)throw Error('Пошкоджений сенсорний кадр');}}
 return data;}
function useRecording(data){recording=validate(data);frames=recording.frames;manifest=recording.manifest;cursor=0;playing=false;error(null);
 $('seek').max=recording.end_time_s??frames.at(-1).time_s;showManifest();camera.local=true;$('cameraFocus').value='local';draw();}
async function loadSelected(){if(!$('recordings').value||mode!=='replay')return;const epoch=++sourceEpoch;
 try{const data=await api('/api/replay?name='+encodeURIComponent($('recordings').value));if(mode==='replay'&&epoch===sourceEpoch)useRecording(data);}catch(e){if(epoch===sourceEpoch)error(e);}}
$('recordings').onchange=loadSelected;
$('upload').onchange=async event=>{const epoch=++sourceEpoch;try{let bytes=event.target.files[0];if(!bytes)return;let text;
 if(bytes.name.endsWith('.gz'))text=await new Response(bytes.stream().pipeThrough(new DecompressionStream('gzip'))).text();else text=await bytes.text();
 if(mode==='replay'&&epoch===sourceEpoch)useRecording(JSON.parse(text));}catch(e){if(epoch===sourceEpoch)error(e);}};
function showManifest(){$('manifest').textContent=JSON.stringify(manifest,null,2);$('sceneName').textContent=manifest.scene||manifest.original?.scene||'';
 const sitl=manifest.backend==='sitl';
 $('profileLabel').textContent=sitl?'Candidate A · SmoothPhasedCoordinator · TWR 5':manifest.candidate||'';
 $('profileDetails').textContent=sitl?
 'Ідеальні flow і attitude 50 Гц, дальномір 30 Гц, ToF 15 Гц. Приймач підтверджує Accepted у reference PWM. SITL RC показує команду після перетворення. Сферичний контакт завершує політ.':
 'Чисельна симуляція з ідеалізованими сенсорами та програмним приймачем. Деталі моделі, сенсорів і меж перевірки наведено в маніфесті.';
 const spec=manifest.geometry_spec;let summary='';
 if(spec&&mode==='live'){
 if(Number.isFinite(spec.actual_density_pct))summary=`У створеній сцені: ${fmt(spec.actual_density_pct,1)}% площі · ${spec.random_obstacle_count} випадкових перешкод${spec.settings.wall_enabled?' + стіна':''}.`+(spec.target_reached?'':` Запитано ${fmt(spec.requested_density_pct,1)}%: бракує вільного місця з потрібними зазорами.`);
 else summary=`У створеній сцені: ${spec.actual_obstacle_count} перешкод.`;
 }$('roomSummary').textContent=summary||'Кімната 8×8 м · щільність 20%';$('roomSummary').hidden=false;
 $('bodySize').textContent=Number.isFinite(manifest.geometry?.collision_radius_m)?`Радіус ${fmt(manifest.geometry.collision_radius_m*100,0)} см`:'Радіус недоступний';}
const referenceOption=new Option('Еталон SITL · кімната 12×12','long_arena');
$('scene').append(referenceOption);
function roomScene(){return ['room','random_room','wall_room'].includes($('scene').value);}
function syncBackend(){
 const sitl=$('backend').value==='sitl';
 $('controllerMode').disabled=sitl;
 $('controllerMode').closest('label').hidden=sitl;
 $('backendHint').textContent=sitl?'Фіксований Candidate A · SmoothPhasedCoordinator · TWR 5':'Окрема фізика Standalone. Для SITL потрібен запуск через Docker Desktop.';
 for(const option of $('scene').options){
  const room=['room','random_room','wall_room'].includes(option.value);
  option.hidden=option.disabled=!room&&(sitl!==(option.value==='long_arena'));
 }
 if($('scene').selectedOptions[0]?.disabled)$('scene').value='random_room';
 $('scene').onchange();
}
$('backend').onchange=()=>{syncBackend();sceneDirty=true;draw();};
function restoreLiveInputs(current){const start=manifest.start,options=manifest.scene_options||{};
 $('backend').value=manifest.backend||'standalone';syncBackend();
 $('controllerMode').value=manifest.controller_mode||'original';
 if(manifest.scene)$('scene').value=manifest.scene;
 if(Number.isFinite(manifest.seed))$('seed').value=manifest.seed;
 if(start){$('startX').value=start.position_m[0];$('startY').value=start.position_m[1];$('startYaw').value=start.yaw_rad*180/Math.PI;}
 for(const [id,key]of [['obstacleDensity','obstacle_density_pct'],['layoutStyle','layout_style'],['wallX','wall_x'],['wallY','wall_y'],['wallLength','wall_length_m'],['wallAxis','wall_axis']])if(options[key]!==undefined)$(id).value=options[key];
 $('wallEnabled').checked=!!options.wall_enabled;$('scene').onchange();
 $('duration').value=current.max_duration_s??manifest.max_duration_s??180;
 const fresh=current.terminal||current.time_s===0;
 $('flightMode').value=fresh||(current.paused&&manifest.backend!=='sitl')?'AUTO':current.request;sceneDirty=false;}
function sceneOptions(){const options={backend:$('backend').value,scene:$('scene').value,seed:Number($('seed').value),start:[Number($('startX').value),Number($('startY').value),1],yaw:Number($('startYaw').value)*Math.PI/180,max_duration_s:Number($('duration').value)};
 if(options.backend==='standalone')options.controller_mode=$('controllerMode').value;
 if(roomScene())options.scene_options={wall_enabled:$('wallEnabled').checked,wall_x:Number($('wallX').value),wall_y:Number($('wallY').value),wall_length_m:Number($('wallLength').value),wall_axis:$('wallAxis').value};
 if($('scene').value==='random_room')Object.assign(options.scene_options,{obstacle_density_pct:Number($('obstacleDensity').value),layout_style:$('layoutStyle').value});
 return options;}
$('scene').onchange=()=>{
 const fixed=$('scene').value==='long_arena';
 for(const id of ['startX','startY','startYaw']){$(id).readOnly=fixed;if(fixed)$(id).value=0;}
 $('roomOptions').hidden=!roomScene();for(const id of ['obstacleDensity','layoutStyle'])$(id).disabled=$('scene').value!=='random_room';
 $('wallEnabled').disabled=$('scene').value==='wall_room';if($('scene').value==='wall_room')$('wallEnabled').checked=true;
};
syncBackend();
$('randomizeRoom').onclick=()=>{$('scene').value='random_room';$('seed').value=crypto.getRandomValues(new Uint32Array(1))[0];$('scene').onchange();$('newRun').click();};
function installLiveRun(result){state=result.state;manifest=result.manifest;recording=null;frames=[];received=0;liveSession=state.session_id;cursor=0;
 camera.local=!manifest.geometry?.room||manifest.geometry.room.x_max-manifest.geometry.room.x_min>10;camera.zoom=1;$('cameraFocus').value=camera.local?'local':'room';sceneDirty=false;keys.clear();error(null);showManifest();draw();}
$('newRun').onclick=async()=>{try{if(loading)return;loading=true;sourceEpoch++;draw();await requestChain;
 const result=await api('/api/new',{options:sceneOptions()});
 installLiveRun(result);
 }catch(e){error(e);}finally{loading=false;draw();}};
$('play').onclick=async()=>{if(mode==='live'){if(loading)return;loading=true;error(null);draw();try{
 const requested=$('flightMode').value;
 if(state&&!state.paused){keys.clear();await send({type:'pause'});return;}
 if(!state||state.terminal||sceneDirty||(state.reset_required&&requested==='AUTO')){
 sourceEpoch++;await requestChain;installLiveRun(await api('/api/new',{options:sceneOptions()}));}
 if(manifest.backend==='sitl')await send({type:'resume',mode:requested});
 else{await send({type:'resume'});await send({type:'mode',mode:requested});}
 }catch(e){error(e);}finally{loading=false;draw();}}
 else if(frames.length){if(cursor>=frames.at(-1).time_s)cursor=0;playing=!playing;draw();}};
$('step').onclick=()=>{playing=false;const f=findFrame(cursor);if(f){const i=frames.indexOf(f);cursor=frames[Math.min(i+1,frames.length-1)].time_s;draw();}};
$('seek').oninput=()=>{playing=false;cursor=Number($('seek').value);draw();};
$('channel').onchange=draw;
$('flightMode').onchange=async()=>{draw();if(mode!=='live'||!state||state.paused)return;
 if(state.reset_required&&$('flightMode').value==='AUTO'){await send({type:'pause'});error('Автономне керування потребує нової сесії. Натисніть «Новий старт».');return;}
 await send({type:'mode',mode:$('flightMode').value});};
let durationTimer=null;
async function updateDuration(){clearTimeout(durationTimer);const seconds=Number($('duration').value);
 if(!Number.isFinite(seconds)||seconds<1||seconds>600){error('Тривалість має бути від 1 до 600 секунд.');return;}
 if(mode==='live'&&state&&!state.terminal){await send({type:'duration',seconds});
 if(state.max_duration_s!==seconds){$('duration').value=state.max_duration_s;return;}}
 error(null);$('durationHint').textContent=state?.terminal?'Застосується під час нового старту':'Тривалість оновлено · час симуляції, 1–600 с';draw();}
$('duration').oninput=()=>{clearTimeout(durationTimer);durationTimer=setTimeout(updateDuration,400);};
$('duration').onblur=updateDuration;
document.querySelectorAll('#sceneSettings input,#sceneSettings select').forEach(input=>input.addEventListener('change',()=>{sceneDirty=true;$('settingsHint').textContent='Налаштування змінено. Застосуйте через «Нова сцена» або наступний старт.';}));
$('stop').onclick=()=>send({type:'stop'});
$('endRun').onclick=()=>{keys.clear();send({type:'end'});};
$('export').onclick=async()=>{try{await send({type:'pause'});const data=await api('/api/export');const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='chong-fly-'+state.session_id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){error(e);}};
const flightCodes=new Set(['KeyW','KeyS','KeyA','KeyD','ArrowLeft','ArrowRight']);
const editable=e=>e.target?.matches('input,textarea,select,[contenteditable=true]');
window.addEventListener('keydown',e=>{if(mode!=='live'||editable(e)||e.ctrlKey||e.altKey||e.metaKey)return;
 if(flightCodes.has(e.code)){e.preventDefault();if(e.repeat||keys.has(e.code)||!state||state.paused)return;keys.add(e.code);$('flightMode').value='MANUAL';send({type:'key',code:e.code,down:true});}
 else if(!e.repeat){const event={KeyM:{type:'mode',mode:'MANUAL'},KeyH:{type:'mode',mode:'SHADOW'},Enter:{type:'mode',mode:'AUTO'},Space:{type:'stop'}}[e.code];
 if(event){e.preventDefault();if(event.mode)$('flightMode').value=event.mode;send(event);}else if(e.code==='KeyP')$('play').click();else if(e.code==='KeyR')$('newRun').click();}});
window.addEventListener('keyup',e=>{if(keys.has(e.code)){keys.delete(e.code);send({type:'key',code:e.code,down:false});}});
function loseFocus(){keys.clear();playing=false;if(mode==='live'&&state)send({type:'focus_lost'});draw();}
window.addEventListener('blur',loseFocus);document.addEventListener('visibilitychange',()=>{if(document.hidden)loseFocus();});
// Text-entry focus also clears held flight keys; never intercept the text itself.
document.addEventListener('focusin',e=>{if(editable(e)&&keys.size){for(const code of keys)send({type:'key',code,down:false});keys.clear();}});
async function refreshLive(){if(pollBusy||mode!=='live'||loading)return;pollBusy=true;const epoch=sourceEpoch;try{
 const result=await api('/api/snapshot?after='+received+'&session_id='+encodeURIComponent(liveSession??''));if(epoch!==sourceEpoch||mode!=='live')return;if(result.state){
 if(liveSession!==result.state.session_id){frames=[];received=0;liveSession=result.state.session_id;manifest=await api('/api/manifest');restoreLiveInputs(result.state);showManifest();}
 if(!state||result.state.session_id!==state.session_id||result.state.revision>=state.revision)state=result.state;
 if(result.count<received){frames=[];received=0;}frames.push(...result.frames);received=result.count;cursor=frames.at(-1)?.time_s||0;
 cursor=state?.time_s??cursor;
 }if(result.error)error(result.error);draw();}catch(e){playing=false;error(e);}finally{pollBusy=false;}}
setInterval(refreshLive,80);
setInterval(()=>{if(mode==='live'&&!document.hidden)api('/api/heartbeat',{}).catch(error);},350);

/* Small software 3D renderer: real XYZ geometry, Euler attitude and orbit camera. */
const camera={az:-1.9,el:.55,zoom:1,top:false,local:true};let drag=null,showCollision=false;
$('collisionOutline').onchange=()=>{showCollision=$('collisionOutline').checked;draw();};
$('cameraView').onchange=()=>{camera.top=$('cameraView').value==='top';draw();};
$('cameraFocus').onchange=()=>{camera.local=$('cameraFocus').value==='local';camera.zoom=1;draw();};
document.querySelectorAll('details').forEach(details=>details.addEventListener('toggle',draw));
$('sceneCanvas').onpointerdown=e=>{drag=[e.clientX,e.clientY];e.target.setPointerCapture(e.pointerId);};
$('sceneCanvas').onpointermove=e=>{if(!drag)return;camera.az+=(e.clientX-drag[0])*.008;camera.el=Math.max(.12,Math.min(1.5,camera.el+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];draw();};
$('sceneCanvas').onpointerup=()=>drag=null;
$('sceneCanvas').addEventListener('wheel',e=>{e.preventDefault();camera.zoom=Math.max(.25,Math.min(8,camera.zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
function context(id){const canvas=$(id),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
 if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);return [ctx,rect.width,rect.height];}
function renderScene(frame){const [ctx,w,h]=context('sceneCanvas');const geometry=manifest.geometry||{},room=geometry.room;
 const position=frame?.position_m||manifest.start?.position_m||[-.7,0,1];
 let center=[position[0]*.5+.3,position[1]*.5,.4],span=5;
 if(!camera.local&&room){center=[(room.x_min+room.x_max)/2,(room.y_min+room.y_max)/2,.5];span=Math.max(room.x_max-room.x_min,room.y_max-room.y_min)*1.3;}
 const az=camera.az,el=camera.top?Math.PI/2:camera.el;let scale=Math.min(w,h)*1.05/span*camera.zoom,anchorX=w/2,anchorY=h*.59;
 if(!camera.local&&room){const bounds=[];
 for(const x of [room.x_min,room.x_max])for(const y of [room.y_min,room.y_max])for(const z of [room.z_min,room.z_max]){
 const dx=x-center[0],dy=y-center[1],dz=z-center[2];bounds.push([-Math.sin(az)*dx+Math.cos(az)*dy,Math.sin(el)*(Math.cos(az)*dx+Math.sin(az)*dy)-Math.cos(el)*dz]);}
 const xs=bounds.map(p=>p[0]),ys=bounds.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 scale=Math.min((w-64)/(x1-x0),(h-110)/(y1-y0))*camera.zoom;anchorX=w/2-(x0+x1)/2*scale;anchorY=(h+30)/2-(y0+y1)/2*scale;}
 const project=p=>{const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];const depth=Math.cos(az)*x+Math.sin(az)*y;
 return [anchorX+(-Math.sin(az)*x+Math.cos(az)*y)*scale,anchorY+(Math.sin(el)*depth-Math.cos(el)*z)*scale,Math.cos(el)*depth+Math.sin(el)*z];};
 function line(points,color,width=1,closed=false,fill=null){ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});if(closed)ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
 const grid=camera.local?5:20;const gx=Math.round(center[0]),gy=Math.round(center[1]);
 const xmin=Math.max(gx-grid,room?.x_min??-Infinity),xmax=Math.min(gx+grid,room?.x_max??Infinity),ymin=Math.max(gy-grid,room?.y_min??-Infinity),ymax=Math.min(gy+grid,room?.y_max??Infinity);
 line([[xmin,ymin,0],[xmax,ymin,0],[xmax,ymax,0],[xmin,ymax,0]],'#304552',1,true,'#12232b');
 for(let x=Math.ceil(xmin);x<=xmax;x++)line([[x,ymin,0],[x,ymax,0]],'#223640');
 for(let y=Math.ceil(ymin);y<=ymax;y++)line([[xmin,y,0],[xmax,y,0]],'#223640');
 if(room){const corners=[[room.x_min,room.y_min],[room.x_max,room.y_min],[room.x_max,room.y_max],[room.x_min,room.y_max]];
 line(corners.map(p=>[...p,room.z_min]),'#486273',1,true);line(corners.map(p=>[...p,room.z_max]),'#304a58',1,true);
 corners.forEach(p=>line([[...p,room.z_min],[...p,room.z_max]],'#304a58'));}
 const faces=[];
 function box(b){const p=[[b.x_min,b.y_min,b.z_min],[b.x_max,b.y_min,b.z_min],[b.x_max,b.y_max,b.z_min],[b.x_min,b.y_max,b.z_min],
 [b.x_min,b.y_min,b.z_max],[b.x_max,b.y_min,b.z_max],[b.x_max,b.y_max,b.z_max],[b.x_min,b.y_max,b.z_max]];
 for(const ids of [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]])faces.push(ids.map(i=>p[i]));}
 for(const b of geometry.boxes||[])box(b);
 for(const c of geometry.cylinders||[]){const points=[];for(let i=0;i<20;i++){const a=i*Math.PI/10;points.push([c.center_x+c.radius*Math.cos(a),c.center_y+c.radius*Math.sin(a),0]);}
 faces.push(points.map(p=>[p[0],p[1],c.height]));for(let i=0;i<20;i++){const a=points[i],b=points[(i+1)%20];faces.push([a,b,[b[0],b[1],c.height],[a[0],a[1],c.height]]);}}
 faces.sort((a,b)=>a.reduce((s,p)=>s+project(p)[2],0)/a.length-b.reduce((s,p)=>s+project(p)[2],0)/b.length);
 for(const face of faces)line(face,'#7e8f91',1,true,'rgba(89,104,108,.57)');
 const history=frames.filter((f,i)=>f.time_s<=cursor&&i%5===0).map(f=>f.position_m);if(history.length>1)line(history,'#62d6dd',2);
 const start=manifest.start?.position_m||frames[0]?.position_m;if(start){const p=project(start);ctx.beginPath();ctx.arc(p[0],p[1],4,0,2*Math.PI);ctx.fillStyle='#9dcca5';ctx.fill();ctx.font='10px Segoe UI';ctx.fillText('СТАРТ',p[0]+8,p[1]-5);}
 if(frame||manifest.start){const a=frame?.attitude_rad||[0,0,manifest.start.yaw_rad];const [roll,pitch,yaw]=a.map(v=>v??0);
 const cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw);
 const body=geometry.drone_visual?.unit_geometry||droneVisualTemplate;
 const radius=geometry.drone_visual?.collision_radius_m??geometry.collision_radius_m;
 const hasBody=body&&Number.isFinite(radius)&&radius>0,bodyScale=hasBody?radius:0;
 const transform=p=>{const x=p[0]*bodyScale,y=(cr*p[1]-sr*p[2])*bodyScale,z=(sr*p[1]+cr*p[2])*bodyScale;const xx=cp*x+sp*z,zz=-sp*x+cp*z;return [position[0]+cy*xx-sy*y,position[1]+sy*xx+cy*y,position[2]+zz];};
 line([[position[0],position[1],0],position],'#66878a',1);
 if(hasBody){
 const shadow=project([position[0],position[1],0]);ctx.beginPath();ctx.ellipse(shadow[0],shadow[1],scale*radius*.8,scale*radius*.3,0,0,Math.PI*2);ctx.fillStyle='#0a1219aa';ctx.fill();
 const stroke=body.stroke_width*radius*scale;
 for(const arm of body.arms)line(arm.map(transform),'#d4e0e4',stroke);
 for(const [x,y,z] of body.rotor_centers){const pts=[];for(let i=0;i<=24;i++){const angle=i*Math.PI/12;pts.push(transform([x+body.rotor_radius*Math.cos(angle),y+body.rotor_radius*Math.sin(angle),z]));}line(pts,x>0?'#f5ac68':'#8ca7b7',stroke,true,'#172731aa');}
 line(body.hull.map(transform),'#d6f0f1',stroke,true,'#39717d');
 line(body.nose.map(transform),'#f5ac68',stroke);
 if(showCollision){const c=project(position);ctx.save();ctx.setLineDash([3,3]);ctx.strokeStyle=frame?.collision?'#ec7c68':'#62d6dd';ctx.lineWidth=.8;ctx.beginPath();ctx.arc(c[0],c[1],radius*scale,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=ctx.strokeStyle;ctx.font='10px Segoe UI';ctx.fillText(`Контакт: R=${fmt(radius,2)} м`,c[0]+radius*scale+6,c[1]-6);ctx.restore();}
 }else{const c=project(position);ctx.fillStyle='#62d6dd';ctx.beginPath();ctx.arc(c[0],c[1],2,0,Math.PI*2);ctx.fill();ctx.font='10px Segoe UI';ctx.fillText('Позиція · габарити недоступні',c[0]+7,c[1]);}
 if(a[0]===null||a[1]===null){ctx.font='10px Segoe UI';ctx.fillStyle='#91a5b2';ctx.fillText('Roll/pitch недоступні · показано лише yaw',18,h-38);}}
 const origin=[center[0]-span*.28,center[1]-span*.22,0];for(const [delta,color,label]of [[[.5,0,0],'#f5ac68','X / уперед'],[[0,.5,0],'#9dcca5','Y / ліворуч'],[[0,0,.5],'#62d6dd','Z']]){
 const end=origin.map((v,i)=>v+delta[i]);line([origin,end],color,2);const q=project(end);ctx.fillStyle=color;ctx.font='9px Segoe UI';ctx.fillText(label,q[0]+4,q[1]-4);}}
function chart(id,series,range,unit){const [ctx,w,h]=context(id),left=48,right=w-18,top=14,bottom=h-23;
 const end=Math.max(.1,frames.at(-1)?.time_s||1),start=0;
 const all=series.flatMap(s=>frames.filter((_,i)=>i%5===0).map(s.read)).filter(Number.isFinite);
 let [min,max]=range||[Math.min(0,...all),Math.max(1,...all)];if(max-min<.1)max=min+.1;
 const x=t=>left+(t-start)/(end-start)*(right-left),y=v=>bottom-(v-min)/(max-min)*(bottom-top);
 ctx.font='9px Consolas';for(let i=0;i<=3;i++){const val=min+(max-min)*i/3;ctx.strokeStyle='#2a3a45';ctx.beginPath();ctx.moveTo(left,y(val));ctx.lineTo(right,y(val));ctx.stroke();ctx.fillStyle='#91a5b2';ctx.fillText(fmt(val,range?0:1),5,y(val)+3);}
 for(let i=0;i<=4;i++){const t=end*i/4;ctx.fillText(fmt(t,1)+'s',x(t)-8,h-6);}ctx.fillText(unit,5,10);
 const stride=Math.max(1,Math.floor(frames.length/1200));
 for(const s of series){ctx.beginPath();ctx.strokeStyle=s.color;ctx.lineWidth=1.4;let open=false,prev=null;
 for(let i=0;i<frames.length;i+=stride){const f=frames[i],v=s.read(f);if(v===null||v===undefined||!Number.isFinite(v)){open=false;prev=null;continue;}
 if(!open){ctx.moveTo(x(f.time_s),y(v));open=true;}else {ctx.lineTo(x(f.time_s),y(prev));ctx.lineTo(x(f.time_s),y(v));}prev=v;}ctx.stroke();}
 ctx.setLineDash([3,3]);ctx.strokeStyle='#e6eef1';ctx.beginPath();ctx.moveTo(x(cursor),top);ctx.lineTo(x(cursor),bottom);ctx.stroke();ctx.setLineDash([]);}
function charts(){const ch=Number($('channel').value);chart('commandChart',stages.filter(([s])=>enabled.has(s)).map(([s,label,color])=>({color,read:f=>(f[s]??f.trace?.[s])?.[ch]??null})),ch===0&&manifest.backend!=='sitl'?[1100,1400]:[1000,2000],'PWM µs');
 chart('physicsChart',[{color:'#62d6dd',read:f=>f.position_m[2]}, {color:'#f5ac68',read:f=>speedXY(f)}, {color:'#9dcca5',read:f=>f.clearance_m??null}],null,'м / м·с⁻¹');}
function speedXY(f){const v=f.velocity_mps||f.body_velocity_mps;return v?Math.hypot(v[0],v[1]):null;}
const reasons={engage_denied:'AUTO не допущено',ready:'Admission готовий',engaged:'AUTO допущено',auto:'AUTO: команда прийнята',manual_override:'Ручне керування + підтримання висоти',fault:'Помилка контуру',reset_required:'Потрібен reset',awaiting_command:'Очікування команди'};
const eventNames={duration:'Тривалість',resume:'Продовження',pause:'Пауза',mode_request:'Запит режиму',plant_authority:'Передавання керування',stop:'STOP',stop_unavailable:'STOP недоступний поза AUTO',termination:'Завершення',bridge_fault:'Помилка контуру',simulation_fault:'Помилка симуляції'};
function drawNeural(f){const n=mode==='live'?state?.neural:null;
 const network=f?.trace?.network_pwm,coordinated=f?.trace?.coordinated_pwm;
 const command=pwm=>pwm?`${fmt(pwm[2],1)} / ${fmt(pwm[3],1)} µs`:'недоступно';
 const sitl=manifest.backend==='sitl';
 $('plantCommandLabel').textContent=sitl?'Вхід Betaflight · SITL RC':'Вхід фізики';
 $('networkCommand').textContent=command(network);$('coordinatorCommand').textContent=command(coordinated);$('plantCommand').textContent=command(sitl?f?.fc_rc_pwm:f?.plant_input_pwm);
 let description;
 if(mode==='replay'){
  $('neuralStatus').textContent=network?'МУХА · запис її команд':'МУХА · у цьому кадрі не виконується';
  description=network?'Мережеві команди записані під час справжнього обчислення прогону. Внутрішній h у цих записах не зберігався; для живої активності перейдіть у «Наживо» та натисніть «Старт».':'Це ручний фрагмент. Виберіть автономний запис або почніть живий політ.';
 }else if(n?.status==='running'){
  $('neuralStatus').textContent=`МУХА · ${n.count} станів · 50 Гц`;
  description=f?.plant_source==='auto'?'Мережа отримує нові сенсори й формує команди. Приймач надав AUTO; показано також зміни координатора й фактичний вхід фізики.':
   state?.request==='AUTO'?'Мережа вже обчислюється. Очікуємо штатного допуску AUTO: до прийняття команди фізику веде ручний нейтральний канал із підтриманням висоти.':'SHADOW: мережа обчислює пропозиції, а фізику веде ручний канал.';
 }else{
  $('neuralStatus').textContent=n?.status==='paused'?'МУХА · пауза':'МУХА · мережа не виконується';
  description=n?.status==='paused'?'Обчислення призупинені разом із часом симуляції.':'Оберіть автономне керування й натисніть «Старт», щоб побачити роботу мережі.';
 }
 $('neuralExplanation').textContent=description;
 const container=$('neurons'),values=n?.values;
 if(!values){container.replaceChildren();container.classList.add('unavailable');container.textContent=mode==='replay'?'Внутрішні стани h у записі недоступні':'Очікування першого обчислення мережі';$('neuralSample').textContent='Стан нейронів h';$('neuralScale').textContent='Живі комірки показують фактичний h за індексами моделі; це rate-state, не спайки й не анатомічне розміщення.';return;}
 container.classList.remove('unavailable');
 if(container.children.length!==values.length){container.replaceChildren();for(let i=0;i<values.length;i++)container.append(document.createElement('i'));}
 const scale=.1; // Display-only fixed scale; raw h and all control math are unchanged.
 values.forEach((v,i)=>{const strength=Math.min(1,Math.abs(v)/scale),color=v<0?[163,145,214]:[98,214,221];
  container.children[i].style.background=`rgb(${color.map((c,j)=>Math.round([28,43,54][j]+(c-[28,43,54][j])*strength)).join(',')})`;
  container.children[i].title=`h[${i}] = ${v.toFixed(6)}`;});
 $('neuralSample').textContent=`${values.length} станів h · t=${fmt(n.control_time_s,3)} s`;
 $('neuralScale').textContent=`Шкала індикатора ±${fmt(scale,2)}, поза нею колір насичений. Реальний h: ${fmt(Math.min(...values),3)}…${fmt(Math.max(...values),3)}. Комірки — індекси моделі, не анатомія; це rate-state, не спайки.`;
}
function draw(){const f=findFrame(cursor);
 const sitl=manifest.backend==='sitl';
 renderScene(mode==='live'&&state?.pose?{...f,...state.pose}:f);if($('diagnostics').open)charts();
 drawNeural(f);
 const paused=mode==='live'?(!state||state.paused):!playing;$('pausedBadge').hidden=!paused;
 $('pausedBadge').textContent=mode==='live'&&state?.terminal?'ЗАВЕРШЕНО':cursor===0?'ГОТОВО':'ПАУЗА';
 const backendLabel=sitl?'BETAFLIGHT SITL':manifest.kind==='legacy_replay'?'ІСТОРИЧНИЙ':'STANDALONE';
 $('modeBadge').textContent=(mode==='live'?'НАЖИВО':'ЗАПИС')+' · '+backendLabel;
 $('realTimeFactor').hidden=mode!=='live'||!sitl||!Number.isFinite(state?.real_time_factor);
 $('realTimeFactor').textContent=`Темп ${fmt(state?.real_time_factor)}× реального часу`;
 $('clockRate').textContent=sitl?'Фізика 4000 Гц · керування 50 Гц':'Фізика Standalone 250 Гц · керування 50 Гц';
 const fresh=!state||state.terminal||sceneDirty||(state.reset_required&&$('flightMode').value==='AUTO');
 $('play').textContent=mode==='live'?(paused?(fresh&&state?'↻ Новий старт':fresh||cursor===0?'▶ Старт':'▶ Продовжити'):'Ⅱ Пауза'):(paused?'▶ Відтворити':'Ⅱ Пауза');
 $('play').disabled=loading;
 $('newRun').disabled=loading;$('flightMode').disabled=loading;$('duration').disabled=loading;
 const clock=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
 const duration=mode==='live'?(state?.max_duration_s??Number($('duration').value)):(recording?.end_time_s??frames.at(-1)?.time_s??0);
 $('time').textContent=clock(cursor)+' / '+clock(duration);$('seek').value=cursor;
 $('stop').hidden=mode!=='live'||paused||f?.plant_source!=='auto';
 $('endRun').hidden=mode!=='live'||!sitl||!state||state.terminal;
 $('controlHint').textContent=$('flightMode').value==='AUTO'?'Мережа керує польотом. WASD — ручне перехоплення.':'W/S — вперед / назад · A/D — убік · ←/→ — поворот';
 $('authority').textContent=mode==='live'&&paused?(state?.terminal?'ПРОГІН ЗАВЕРШЕНО':cursor>0?'ПАУЗА':'ГОТОВО ДО СТАРТУ'):f?.request==='AUTO'&&f?.authority==='MANUAL'?'ДОПУСК АВТОПІЛОТА':({AUTO:'АВТОНОМНИЙ ПОЛІТ',MANUAL:'РУЧНЕ КЕРУВАННЯ',FALLBACK:'РЕЗЕРВНИЙ КАНАЛ',NONE:'ЗАВЕРШЕНО'}[f?.authority]||'ГОТОВО');
 $('source').textContent=mode==='replay'?'Відтворення записаних сенсорів, команд і руху':state?.terminal?'Натисніть «Новий старт» для наступного польоту':state?.reset_required&&paused?'Автономний політ потребує нової сесії':paused&&cursor>0?'Час симуляції зупинено':f?.plant_source==='auto'?'Мережа обчислює команди та керує дроном':f?.request==='AUTO'?'Мережа працює; очікуємо допуску керування':!state||cursor===0?'Натисніть «Старт» для обраного режиму польоту':'Ручний канал із підтриманням висоти';
 const tof=f?.sensors?.tof;const valid=tof?.valid&&tof.age_s<=.1+1e-8;
 tofCells.forEach((cell,i)=>{if(!valid){cell.style.background='';cell.textContent='';cell.title='Недоступно / invalid';return;}
 const v=Math.max(0,Math.min(1,tof.value[i]));cell.style.background=`rgb(${245+(98-245)*v},${172+(214-172)*v},${104+(221-104)*v})`;cell.textContent=(v*3.5).toFixed(1);cell.title=`r${Math.floor(i/8)} c${i%8}: ${(v*3.5).toFixed(3)} м`;});
 $('tofState').textContent=tof?`${tof.valid?'valid':'INVALID'} · ${tof.state} · вік ${fmt(tof.age_s*1000,0)} мс · кадр від ${fmt(tof.timestamp_s,3)} с`:'Повна матриця ToF недоступна в цьому записі';
 $('height').textContent=fmt(f?.position_m?.[2])+' / '+fmt(manifest.target_height_m??1)+' м';
 $('velocity').textContent=fmt(f?speedXY(f):null)+' м/с';$('clearance').textContent=fmt(f?.clearance_m)+' м';$('minTof').textContent=fmt(f?.min_tof_m)+' м';
 $('flow').textContent=(f?.flow_xy||[null,null]).map(x=>fmt(x,3)).join(' / ');
 $('sensorAges').textContent=f?.sensors?['flow','altitude','attitude'].map(n=>`${n}: ${f.sensors[n].valid?'valid':'INVALID'} / ${fmt(f.sensors[n].age_s*1000,0)}ms`).join(' · '):'Capture ages / validity недоступні';
 $('modelMode').textContent=f?.authority==='NONE'?'Завершено':({AUTO:'Авто',SHADOW:'SHADOW',MANUAL:'Ручне'}[f?.model_mode]||'Очікування');
 const pauseNames={ready:'Готово до старту',duration_limit:'Задану тривалість вичерпано',collision:'Зіткнення — прогін завершено',user_end:'Сесію завершено',user_pause:'Політ призупинено',focus_lost:'Пауза: вікно втратило фокус',client_disconnected:'Пауза: зв’язок із вікном перервано'};
 $('decision').textContent=mode==='live'&&state?.paused?(pauseNames[state.pause_reason]||('Пауза: '+state.pause_reason)):
 f?.authority==='NONE'?'Кінцевий стан прогону':
 f?`${f.request==='SHADOW'?'SHADOW: ручний канал керує фізикою':reasons[f.reason]||f.reason} · ${f.model_mode==='MANUAL'?'Модель не виконується':f.model_mode==='SHADOW'?'Preview не керує фізикою':f.trace?.behavior_state||''}`:'Очікування даних';
 if(mode==='live'&&state&&!state.paused&&f?.plant_source==='auto')$('decision').textContent=f.stop_requested?'Гальмування активне':'Автономне керування активне';
 if(f?.decision?.phase)$('decision').textContent+=' · '+f.decision.phase;
 
 $('admission').textContent=(state?.reset_required&&mode==='live'?
 (state.receiver_block_reason==='authority_revoked'?'Звичайне відкликання AUTO. Чинний приймач утримує блокування до нової сесії / R. ':'Аварійне блокування приймача: '+state.receiver_block_reason+'. Потрібна нова сесія / R. '):'')+
 (f?.admission||[]).map(x=>`${x.check}: ${fmt(x.value??x.elapsed,3)} / ${fmt(x.limit??x.minimum??x.required,3)}`).join(' · ');
 if(f?.request==='AUTO'&&f?.authority==='MANUAL'&&f.admission?.some(x=>x.check==='flow'))$('admission').textContent='AUTO очікує: рух/flow завеликий для допуску. M повертає MANUAL одразу; відпускання клавіш не є гальмуванням. '+$('admission').textContent;
 const events=(mode==='live'?state?.events:recording?.events)||[];$('events').replaceChildren();
 for(const event of events.filter(e=>e.time_s<=cursor+1e-8).slice(-12).reverse()){const d=document.createElement('div'),t=document.createElement('time');t.textContent=fmt(event.time_s,2)+'s';d.append(t,document.createTextNode((eventNames[event.kind]||event.kind)+' '+(event.mode||event.source||event.reason||(event.seconds===undefined?(event.active===undefined?'':String(event.active)):event.seconds+' с'))));$('events').append(d);}
 const bursts=mode==='live'?state?.yaw_bursts:recording?.yaw_bursts;
 if(bursts){const p=document.createElement('div');p.textContent=`Yaw bursts (physical-rate detector): ${bursts.events.filter(e=>e.start_s<=cursor).length}. Не кількість PWM-стрибків.`;$('events').append(p);}}
let lastTime=performance.now();function animate(now){const dt=Math.min(.1,(now-lastTime)/1000);lastTime=now;
 if(mode==='replay'&&playing&&frames.length){const end=recording.end_time_s??frames.at(-1).time_s;cursor=Math.min(end,cursor+dt*Number($('speed').value));if(cursor>=end)playing=false;draw();}requestAnimationFrame(animate);}
window.addEventListener('resize',draw);requestAnimationFrame(animate);
api('/api/catalog').then(data=>{droneVisualTemplate=data.drone_visual_template||null;
 $('backend').querySelector('option[value="sitl"]').disabled=!data.sitl_enabled;
 const labels={'03_auto_stop_takeover.json.gz':'МУХА · AUTO, перешкода, STOP','02_shadow.json.gz':'МУХА · SHADOW, лише preview','01_manual.json.gz':'Ручний політ · мережа вимкнена','04_denied_near_obstacle.json.gz':'Відмова допуску AUTO','05_legacy_arena.json.gz':'Історичний автономний прогін'};
 for(const name of data.recordings){const o=document.createElement('option');o.value=name;o.textContent=labels[name]||name;$('recordings').append(o);}
 if(data.recordings.includes('03_auto_stop_takeover.json.gz'))$('recordings').value='03_auto_stop_takeover.json.gz';}).catch(error);
draw();
