// 녹음·업로드한 소리를 바로 들어볼 수 있게 재생기를 붙인다(음량 표시 포함).
// 기본 페이지는 건드리지 않고 NAT_INDEX_EXTRA_JS 로 켠 인스턴스에만 주입된다.
(function(){
  const card=document.querySelector('#rec').closest('div');
  const box=document.createElement('div');
  box.style.cssText='margin-top:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap';
  box.innerHTML='<audio id="playback" controls preload="none" style="height:34px;flex:1;min-width:240px;display:none"></audio>'
    + '<span id="lvl" style="font-size:13px;opacity:.75"></span>';
  card.appendChild(box);
  const audio=box.querySelector('#playback'), lvl=box.querySelector('#lvl');
  function show(blob){
    if(audio.src) URL.revokeObjectURL(audio.src);
    audio.src=URL.createObjectURL(blob); audio.style.display='block';
  }
  const origFetch=window.fetch;
  window.fetch=async function(url, opt){
    if(typeof url==='string' && url.endsWith('/predict') && opt && opt.body instanceof FormData){
      const f=opt.body.get('audio'); if(f) show(f);
      const r=await origFetch.apply(this, arguments);
      try{
        const d=await r.clone().json();
        if(d && d.rms!=null) lvl.textContent=`입력 음량 ${d.rms}` + (d.gain&&d.gain!==1? ` · 정규화 ×${d.gain}`:'')
          + (d.rms<0.01? ' — 너무 작다. 마이크에 가까이서 크게 말할 것':'');
      }catch(e){}
      return r;
    }
    return origFetch.apply(this, arguments);
  };
})();
