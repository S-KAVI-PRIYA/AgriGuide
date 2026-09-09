// =====================================================
// Config — point this at your FastAPI wrapper (app.py)
// =====================================================
const API_BASE = window.AGRI_API_BASE || "http://127.0.0.1:8000";

// =====================================================
// Hero field illustration — animated growing crop rows
// =====================================================
(function buildField(){
  const rowsGroup = document.getElementById("rows");
  if(!rowsGroup) return;
  const rowCount = 6, perRow = 9;
  const startY = 300, rowGap = 20;
  for(let r=0;r<rowCount;r++){
    const y = startY + r*rowGap;
    for(let i=0;i<perRow;i++){
      const x = 20 + i*58 + (r%2===0?0:20);
      const h = 26 + Math.random()*14;
      const delay = (r*perRow + i) * 0.03;
      const blade = document.createElementNS("http://www.w3.org/2000/svg","path");
      blade.setAttribute("d", `M${x},${y} q-4,-${h*0.6} 0,-${h} q4,${h*0.6} 0,${h}`);
      blade.setAttribute("fill", r % 3 === 0 ? "#7FA84F" : (r % 3 === 1 ? "#5C8A3E" : "#3F6B2C"));
      blade.style.transformOrigin = `${x}px ${y}px`;
      blade.style.animation = `growBlade .6s ease-out ${delay}s both`;
      rowsGroup.appendChild(blade);
    }
  }
  const styleTag = document.createElement("style");
  styleTag.textContent = `@keyframes growBlade{from{transform:scaleY(0);opacity:0;}to{transform:scaleY(1);opacity:1;}}`;
  document.head.appendChild(styleTag);
})();

// =====================================================
// Soil data ticker — sample readings styled like sensor feed
// (mirrors the six signals used by the backend dataset)
// =====================================================
const SAMPLE_READINGS = [
  {crop:"Rice", n:80, p:47, k:39, temp:23.7, hum:82.3, ph:6.4, rain:236.2},
  {crop:"Maize", n:71, p:54, k:16, temp:22.4, hum:65.1, ph:6.2, rain:84.8},
  {crop:"Chickpea", n:40, p:67, k:79, temp:18.9, hum:16.9, ph:7.3, rain:80.1},
  {crop:"Cotton", n:117, p:46, k:19, temp:24.0, hum:79.8, ph:6.9, rain:80.4},
  {crop:"Banana", n:100, p:82, k:50, temp:27.4, hum:80.4, ph:5.9, rain:104.6},
  {crop:"Coffee", n:101, p:28, k:29, temp:25.5, hum:58.9, ph:6.8, rain:158.1},
];

function buildTicker(){
  const track = document.getElementById("tickerTrack");
  if(!track) return;
  const renderSet = () => SAMPLE_READINGS.map(r => `
    <span class="ticker__item">
      <b>${r.crop}</b>
      <span class="sep">·</span> N ${r.n}
      <span class="sep">·</span> P ${r.p}
      <span class="sep">·</span> K ${r.k}
      <span class="sep">·</span> ${r.temp}°C
      <span class="sep">·</span> ${r.hum}% RH
      <span class="sep">·</span> pH ${r.ph}
      <span class="sep">·</span> ${r.rain}mm
    </span>`).join("");
  track.innerHTML = renderSet() + renderSet(); // duplicate for seamless loop
}
buildTicker();

// =====================================================
// Backend health check
// =====================================================
async function checkHealth(){
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try{
    const res = await fetch(`${API_BASE}/api/health`, { method:"GET" });
    if(!res.ok) throw new Error("bad status");
    const data = await res.json();
    dot.classList.add("online");
    text.textContent = `Connected · ${data.records ?? "?"} records indexed`;
  }catch(err){
    dot.classList.add("offline");
    text.textContent = "Backend offline — run app.py to connect";
  }
}
checkHealth();

// =====================================================
// Chat logic
// =====================================================
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");

function scrollToBottom(){
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addMessage(text, who, sourceLabel){
  const wrap = document.createElement("div");
  wrap.className = `msg msg--${who}`;
  const avatar = document.createElement("div");
  avatar.className = "msg__avatar";
  avatar.textContent = who === "user" ? "🧑‍🌾" : "🌾";
  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = text;
  if(sourceLabel){
    const src = document.createElement("span");
    src.className = "msg__source";
    src.textContent = sourceLabel;
    bubble.appendChild(src);
  }
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function addTyping(){
  const wrap = document.createElement("div");
  wrap.className = "msg msg--bot msg--typing";
  wrap.id = "typingIndicator";
  wrap.innerHTML = `
    <div class="msg__avatar">🌾</div>
    <div class="msg__bubble">
      <span class="dot-typing"></span><span class="dot-typing"></span><span class="dot-typing"></span>
    </div>`;
  chatLog.appendChild(wrap);
  scrollToBottom();
}

function removeTyping(){
  const el = document.getElementById("typingIndicator");
  if(el) el.remove();
}

// Splits backend answer text out from its trailing "[Source: ...]" marker
function splitSource(rawText){
  const match = rawText.match(/\[Source:\s*([^\]]+)\]\s*$/i);
  if(match){
    return {
      body: rawText.slice(0, match.index).trim(),
      source: match[1].trim()
    };
  }
  return { body: rawText.trim(), source: null };
}

async function askBackend(question){
  addTyping();
  try{
    const res = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    if(!res.ok) throw new Error(`Server responded ${res.status}`);
    const data = await res.json();
    removeTyping();
    const { body, source } = splitSource(data.answer || "No answer returned.");
    addMessage(body, "bot", source ? `Source: ${source}` : null);
  }catch(err){
    removeTyping();
    addMessage(
      "I couldn't reach the backend. Make sure app.py is running " +
      "(uvicorn app:app --reload) and GEMINI_API_KEY is set, then try again.",
      "bot"
    );
    console.error(err);
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if(!q) return;
  addMessage(q, "user");
  chatInput.value = "";
  askBackend(q);
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    chatInput.value = chip.textContent;
    chatForm.dispatchEvent(new Event("submit"));
  });
});

// =====================================================
// Voice input (browser Speech Recognition — feeds the same /api/ask)
// =====================================================
const micBtn = document.getElementById("micBtn");
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if(SpeechRecognition){
  const recognizer = new SpeechRecognition();
  recognizer.lang = "en-US";
  recognizer.interimResults = false;

  let listening = false;

  micBtn.addEventListener("click", () => {
    if(listening){
      recognizer.stop();
      return;
    }
    recognizer.start();
  });

  recognizer.addEventListener("start", () => {
    listening = true;
    micBtn.classList.add("recording");
  });

  recognizer.addEventListener("end", () => {
    listening = false;
    micBtn.classList.remove("recording");
  });

  recognizer.addEventListener("result", (event) => {
    const transcript = event.results[0][0].transcript;
    chatInput.value = transcript;
    chatForm.dispatchEvent(new Event("submit"));
  });

  recognizer.addEventListener("error", () => {
    listening = false;
    micBtn.classList.remove("recording");
  });
}else{
  micBtn.addEventListener("click", () => {
    addMessage("Voice input isn't supported in this browser — try Chrome, or type your question instead.", "bot");
  });
}