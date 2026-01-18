import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import "./App.css";

// --- COOKIE HELPERS ---
const setCookie = (name, value, days) => {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = name + '=' + encodeURIComponent(JSON.stringify(value)) + '; expires=' + expires + '; path=/';
};

const getCookie = (name) => {
  return document.cookie.split('; ').reduce((r, v) => {
    const parts = v.split('=');
    return parts[0] === name ? JSON.parse(decodeURIComponent(parts[1])) : r;
  }, []);
};

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // --- STATE ---
  const [yoloModel, setYoloModel] = useState(null);
  const [mpLandmarker, setMpLandmarker] = useState(null);
  const [gesture, setGesture] = useState("⏳ Loading Models...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 640, h: 480 });

  // --- UI STATE ---
  const [theme, setTheme] = useState("dark");
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => window.innerWidth > 900);
  const [history, setHistory] = useState([]); 
  const [editingIndex, setEditingIndex] = useState(null); 

  // --- CONFIG ---
  const YOLO_PATH = 'best_web_model/model.json';
  const INPUT_SIZE = 320;
  const CONFIDENCE_THRESHOLD = 0.81; 
  const STABLE_THRESHOLD = 10;

  // --- REFS ---
  const gestureBuffer = useRef([]);
  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const frozenBox = useRef(null);
  const lastBoxUpdate = useRef(0);
  const wasHandPresent = useRef(false);

  // --- 0. THEME & RESIZE EFFECTS ---
  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 900) setIsSidebarOpen(false);
      else setIsSidebarOpen(true);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --- NEW: RESIZE HANDLER ---
  useEffect(() => {
    const handleResize = () => {
      // 1. Calculate new width: max 640px, or 90% of mobile screen
      const newWidth = Math.min(window.innerWidth * 0.90, 640);
      
      // 2. Calculate height maintaining 4:3 Aspect Ratio
      const newHeight = newWidth / (4/3); 
      
      // 3. Update state (this auto-scales the canvas & detection boxes)
      setDim({ w: newWidth, h: newHeight });
    };

    // Run once on mount
    handleResize();

    // Listen for window resize events
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const toggleTheme = () => {
    setTheme(prev => prev === "dark" ? "light" : "dark");
  };

  // --- 1. LOAD MODELS & HISTORY ---
  useEffect(() => {
    try {
      const savedHistory = getCookie('signeticHistory');
      if (Array.isArray(savedHistory)) setHistory(savedHistory);
    } catch (e) { console.error("Cookie error", e); }

    const loadModels = async () => {
      try {
        setGesture("⏳ Downloading Models...");
        await tf.setBackend('webgl');
        await tf.ready();

        const [loadedYolo, vision] = await Promise.all([
          tf.loadGraphModel(YOLO_PATH),
          FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm")
        ]);

        console.log("✅ Downloads Complete");
        setGesture("⚙️ Warming up GPU...");

        const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
        await loadedYolo.executeAsync(dummy);
        dummy.dispose();
        setYoloModel(loadedYolo);

        const loadedMp = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          numHands: 1,
          runningMode: "VIDEO",
          minHandDetectionConfidence: 0.5
        });
        setMpLandmarker(loadedMp);

        setGesture("👀 Show Hand");

      } catch (err) {
        console.error(err);
        setGesture(`Error: ${err.message}`); 
      }
    };
    loadModels();
  }, []);

  // --- 2. SAFE DRAWING ---
  const drawOnCanvas = (ctx, detection, frozen, dim) => {
    ctx.clearRect(0, 0, dim.w, dim.h);
    if (frozen) {
      const [x, y, w, h] = frozen;
      ctx.strokeStyle = "rgba(0, 150, 255, 0.6)";
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 5]); 
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]); 
    }
    if (detection && detection.box) {
      const [x, y, w, h] = detection.box;
      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "#00FF00";
      ctx.fillRect(x, y - 25, 80, 25);
      ctx.fillStyle = "black";
      ctx.font = "bold 16px Arial";
      const scoreTxt = detection.score ? Math.round(detection.score * 100) + "%" : "Hand";
      ctx.fillText(scoreTxt, x + 5, y - 7);
    }
  };

  // --- 3. RECOGNITION LOGIC ---
  const recognizeGesture = useCallback((hand) => {
    const k = hand.keypoints;
    if (!k || k.length < 21) return "";
    const handSize = Math.hypot(k[9].x - k[0].x, k[9].y - k[0].y);
    const T = (factor) => handSize * factor;
    const isExtended = (tipIdx, mcpIdx) => Math.hypot(k[tipIdx].x - k[mcpIdx].x, k[tipIdx].y - k[mcpIdx].y) > T(0.55);

    const indexExt = isExtended(8, 5);
    const middleExt = isExtended(12, 9);
    const ringExt = isExtended(16, 13);
    const pinkyExt = isExtended(20, 17);
    const d = (i1, i2) => Math.hypot(k[i1].x - k[i2].x, k[i1].y - k[i2].y);
    const pointingDown = k[8].y > k[0].y;

    if (pointingDown) {
      if (indexExt && !middleExt && !ringExt && !pinkyExt && d(4, 8) > T(0.6)) return "Q";
      if (indexExt && middleExt && !ringExt && !pinkyExt) return "P";
    }
    const isHorizontal = Math.abs(k[8].x - k[5].x) > Math.abs(k[8].y - k[5].y) * 1.5;
    if (isHorizontal && !ringExt && !pinkyExt) { return middleExt ? "H" : "G"; }

    if (!middleExt && !ringExt && !pinkyExt) {
      if (indexExt) return d(4, 5) > T(0.9) ? "L" : "D";
    }
    if (indexExt && middleExt && !ringExt && !pinkyExt) {
      if (Math.abs(k[8].x - k[12].x) < T(0.25)) return "R";
      if (k[4].y < k[9].y + T(0.2)) return "K";
      return d(8, 12) > T(0.45) ? "V" : "U";
    }
    if (indexExt && middleExt && ringExt) {
      if (pinkyExt) return d(4, 17) < T(1.0) ? "B" : "🖐️";
      return "W";
    }
    if (!indexExt && middleExt && ringExt && pinkyExt) {
      if (d(4, 8) < T(0.5)) return "F";
    }
    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
      return d(4, 17) > T(1.1) ? "Y" : "I";
    }
    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
      const indexLen = d(8, 5); const gap = d(4, 8);
      if (indexLen > T(0.55) && indexLen < T(0.85) && gap > T(0.5) && gap < T(0.9)) return "C";
      const indexCurl = d(8, 5);
      if (d(4, 8) < T(0.5)) {
        if (indexCurl < T(0.35)) return d(4, 10) < T(0.35) ? "S" : "E";
        return "O";
      }
      if (d(4, 5) > T(0.5) && k[4].y < k[5].y) return "A";
      if (d(4, 13) < T(0.3) || d(4, 17) < T(0.35)) return "M";
      if (d(4, 9) < T(0.3)) return "N";
      if (d(4, 5) < T(0.35)) return "T";
      if (indexCurl < T(0.35) && d(4, 13) < T(0.5)) return "E";
      return "S";
    }
    return "🖐️";
  }, []);

  const detect = useCallback(async () => {
    if (!yoloModel || !mpLandmarker || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    // --- MEDIAPIPE ---
    const mpResult = mpLandmarker.detectForVideo(video, Date.now());
    if (mpResult.landmarks && mpResult.landmarks.length > 0) {
      const resultGesture = recognizeGesture({ keypoints: mpResult.landmarks[0] });
      if (resultGesture) {
        gestureBuffer.current.push(resultGesture);
        if (gestureBuffer.current.length > 5) gestureBuffer.current.shift();
        const counts = {}; let max = 0; let stable = resultGesture;
        gestureBuffer.current.forEach(g => { 
          counts[g] = (counts[g] || 0) + 1; 
          if (counts[g] > max) { max = counts[g]; stable = g; } 
        });
        setGesture(stable);

        if (stable === stableGesture.current) {
          stableCount.current++;
          if (stableCount.current > STABLE_THRESHOLD && stable.length === 1) {
            if (lastWrittenLetter.current !== stable) {
              setText(t => t + stable);
              lastWrittenLetter.current = stable;
            }
          }
        } else {
          stableGesture.current = stable;
          stableCount.current = 0;
        }
      }
    }

    // --- YOLO ---
    const input = tf.tidy(() => tf.browser.fromPixels(video).resizeBilinear([INPUT_SIZE, INPUT_SIZE]).div(255.0).expandDims(0));
    let res = await yoloModel.executeAsync(input);
    if (Array.isArray(res)) res = res[0];
    let transRes = res.shape[1] < res.shape[2] ? res.transpose([0, 2, 1]) : res;
    const data = await transRes.data();
    tf.dispose([res, transRes, input]);

    let maxScore = 0; let bestIdx = -1;
    const numChannels = transRes.shape[2]; 

    for (let i = 0; i < transRes.shape[1]; i++) { 
      let score = data[i * numChannels + 4];
      if (score > 1.0) score = 1.0 / (1.0 + Math.exp(-score));
      if (score > CONFIDENCE_THRESHOLD && score > maxScore) { 
        maxScore = score; 
        bestIdx = i * numChannels; 
      }
    }

    let liveDetection = null;
    if (bestIdx >= 0) {
      wasHandPresent.current = true;
      const scaleX = dim.w / INPUT_SIZE; 
      const scaleY = dim.h / INPUT_SIZE;
      
      let bx = data[bestIdx], by = data[bestIdx + 1], bw = data[bestIdx + 2], bh = data[bestIdx + 3];
      if (bx < 2.0) { bx *= INPUT_SIZE; by *= INPUT_SIZE; bw *= INPUT_SIZE; bh *= INPUT_SIZE; }

      const boxW = bw * scaleX; 
      const boxH = bh * scaleY;
      const boxX = (dim.w - (bx * scaleX)) - (boxW / 2);
      const boxY = (by * scaleY) - (boxH / 2);
      
      liveDetection = { score: maxScore, box: [boxX, boxY, boxW, boxH] };
      if (Date.now() - lastBoxUpdate.current > 500 || !frozenBox.current) {
        frozenBox.current = liveDetection.box;
        lastBoxUpdate.current = Date.now();
      }
    } else if (wasHandPresent.current) {
      wasHandPresent.current = false;
      frozenBox.current = null;
    }
    
    const ctx = canvasRef.current.getContext("2d");
    drawOnCanvas(ctx, liveDetection, frozenBox.current, dim);

  }, [yoloModel, mpLandmarker, dim, recognizeGesture]);

  useEffect(() => {
    if (!yoloModel || !mpLandmarker) return;
    const interval = setInterval(detect, 50);
    return () => clearInterval(interval);
  }, [yoloModel, mpLandmarker, detect]);

  // --- 4. CONTROLS ---
  useEffect(() => {
    if (editingIndex !== null && history[editingIndex]) {
      if (history[editingIndex].text !== text) {
        const updatedHistory = [...history];
        updatedHistory[editingIndex] = { ...updatedHistory[editingIndex], text: text };
        setHistory(updatedHistory);
        setCookie('signeticHistory', updatedHistory, 7);
      }
    }
  }, [text, editingIndex, history]);

  const logNewEntry = (content) => {
    if (!content.trim()) return;
    const newItem = { 
      text: content, 
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
    };
    const newHistory = [newItem, ...history].slice(0, 50);
    setHistory(newHistory);
    setCookie('signeticHistory', newHistory, 7);
  };

  const deleteHistoryItem = (e, index) => {
    e.stopPropagation(); // Stop the click from opening the item
    const newHistory = history.filter((_, i) => i !== index);
    setHistory(newHistory);
    setCookie('signeticHistory', newHistory, 7);
    
    // If we deleted the item currently being edited, stop editing
    if (editingIndex === index) {
      setEditingIndex(null);
      setText("");
    } else if (editingIndex !== null && index < editingIndex) {
      // Adjust index if we deleted something above it
      setEditingIndex(prev => prev - 1);
    }
  };

  const handleSpace = () => setText(t => t + " ");
  const handleBackspace = () => setText(t => t.slice(0, -1));
  
  const handleAction = () => {
    if (editingIndex !== null) {
      if (history[editingIndex]) {
        const updatedHistory = [...history];
        updatedHistory[editingIndex] = { ...updatedHistory[editingIndex], text: text };
        setHistory(updatedHistory);
        setCookie('signeticHistory', updatedHistory, 7);
        setText("");
      }
      setEditingIndex(null);
    } else {
      if (text) logNewEntry(text);
      setText("");
    }
  };
  
  const handleSpeak = () => {
    if (!text) return;
    if (editingIndex === null) logNewEntry(text);
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  };

  const handleHistoryClick = (idx) => { 
    setText(history[idx].text); 
    setEditingIndex(idx); 
    if (window.innerWidth < 900) setIsSidebarOpen(false);
  };

  return (
    <div className="app-container">
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <span>📜 History</span>
          <button className="close-btn" onClick={() => setIsSidebarOpen(false)}>×</button>
        </div>
        <div className="history-list">
          {history.length === 0 && <div style={{color:'var(--text-secondary)', textAlign:'center', marginTop:20}}>No history yet</div>}
          {history.map((item, idx) => (
            <div key={idx} className={`history-item ${editingIndex === idx ? 'active' : ''}`} onClick={() => handleHistoryClick(idx)}>
              <div style={{flex: 1}}>
                <div className="item-text">{item.text || "Empty"}</div>
                <span className="item-time">{item.time}</span>
              </div>
              {/* DELETE BUTTON */}
              <button 
                className="delete-item-btn"
                onClick={(e) => deleteHistoryItem(e, idx)}
                title="Delete item"
              >
                🗑️
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}>
        <header className="header">
          <div style={{display:'flex', gap:'12px', alignItems:'center'}}>
            <button className="sidebar-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
              {isSidebarOpen ? '◀' : '📜'}
            </button>
            <div className="brand-tile">
              <img src="/logo.jpg" alt="Signetic Logo" className="logo" />
              <h1 className="app-title">Signetic</h1>
            </div>
          </div>
          <button className="sidebar-toggle" onClick={toggleTheme} title="Toggle Theme">
             {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </header>

        <div className="display-area">
          <div className="camera-wrapper" style={{ width: dim.w, height: dim.h }}>
            <div className="gesture-status">
              {gesture}
            </div>
            <Webcam ref={webcamRef} className="webcam-video" width={dim.w} height={dim.h} mirrored={true} />
            <canvas ref={canvasRef} className="drawing-canvas" width={dim.w} height={dim.h} />
          </div>
          
          <div className={`text-output-container ${editingIndex !== null ? 'editing' : ''}`} style={{width:'100%'}}>
            <div className={`text-output ${!text ? 'placeholder' : ''}`}>
              {text || "Translation will appear here..."}
            </div>
            {editingIndex !== null && <small style={{color:'var(--accent-color)', fontWeight:'bold', display:'block', marginTop:5}}>✏️ Editing History Item...</small>}
          </div>
          
          <div className="controls">
            <button className="btn btn-primary" onClick={handleSpace}>Space</button>
            <button className="btn btn-warning" onClick={handleBackspace}>Backspace</button>
            <button className="btn btn-speak" onClick={handleSpeak}>Speak 🗣️</button>
            
            {editingIndex !== null ? (
               <button className="btn" style={{backgroundColor:'#10b981', color:'white', borderColor:'#10b981'}} onClick={handleAction}>
                 Done & Clear✅
               </button>
            ) : (
               <button className="btn btn-danger" onClick={handleAction}>
                 Clear & Save🗑️
               </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}