import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { drawDetection } from "./drawUtil";
import "./App.css"; 

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  // --- STATE ---
  const [yoloModel, setYoloModel] = useState(null); // Visuals
  const [mpLandmarker, setMpLandmarker] = useState(null); // Logic
  const [gesture, setGesture] = useState("⏳ Loading Models...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 480, h: 360 });

  // --- CONFIG ---
  const YOLO_PATH = 'best_web_model/model.json';
  const INPUT_SIZE = 320;
  const CONFIDENCE_THRESHOLD = 0.75;
  const STABLE_THRESHOLD = 10;

  // --- REFS ---
  const gestureBuffer = useRef([]);
  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  
  // Visual Refs (YOLO)
  const frozenBox = useRef(null);
  const lastBoxUpdate = useRef(0);
  const wasHandPresent = useRef(false);

  // --- 1. LOAD BOTH MODELS ---
  useEffect(() => {
    const loadModels = async () => {
      // A. Load Custom YOLO (Visuals)
      await tf.setBackend('webgl');
      await tf.ready();
      const loadedYolo = await tf.loadGraphModel(YOLO_PATH);
      
      // Warmup YOLO
      const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
      await loadedYolo.executeAsync(dummy);
      dummy.dispose();
      setYoloModel(loadedYolo);

      // B. Load MediaPipe (Logic)
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
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
    };
    loadModels();
  }, []);

  // --- 2. MEDIAPIPE LOGIC (Complete Alphabet Support) ---
  const recognizeGestureMP = useCallback((mpLandmarks) => {
    if (!mpLandmarks || mpLandmarks.length < 21) return "";
    const k = mpLandmarks;

    // 1. 3D Distance Helper (Depth-aware)
    const d = (i1, i2) => Math.hypot(
       k[i1].x - k[i2].x, 
       k[i1].y - k[i2].y, 
       (k[i1].z - k[i2].z) * 2
    );

    const handSize = d(0, 9); 
    const isExtended = (tip, knuckle) => d(0, tip) > d(0, knuckle) * 1.1;

    const indexOpen  = isExtended(8, 5);
    const middleOpen = isExtended(12, 9);
    const ringOpen   = isExtended(16, 13);
    const pinkyOpen  = isExtended(20, 17);
    const thumbOpen  = d(4, 17) > handSize * 0.5; 

    let count = [indexOpen, middleOpen, ringOpen, pinkyOpen].filter(Boolean).length;
    const totalCount = count + (thumbOpen ? 1 : 0);

    // --- LOGIC TREE ---
    if (totalCount === 5) {
        if (d(8, 20) < handSize * 0.6) return "B"; 
        return "🖐️"; 
    }
    if (totalCount === 4) {
        if (!thumbOpen) return "B"; 
    }
    if (count === 3) {
        if (indexOpen && middleOpen && ringOpen && !pinkyOpen) return "W";
        if (!indexOpen && d(4, 8) < handSize * 0.3) return "F";
    }
    if (count === 2) {
        if (indexOpen && middleOpen) {
            if (k[8].x > k[12].x) return "R"; 
            if (d(8, 12) < handSize * 0.25) return "U";
            return "V";
        }
        if (thumbOpen && pinkyOpen) return "Y";
        if (thumbOpen && indexOpen && pinkyOpen) return "🤟"; 
    }
    if (count === 1) {
        if (indexOpen && thumbOpen) return "L";
        if (indexOpen) return "D";
        if (pinkyOpen) return "I";
    }
    if (count === 0) {
        const thumbTip = k[4];
        const indexKnuckle = k[5];
        const middleKnuckle = k[9];

        if (d(4, 8) < handSize * 0.3) return "O";
        if (d(4, 8) < handSize * 0.9 && d(4, 8) > handSize * 0.4) return "C";
        
        if (thumbTip.y > indexKnuckle.y) return "E"; 
        if (Math.abs(thumbTip.x - indexKnuckle.x) > handSize * 0.2) return "A";
        if (thumbTip.x > indexKnuckle.x && thumbTip.x < middleKnuckle.x) return "T";
        return "S";
    }
    return "";
  }, []);

  // --- 3. THE HYBRID LOOP ---
  const detect = useCallback(async () => {
    if (!yoloModel || !mpLandmarker || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    // STEP A: RUN MEDIAPIPE (Logic)
    const mpResult = mpLandmarker.detectForVideo(video, Date.now());
    if (mpResult.landmarks && mpResult.landmarks.length > 0) {
        const mpKeypoints = mpResult.landmarks[0]; 
        const resultGesture = recognizeGestureMP(mpKeypoints);

        if (resultGesture) {
            gestureBuffer.current.push(resultGesture);
            if (gestureBuffer.current.length > 5) gestureBuffer.current.shift();
            
            const counts = {}; let max = 0; let stable = resultGesture;
            gestureBuffer.current.forEach(g => { counts[g]=(counts[g]||0)+1; if(counts[g]>max){max=counts[g]; stable=g;} });
            
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

    // STEP B: RUN CUSTOM YOLO (Visuals)
    const input = tf.tidy(() => {
        return tf.browser.fromPixels(video)
          .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
          .div(255.0)
          .expandDims(0);
    });

    let res = await yoloModel.executeAsync(input);
    if (Array.isArray(res)) res = res[0]; 

    let transRes;
    const shape = res.shape;
    if (shape[1] < shape[2]) {
        transRes = res.transpose([0, 2, 1]); 
    } else {
        transRes = res;
    }

    const data = await transRes.data();
    tf.dispose([res, transRes, input]); 

    let maxScore = 0;
    let bestIdx = -1;
    const numAnchors = shape[1] < shape[2] ? shape[2] : shape[1];
    const numChannels = shape[1] < shape[2] ? shape[1] : shape[2];

    for (let i = 0; i < numAnchors; i++) {
        const score = data[i * numChannels + 4];
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

        const rawKeypoints = [];
        for (let k = 0; k < 21; k++) {
            const x = data[bestIdx + 5 + (k * 3)] * scaleX;
            const y = data[bestIdx + 5 + (k * 3) + 1] * scaleY;
            rawKeypoints.push({ x: dim.w - x, y: y }); 
        }

        const bx = data[bestIdx];
        const by = data[bestIdx + 1];
        const bw = data[bestIdx + 2];
        const bh = data[bestIdx + 3];

        const boxW = bw * scaleX;
        const boxH = bh * scaleY;
        const boxX = (dim.w - (bx * scaleX)) - (boxW / 2); 
        const boxY = (by * scaleY) - (boxH / 2);

        const currentBox = [boxX, boxY, boxW, boxH];
        liveDetection = { keypoints: rawKeypoints, score: maxScore, box: currentBox };

        const now = Date.now();
        if (now - lastBoxUpdate.current > 500 || !frozenBox.current) {
            frozenBox.current = currentBox;
            lastBoxUpdate.current = now;
        }
    } else {
        if (wasHandPresent.current) {
             wasHandPresent.current = false;
             frozenBox.current = null;
        }
    }

    const ctx = canvasRef.current.getContext("2d");
    drawDetection(ctx, liveDetection, frozenBox.current, dim);

  }, [yoloModel, mpLandmarker, dim, recognizeGestureMP]);

  useEffect(() => {
    if (!yoloModel || !mpLandmarker) return;
    const interval = setInterval(detect, 50); 
    return () => clearInterval(interval);
  }, [yoloModel, mpLandmarker, detect]);

  // --- BUTTON HANDLERS ---
  const handleSpace = () => setText(t => t + " ");
  const handleBackspace = () => setText(t => t.slice(0, -1)); 
  const handleClear = () => setText("");
  
  // NEW: TEXT TO SPEECH HANDLER
  const handleSpeak = () => {
    if (!text) return;
    // Cancel any ongoing speech to prevent overlapping
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    // Optional: Select a specific voice if desired, usually default is fine
    utterance.rate = 0.9; // Slightly slower for clarity
    window.speechSynthesis.speak(utterance);
  };

  // --- UI RENDER ---
  return (
    <div className="app-container">
      
      {/* HEADER */}
      <header className="header">
        <img src="/logo.jpg" alt="Signetic Logo" className="logo" />
        <h1 className="app-title">Signetic</h1>
      </header>

      {/* MAIN CONTENT */}
      <div className="display-area">
        
        {/* GESTURE INDICATOR */}
        <h2 className="gesture-status">
          {gesture}
        </h2>
        
        {/* TEXT OUTPUT */}
        <div className={`text-output ${!text ? 'placeholder' : ''}`}>
          {text || "Start signing to translate..."}
        </div>

        {/* CAMERA FEED */}
        <div className="camera-wrapper" style={{ width: dim.w, height: dim.h }}>
          <Webcam 
            ref={webcamRef} 
            className="webcam-video"
            width={dim.w} 
            height={dim.h} 
            mirrored={true} 
          />
          <canvas 
            ref={canvasRef} 
            className="drawing-canvas"
            width={dim.w} 
            height={dim.h} 
          />
        </div>

        {/* CONTROLS */}
        <div className="controls">
          <button className="btn btn-primary" onClick={handleSpace}>
            Space
          </button>
          
          <button className="btn btn-warning" onClick={handleBackspace}>
            Backspace
          </button>

          <button className="btn btn-speak" onClick={handleSpeak}>
             Speak 🗣️
          </button>

          <button className="btn btn-danger" onClick={handleClear}>
            Clear
          </button>
        </div>

      </div>
    </div>
  );
}