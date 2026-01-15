import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { drawDetection } from "./drawUtil";

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [gesture, setGesture] = useState("⏳ Loading...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 480, h: 360 });

  // --- CONFIGURATION ---
  const MODEL_PATH = 'best_web_model/model.json';
  const INPUT_SIZE = 320; 
  const CONFIDENCE_THRESHOLD = 0.80;

  // --- REFS ---
  const wasHandPresent = useRef(false);
  const gestureBuffer = useRef([]); 
  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const STABLE_THRESHOLD = 10; 

  // --- STATIC BOX REFS ---
  const frozenBox = useRef(null);
  const lastBoxUpdate = useRef(0);
  const BOX_REFRESH_RATE = 2000; 

  // --- 1. LOAD MODEL ---
  useEffect(() => {
    const init = async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      const m = await tf.loadGraphModel(MODEL_PATH);
      const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
      await m.executeAsync(dummy);
      dummy.dispose();
      setModel(m);
      setGesture("👀 Show Hand");
    };
    init();
  }, []);

  // --- 2. GESTURE LOGIC (A-Z) ---
  // Note: This logic assumes a right hand in standard orientation.
  // We will pass it un-mirrored data so the logic holds up.
  const recognizeGesture = useCallback((keypoints) => {
    if (!keypoints || keypoints.length < 21) return "";
    const k = keypoints;
    const d = (i1, i2) => Math.hypot(k[i1].x - k[i2].x, k[i1].y - k[i2].y);
    const handSize = d(0, 9); 
    const T = (factor) => handSize * factor; 

    const thumbOpen = d(4, 17) > T(1.1); 
    const indexOpen = d(8, 0) > T(1.2) && d(8, 5) > T(0.6);
    const middleOpen = d(12, 0) > T(1.2) && d(12, 9) > T(0.6);
    const ringOpen = d(16, 0) > T(1.2) && d(16, 13) > T(0.6);
    const pinkyOpen = d(20, 0) > T(1.1) && d(20, 17) > T(0.6);
    const fingersCount = (indexOpen ? 1 : 0) + (middleOpen ? 1 : 0) + (ringOpen ? 1 : 0) + (pinkyOpen ? 1 : 0);
    const indexCurl = d(8, 0) < T(1.0); 

    // 1. NO FINGERS UP -> A, E, M, N, S, T
    if (fingersCount === 0) {
        if (d(4, 10) < T(0.4) || d(4, 14) < T(0.4)) return "S";
        if (d(4, 5) < T(0.4) && k[4].y < k[5].y) return "A";
        if (d(8, 0) < T(0.8) && d(4, 13) < T(0.5)) return "E";
        if (d(4, 14) < T(0.3) && d(4, 18) < T(0.3)) return "M";
        if (d(4, 10) < T(0.3) && d(4, 14) < T(0.3)) return "N";
        if (d(4, 5) < T(0.4) && d(4, 9) < T(0.4)) return "T";
        return "E";
    }
    // 2. ONE FINGER UP -> D, L, X, Z, P, Q
    if (fingersCount === 1 && indexOpen) {
        if (thumbOpen) return "L";
        if (d(8, 6) < T(0.4)) return "X";
        if (k[8].y > k[5].y) return "Q"; 
        return "D";
    }
    // 3. ONE FINGER UP (Pinky) -> I, Y
    if (fingersCount === 1 && pinkyOpen) {
        if (d(4, 17) > T(1.1)) return "Y";
        return "I";
    }
    // 4. TWO FINGERS UP -> H, K, R, U, V, P
    if (fingersCount === 2 && indexOpen && middleOpen) {
        if (k[8].y > k[5].y) return "P";
        const isHorizontal = Math.abs(k[8].x - k[0].x) > Math.abs(k[8].y - k[0].y);
        if (isHorizontal) return "H";
        if (d(8, 12) < T(0.3)) return "R";
        if (d(8, 12) < T(0.5)) return "U";
        if (d(8, 12) > T(0.6)) return "V";
        if (k[4].y < k[5].y && k[4].y > k[8].y) return "K";
        return "V";
    }
    // 5. THREE FINGERS UP -> W, F
    if (indexOpen && middleOpen && ringOpen && !pinkyOpen) {
        return "W";
    }
    // 6. F / OK SIGN CHECK
    if (d(4, 8) < T(0.5) && middleOpen && ringOpen && pinkyOpen) {
        return "F";
    }
    // 7. O / C CHECK
    if (!indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
       if (d(8, 0) > T(0.9)) { 
           if (d(4, 8) < T(0.4)) return "O";
           return "C";
       }
    }
    // 8. FOUR/FIVE FINGERS -> B, 5
    if (indexOpen && middleOpen && ringOpen && pinkyOpen) {
        if (d(4, 9) < T(0.6)) return "B";
        return "🖐️"; 
    }
    // 9. G
    if (!middleOpen && !ringOpen && !pinkyOpen) {
        if (Math.abs(k[8].x - k[0].x) > Math.abs(k[8].y - k[0].y)) return "G";
    }
    return "";
  }, []);

  // --- 3. DETECTION LOOP ---
  const detect = useCallback(async () => {
    if (!model || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    // A. PREPARE INPUT
    const input = tf.tidy(() => {
        return tf.browser.fromPixels(video)
          .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
          .div(255.0)
          .expandDims(0);
    });

    // B. INFERENCE & SHAPE HANDLING
    let res = await model.executeAsync(input);
    if (Array.isArray(res)) res = res[0];
    const shape = res.shape;
    let transRes, numChannels, numAnchors;
    if (shape[1] < shape[2]) {
        transRes = res.transpose([0, 2, 1]); 
        numChannels = shape[1]; numAnchors = shape[2];
    } else {
        transRes = res; 
        numChannels = shape[2]; numAnchors = shape[1];
    }
    const data = await transRes.data();
    if(shape[1] < shape[2]) res.dispose();
    transRes.dispose();
    input.dispose();

    // C. FIND BEST DETECTION
    let maxScore = 0;
    let bestRaw = null;
    for (let i = 0; i < numAnchors; i++) {
        const offset = i * numChannels;
        const score = data[offset + 4];
        if (score > CONFIDENCE_THRESHOLD && score > maxScore) {
            maxScore = score;
            bestRaw = { offset, score };
        }
    }

    // D. PROCESS RESULT
    let liveDetection = null;

    if (bestRaw) {
        wasHandPresent.current = true;
        const scaleX = dim.w / INPUT_SIZE;
        const scaleY = dim.h / INPUT_SIZE;
        const { offset, score } = bestRaw;

        // --- 1. Calculate Raw (Unmirrored) Data ---
        // We use this for gesture recognition so right/left logic remains correct relative to the hand itself.
        const rawKeypoints = [];
        for (let k = 0; k < 21; k++) {
            const kIdx = offset + 5 + (k * 3);
            rawKeypoints.push({
                x: data[kIdx] * scaleX,
                y: data[kIdx + 1] * scaleY,
                score: data[kIdx + 2]
            });
        }

        const rawBx = data[offset];     // center x
        const rawBy = data[offset + 1]; // center y
        const rawBw = data[offset + 2]; // width
        const rawBh = data[offset + 3]; // height
        
        // Calculate scaled box dimensions
        const boxW = rawBw * scaleX;
        const boxH = rawBh * scaleY;
        // Calculate unmirrored top-left X and Y
        const boxX_unmirrored = (rawBx * scaleX) - (boxW / 2);
        const boxY = (rawBy * scaleY) - (boxH / 2);

        // --- 2. Create Mirrored Data for Drawing ---
        // === MIRRORING ADJUSTMENT START ===
        // Since the webcam view is mirrored, we flip X coordinates for drawing.
        
        // Flip Keypoints
        const drawingKeypoints = rawKeypoints.map(kp => ({
            ...kp,
            x: dim.w - kp.x // Flip X across canvas width
        }));

        // Flip Box X coordinate: NewX = CanvasWidth - OriginalX - BoxWidth
        const boxX_mirrored = dim.w - boxX_unmirrored - boxW;
        const drawingBox = [ boxX_mirrored, boxY, boxW, boxH ];
        // === MIRRORING ADJUSTMENT END ===

        // Prepare object for drawing utility using MIRRORED data
        liveDetection = { keypoints: drawingKeypoints, score, box: drawingBox };

        // --- 3. STATIC BOX LOGIC ---
        const now = Date.now();
        if (now - lastBoxUpdate.current > BOX_REFRESH_RATE || !frozenBox.current) {
            // Use the MIRRORED box for the static drawing
            frozenBox.current = drawingBox;
            lastBoxUpdate.current = now;
        }

        // --- 4. RECOGNIZE GESTURE (Use RAW/UNMIRRORED data) ---
        const rawGesture = recognizeGesture(rawKeypoints);
        
        // --- 5. STABILIZATION & TYPING ---
        if (rawGesture) {
            gestureBuffer.current.push(rawGesture);
            if (gestureBuffer.current.length > 5) gestureBuffer.current.shift();
            const counts = {}; let maxCount = 0; let stableG = rawGesture;
            gestureBuffer.current.forEach(g => {
                counts[g] = (counts[g] || 0) + 1;
                if(counts[g] > maxCount) { maxCount = counts[g]; stableG = g; }
            });
            setGesture(stableG);

            if (stableG === stableGesture.current) stableCount.current += 1;
            else { stableGesture.current = stableG; stableCount.current = 0; }

            if (stableCount.current > STABLE_THRESHOLD && stableG !== "🖐️" && stableG.length === 1) {
                if (lastWrittenLetter.current !== stableG) {
                    setText(prev => prev + stableG);
                    lastWrittenLetter.current = stableG;
                }
            }
        }
    } else {
        if (wasHandPresent.current) {
            wasHandPresent.current = false;
            frozenBox.current = null;
            setGesture("👀 Show Hand");
        }
    }

    // E. DRAW (Pass mirrored data)
    const ctx = canvasRef.current.getContext("2d");
    drawDetection(ctx, liveDetection, frozenBox.current, dim);

  }, [model, dim, recognizeGesture]);

  // --- 4. LOOP ---
  useEffect(() => {
    if(!model) return;
    let raf;
    const loop = () => { detect(); raf = requestAnimationFrame(loop); };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [model, detect]);

  // --- KEYBOARD HANDLERS ---
  const handleSpace = () => setText(t => t + " ");
  const handleClear = () => setText("");
  const handleBackspace = () => setText(t => t.slice(0, -1));

  return (
    <div style={{ background: "#ede7e7", minHeight: "100vh", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center" }}>
      <h1 style={{ color: "#333", marginBottom: "10px" }}>Sign Language Detector</h1>
      <h2 style={{ fontSize: "3rem", color: "#2b9308", margin: "10px 0" }}>{gesture}</h2>
      <div style={{ width: "90%", maxWidth: "500px", padding: "15px", background: "white", borderRadius: "8px", border: "2px solid #ccc", minHeight: "60px", fontSize: "1.5rem", marginBottom: "20px", display:"flex", alignItems:"center", justifyContent:"center" }}>
        {text || <span style={{color: "#ccc"}}>Text will appear here...</span>}
      </div>
      <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
        <button onClick={handleSpace} style={btnStyle}>SPACE</button>
        <button onClick={handleBackspace} style={{...btnStyle, background:"orange"}}>BACK</button>
        <button onClick={handleClear} style={{...btnStyle, background:"red"}}>CLEAR</button>
      </div>
      {/* Ensure Webcam is mirrored AND Canvas is exactly on top */}
      <div style={{ position: "relative", width: dim.w, height: dim.h, borderRadius: "12px", overflow: "hidden", border: "4px solid #333", boxShadow: "0 10px 20px rgba(0,0,0,0.3)" }}>
        <Webcam 
            ref={webcamRef} 
            width={dim.w} 
            height={dim.h} 
            mirrored={true} // Visual mirroring
            style={{ width: "100%", height: "100%", objectFit: "cover" }} 
        />
        <canvas 
            ref={canvasRef} 
            width={dim.w} 
            height={dim.h} 
            style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }} // Canvas on top
        />
      </div>
    </div>
  );
}

const btnStyle = { padding: "10px 20px", borderRadius: "6px", border: "none", background: "#333", color: "white", fontWeight: "bold", cursor: "pointer", fontSize: "1rem" };