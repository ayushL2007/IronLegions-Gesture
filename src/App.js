import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { drawDetection } from "./drawUtil";

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  // --- STATE ---
  const [yoloModel, setYoloModel] = useState(null); // For visuals
  const [mpLandmarker, setMpLandmarker] = useState(null); // For logic
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

  // --- 2. MEDIAPIPE LOGIC (High Performance 3D Math) ---
const recognizeGestureMP = useCallback((mpLandmarks) => {
    if (!mpLandmarks || mpLandmarks.length < 21) return "";
    const k = mpLandmarks;

    // 1. 3D Distance Helper (Depth-aware)
    const d = (i1, i2) => Math.hypot(
       k[i1].x - k[i2].x, 
       k[i1].y - k[i2].y, 
       (k[i1].z - k[i2].z) * 2
    );

    // 2. Reference Measurements
    const handSize = d(0, 9); // Wrist to Middle Knuckle
    
    // 3. Finger States
    // Tip must be significantly higher/further than knuckle to be "Open"
    const isExtended = (tip, knuckle) => d(0, tip) > d(0, knuckle) * 1.1;

    const indexOpen  = isExtended(8, 5);
    const middleOpen = isExtended(12, 9);
    const ringOpen   = isExtended(16, 13);
    const pinkyOpen  = isExtended(20, 17);

    // Thumb is special: Check distance to Pinky Knuckle (Reference point across palm)
    // If far, thumb is out. If close, thumb is in.
    const thumbOpen  = d(4, 17) > handSize * 0.5; 

    // Count non-thumb fingers (Index to Pinky)
    let count = [indexOpen, middleOpen, ringOpen, pinkyOpen].filter(Boolean).length;
    
    // Add Thumb to count if it's explicitly extended (like in '5' or 'L')
    const totalCount = count + (thumbOpen ? 1 : 0);

    // --- LOGIC TREE ---

    // 

    // === 5 FINGERS UP ===
    if (totalCount === 5) {
        // Fingers splayed? -> 5. Fingers together? -> B.
        // Simple check: Distance between Index and Pinky tips
        if (d(8, 20) < handSize * 0.6) return "B"; 
        return "🖐️"; // 5 or Open Palm
    }

    // === 4 FINGERS UP ===
    if (totalCount === 4) {
        // If thumb is tucked (count=4), it's B
        if (!thumbOpen) return "B"; 
    }

    // === 3 FINGERS UP ===
    if (count === 3) {
        // W: Index, Middle, Ring are up. Pinky down.
        if (indexOpen && middleOpen && ringOpen && !pinkyOpen) return "W";
        
        // F: Middle, Ring, Pinky up. Index+Thumb touching (OK sign).
        if (!indexOpen && d(4, 8) < handSize * 0.3) return "F";
    }

    // === 2 FINGERS UP ===
    if (count === 2) {
        // V / R / U: Index + Middle
        if (indexOpen && middleOpen) {
            // R: Crossed fingers (Index X > Middle X) (Assumes Right Hand/Mirror)
            if (k[8].x > k[12].x) return "R"; 
            
            // U: Fingers held very close together
            if (d(8, 12) < handSize * 0.25) return "U";

            return "V";
        }
        
        // Y: Thumb + Pinky (Hang loose)
        if (thumbOpen && pinkyOpen) return "Y";
        
        // ILY (I Love You): Thumb + Index + Pinky
        if (thumbOpen && indexOpen && pinkyOpen) return "🤟"; 
    }

    // === 1 FINGER UP ===
    if (count === 1) {
        // L: Index + Thumb
        if (indexOpen && thumbOpen) return "L";

        // D: Index only (Thumb touches middle finger)
        if (indexOpen) return "D";

        // I: Pinky only
        if (pinkyOpen) return "I";
    }

    // === 0 FINGERS (FISTS & CURVES) ===
    if (count === 0) {
        const thumbTip = k[4];
        const indexTip = k[8];
        const indexKnuckle = k[5];
        const middleKnuckle = k[9];

        // O: Thumb tip touches Index tip (making a hole)
        if (d(4, 8) < handSize * 0.3) return "O";

        // C: Hand forms a C (Thumb and fingers curved but not touching)
        // Check if Thumb is "under" the Index finger vertically
        if (d(4, 8) < handSize * 0.9 && d(4, 8) > handSize * 0.4) return "C";

        // --- FIST VARIANTS (A, S, E, M, N, T) ---
        // These depend on WHERE the thumb is crossing the fingers.
        
        // E: Thumb is curled low, touching tips of fingers (implied by 0 count)
        // Hard to distinguish from S, but usually thumb tip is lower
        if (thumbTip.y > indexKnuckle.y) return "E"; // Thumb is low

        // A: Thumb is sticking out to the side (flush with palm)
        // Check horizontal distance of thumb tip from index knuckle
        if (Math.abs(thumbTip.x - indexKnuckle.x) > handSize * 0.2) return "A";

        // T: Thumb tucked between Index and Middle
        // Thumb X is between Index Knuckle and Middle Knuckle
        if (thumbTip.x > indexKnuckle.x && thumbTip.x < middleKnuckle.x) return "T";

        // S: Thumb wrapped over fingers (Default Fist)
        return "S";
    }

    return "";
  }, []);

  // --- 3. THE HYBRID LOOP ---
  const detect = useCallback(async () => {
    if (!yoloModel || !mpLandmarker || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    // ---------------------------------------------------------
    // STEP A: RUN MEDIAPIPE (For the Brains/Text)
    // ---------------------------------------------------------
    const mpResult = mpLandmarker.detectForVideo(video, Date.now());
    if (mpResult.landmarks && mpResult.landmarks.length > 0) {
        // We use these High-Quality 3D points for the logic
        const mpKeypoints = mpResult.landmarks[0]; 
        const resultGesture = recognizeGestureMP(mpKeypoints);

        // Update Text Buffer (Debouncing)
        if (resultGesture) {
            gestureBuffer.current.push(resultGesture);
            if (gestureBuffer.current.length > 5) gestureBuffer.current.shift();
            
            // Simple frequency check
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

    // ---------------------------------------------------------
    // STEP B: RUN CUSTOM YOLO (For the Visuals)
    // ---------------------------------------------------------
    const input = tf.tidy(() => {
        return tf.browser.fromPixels(video)
          .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
          .div(255.0)
          .expandDims(0);
    });

    let res = await yoloModel.executeAsync(input);
    if (Array.isArray(res)) res = res[0]; // Handle array output

    // Parse YOLO Output (Transpose if needed)
    let transRes;
    const shape = res.shape;
    if (shape[1] < shape[2]) {
        transRes = res.transpose([0, 2, 1]); // [1, 8400, 56]
    } else {
        transRes = res;
    }

    const data = await transRes.data();
    tf.dispose([res, transRes, input]); // Cleanup Tensors

    // Find Best Box (NMS style logic)
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

        // Extract YOLO Keypoints
        const rawKeypoints = [];
        for (let k = 0; k < 21; k++) {
            const x = data[bestIdx + 5 + (k * 3)] * scaleX;
            const y = data[bestIdx + 5 + (k * 3) + 1] * scaleY;
            rawKeypoints.push({ x: dim.w - x, y: y }); // Mirror X
        }

        // Extract YOLO Box
        const bx = data[bestIdx];
        const by = data[bestIdx + 1];
        const bw = data[bestIdx + 2];
        const bh = data[bestIdx + 3];

        const boxW = bw * scaleX;
        const boxH = bh * scaleY;
        const boxX = (dim.w - (bx * scaleX)) - (boxW / 2); // Mirror Box X
        const boxY = (by * scaleY) - (boxH / 2);

        // Stabilize Box (Your Custom Logic)
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

    // Draw using CUSTOM MODEL data
    const ctx = canvasRef.current.getContext("2d");
    drawDetection(ctx, liveDetection, frozenBox.current, dim);

  }, [yoloModel, mpLandmarker, dim, recognizeGestureMP]);

  // --- LOOP ---
  useEffect(() => {
    if (!yoloModel || !mpLandmarker) return;
    const interval = setInterval(detect, 50); // Cap at 20fps to save CPU
    return () => clearInterval(interval);
  }, [yoloModel, mpLandmarker, detect]);

  // --- UI RENDER ---
  return (
    <div style={{ background: "#ede7e7", minHeight: "100vh", padding: "20px", display: "flex", flexDirection: "column", alignItems: "center", fontFamily: "sans-serif" }}>
      <h1>ASL Detector (Hybrid Engine)</h1>
      <h2 style={{ fontSize: "3rem", color: "#2b9308" }}>{gesture}</h2>
      
      <div style={{ width: "90%", padding: "15px", background: "white", fontSize: "1.5rem", marginBottom: "20px", border: "1px solid #ccc" }}>
        {text || "Start signing..."}
      </div>

      <div style={{ position: "relative", width: dim.w, height: dim.h, border: "4px solid #333", borderRadius: "10px", overflow: "hidden" }}>
        <Webcam ref={webcamRef} width={dim.w} height={dim.h} mirrored={true} style={{ width: "100%", height: "100%", objectFit:"cover" }} />
        <canvas ref={canvasRef} width={dim.w} height={dim.h} style={{ position: "absolute", top: 0, left: 0 }} />
      </div>

       <div style={{marginTop: "20px"}}>
         <button onClick={() => setText(t => t + " ")}>SPACE</button>
         <button onClick={() => setText("")} style={{marginLeft: "10px"}}>CLEAR</button>
      </div>
    </div>
  );
}