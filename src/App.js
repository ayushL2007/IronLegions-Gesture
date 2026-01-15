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
  const [yoloModel, setYoloModel] = useState(null);
  const [mpLandmarker, setMpLandmarker] = useState(null);
  const [gesture, setGesture] = useState("⏳ Loading Models...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 480, h: 360 });

  // --- CONFIG ---
  const YOLO_PATH = 'best_web_model/model.json';
  const INPUT_SIZE = 320;
  const CONFIDENCE_THRESHOLD = 0.85;
  const STABLE_THRESHOLD = 10;

  // --- REFS ---
  const gestureBuffer = useRef([]);
  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const frozenBox = useRef(null);
  const lastBoxUpdate = useRef(0);
  const wasHandPresent = useRef(false);

  // --- 1. LOAD BOTH MODELS ---
  useEffect(() => {
    const loadModels = async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      const loadedYolo = await tf.loadGraphModel(YOLO_PATH);
      const dummy = tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3]);
      await loadedYolo.executeAsync(dummy);
      dummy.dispose();
      setYoloModel(loadedYolo);

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

  // --- 2. NEW RECOGNITION LOGIC ---
  const recognizeGesture = useCallback((hand) => {
    const k = hand.keypoints; // Accessing keypoints from the passed object
    if (!k || k.length < 21) return "";

    const handSize = Math.hypot(k[9].x - k[0].x, k[9].y - k[0].y);
    const T = (factor) => handSize * factor;

    const isExtended = (tipIdx, mcpIdx) => {
      const dist = Math.hypot(k[tipIdx].x - k[mcpIdx].x, k[tipIdx].y - k[mcpIdx].y);
      return dist > T(0.55);
    };

    const indexExt = isExtended(8, 5);
    const middleExt = isExtended(12, 9);
    const ringExt = isExtended(16, 13);
    const pinkyExt = isExtended(20, 17);

    const d = (i1, i2) => Math.hypot(k[i1].x - k[i2].x, k[i1].y - k[i2].y);
    const pointingDown = k[8].y > k[0].y;

    if (pointingDown) {
      if (indexExt && !middleExt && !ringExt && !pinkyExt) {
        if (d(4, 8) > T(0.6)) return "Q";
      }
      if (indexExt && middleExt && !ringExt && !pinkyExt) {
        return "P";
      }
    }

    const isHorizontal = Math.abs(k[8].x - k[5].x) > Math.abs(k[8].y - k[5].y) * 1.5;
    if (isHorizontal && !ringExt && !pinkyExt) {
      if (middleExt) return "H";
      return "G";
    }

    if (!middleExt && !ringExt && !pinkyExt) {
      if (indexExt) {
        if (d(4, 5) > T(0.9)) return "L";
        return "D";
      }
    }

    if (indexExt && middleExt && !ringExt && !pinkyExt) {
      if (Math.abs(k[8].x - k[12].x) < T(0.25)) return "R";
      const thumbY = k[4].y;
      const middleKnuckleY = k[9].y;
      if (thumbY < middleKnuckleY + T(0.2)) return "K";
      if (d(8, 12) > T(0.45)) return "V";
      return "U";
    }

    if (indexExt && middleExt && ringExt) {
      if (pinkyExt) {
        if (d(4, 17) < T(1.0)) return "B";
        return "🖐️";
      }
      return "W";
    }

    if (!indexExt && middleExt && ringExt && pinkyExt) {
      if (d(4, 8) < T(0.5)) return "F";
    }

    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
      if (d(4, 17) > T(1.1)) return "Y";
      return "I";
    }

    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
      const indexLen = d(8, 5);
      const gap = d(4, 8);
      const isCurved = indexLen > T(0.55) && indexLen < T(0.85);
      const hasGap = gap > T(0.5) && gap < T(0.9);

      if (isCurved && hasGap) return "C";

      const indexCurl = d(8, 5);
      if (d(4, 8) < T(0.5)) {
        if (indexCurl < T(0.35)) {
          if (d(4, 10) < T(0.35)) return "S";
          return "E";
        }
        return "O";
      }

      if (d(4, 5) > T(0.5) && k[4].y < k[5].y) return "A";

      const dIndex = d(4, 5);
      const dMiddle = d(4, 9);
      const dRing = d(4, 13);
      const dPinky = d(4, 17);

      if (dRing < T(0.3) || dPinky < T(0.35)) return "M";
      if (dMiddle < T(0.3)) return "N";
      if (dIndex < T(0.35)) return "T";

      if (indexCurl < T(0.35) && d(4, 13) < T(0.5)) return "E";
      return "S";
    }

    return "🖐️";
  }, []);

  // --- 3. THE HYBRID LOOP ---
  const detect = useCallback(async () => {
    if (!yoloModel || !mpLandmarker || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    // MEDIAPIPE LOGIC
    const mpResult = mpLandmarker.detectForVideo(video, Date.now());
    if (mpResult.landmarks && mpResult.landmarks.length > 0) {
      // NEW: We pass an object with keypoints to match your new function's requirements
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

    // YOLO VISUALS (Bounding Box Logic)
    const input = tf.tidy(() => {
      return tf.browser.fromPixels(video)
        .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
        .div(255.0)
        .expandDims(0);
    });

    let res = await yoloModel.executeAsync(input);
    if (Array.isArray(res)) res = res[0];
    let transRes = res.shape[1] < res.shape[2] ? res.transpose([0, 2, 1]) : res;
    const data = await transRes.data();
    const shape = transRes.shape;
    tf.dispose([res, transRes, input]);

    let maxScore = 0;
    let bestIdx = -1;
    const numAnchors = shape[1];
    const numChannels = shape[2];

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

      const bx = data[bestIdx], by = data[bestIdx + 1], bw = data[bestIdx + 2], bh = data[bestIdx + 3];
      const boxW = bw * scaleX, boxH = bh * scaleY;
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

  }, [yoloModel, mpLandmarker, dim, recognizeGesture]);

  useEffect(() => {
    if (!yoloModel || !mpLandmarker) return;
    const interval = setInterval(detect, 50);
    return () => clearInterval(interval);
  }, [yoloModel, mpLandmarker, detect]);

  const handleSpace = () => setText(t => t + " ");
  const handleBackspace = () => setText(t => t.slice(0, -1));
  const handleClear = () => setText("");
  const handleSpeak = () => {
    if (!text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="app-container">
      <header className="header">
        <img src="/logo.jpg" alt="Signetic Logo" className="logo" />
        <h1 className="app-title">Signetic</h1>
      </header>
      <div className="display-area">
        <h2 className="gesture-status">{gesture}</h2>
        <div className={`text-output ${!text ? 'placeholder' : ''}`}>
          {text || "Start signing to translate..."}
        </div>
        <div className="camera-wrapper" style={{ width: dim.w, height: dim.h }}>
          <Webcam ref={webcamRef} className="webcam-video" width={dim.w} height={dim.h} mirrored={true} />
          <canvas ref={canvasRef} className="drawing-canvas" width={dim.w} height={dim.h} />
        </div>
        <div className="controls">
          <button className="btn btn-primary" onClick={handleSpace}>Space</button>
          <button className="btn btn-warning" onClick={handleBackspace}>Backspace</button>
          <button className="btn btn-speak" onClick={handleSpeak}>Speak 🗣️</button>
          <button className="btn btn-danger" onClick={handleClear}>Clear</button>
        </div>
      </div>
    </div>
  );
}