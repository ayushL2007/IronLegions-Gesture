import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [gesture, setGesture] = useState("⏳ Loading...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 480, h: 360 });

  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const STABLE_THRESHOLD = 15;

  const wasHandPresent = useRef(false);
  const gestureBuffer = useRef([]);
  const BUFFER_SIZE = 10;

  const THEME_COLOR = "#FFFFC5";
  const THEME_RGB = "255, 255, 197";

  // --- KEYBOARD & TTS ---
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Backspace") {
        setText((prev) => prev.slice(0, -1));
        lastWrittenLetter.current = "";
      } else if (event.key === "Escape") {
        setText("");
        lastWrittenLetter.current = "";
      } else if (event.key === " ") {
        setText((prev) => prev + " ");
        lastWrittenLetter.current = "";
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const speakText = () => {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    window.speechSynthesis.speak(utterance);
  };

  // --- RESPONSIVE DIMENSIONS ---
  useEffect(() => {
    const updateDimensions = () => {
      const screenW = window.innerWidth;
      const screenH = window.innerHeight;
      const isPortrait = screenH > screenW;
      let w, h;
      if (isPortrait) {
        w = Math.min(screenW - 20, 480);
        h = w * 1.333;
      } else {
        w = Math.min(screenW - 40, 480);
        h = w * 0.75;
      }
      setDim({ w: Math.round(w), h: Math.round(h) });
    };
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    window.addEventListener("orientationchange", updateDimensions);
    return () => {
      window.removeEventListener("resize", updateDimensions);
      window.removeEventListener("orientationchange", updateDimensions);
    };
  }, []);

  // --- TF MODEL INIT ---
  useEffect(() => {
    (async () => {
      await tf.ready();
      await tf.setBackend("webgl");
      const d = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        { runtime: "tfjs", modelType: "full", maxHands: 1 }
      );
      setDetector(d);
      setGesture("👀 Show Hand");
    })();
  }, []);

  // --- GESTURE LOGIC ---
  const recognizeGesture = useCallback((hand) => {
    const k = hand.keypoints;
    const handSize = Math.hypot(k[9].x - k[0].x, k[9].y - k[0].y);
    const T = (factor) => handSize * factor;

    // Helper: Distance from tip to knuckle is large (Finger extended)
    const isExtended = (tipIdx, mcpIdx) => {
        const dist = Math.hypot(k[tipIdx].x - k[mcpIdx].x, k[tipIdx].y - k[mcpIdx].y);
        return dist > T(0.55);
    };

    const indexExt = isExtended(8, 5);
    const middleExt = isExtended(12, 9);
    const ringExt = isExtended(16, 13);
    const pinkyExt = isExtended(20, 17);

    // Helper: Distance between two keypoints
    const d = (i1, i2) => Math.hypot(k[i1].x - k[i2].x, k[i1].y - k[i2].y);

    // --- VARIABLES FOR YOUR C LOGIC ---
    const thumbTip = k[4];
    const indexMcp = k[5]; // Index Knuckle
    const distThumbIndex = Math.hypot(thumbTip.x - k[8].x, thumbTip.y - k[8].y);
    const distIndexTipKnuckle = Math.hypot(k[8].x - k[5].x, k[8].y - k[5].y);

    // Check Orientation
    const pointingDown = k[8].y > k[0].y; 

    // --- LOGIC TREE ---

    // 1. DOWNWARD GROUP (P, Q)
    if (pointingDown) {
        if (indexExt && !middleExt && !ringExt && !pinkyExt) {
             if (d(4, 8) > T(0.6)) return "Q";
        }
        if (indexExt && middleExt && !ringExt && !pinkyExt) {
             return "P"; 
        }
    }

    // 2. HORIZONTAL GROUP (G, H)
    const isHorizontal = Math.abs(k[8].x - k[5].x) > Math.abs(k[8].y - k[5].y) * 1.5;
    if (isHorizontal && !ringExt && !pinkyExt) {
        if (middleExt) return "H";
        return "G";
    }

    // 3. SINGLE FINGER GROUP (D, L)
    if (!middleExt && !ringExt && !pinkyExt) {
        if (indexExt) {
             if (d(4, 5) > T(0.9)) return "L"; 
             return "D";
        }
    }

    // 4. TWO FINGER GROUP (U, V, R, K)
    if (indexExt && middleExt && !ringExt && !pinkyExt) {
        if (Math.abs(k[8].x - k[12].x) < T(0.25)) return "R"; 

        // K CHECK
        const thumbY = k[4].y;
        const middleKnuckleY = k[9].y;
        if (thumbY < middleKnuckleY + T(0.2)) {
             return "K";
        }

        if (d(8, 12) > T(0.45)) return "V";
        return "U";
    }

    // 5. OPEN HAND / W / F / B / C (INTEGRATED HERE)
    if (indexExt && middleExt && ringExt) {
        if (pinkyExt) {
            // ** C CHECK (Moved Here per your request) **
            if (distThumbIndex < T(0.8) && distIndexTipKnuckle < T(0.85)) return "C";

            if (d(4, 17) < T(1.0)) return "B"; 
            return "🖐️";
        }
        return "W";
    }

    // F Check (Circle)
    if (!indexExt && middleExt && ringExt && pinkyExt) {
         if (d(4, 8) < T(0.5)) return "F";
    }

    // 6. PINKY GROUP (I, Y)
    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
        if (d(4, 17) > T(1.1)) return "Y";
        return "I";
    }

    // 7. FIST GROUP (A, E, M, N, S, T, O)
    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
        
        // E vs O logic
        const indexCurl = d(8, 5);
        
        // O Check (Thumb touching Index AND Index is Arched)
        if (d(4, 8) < T(0.5)) {
             if (indexCurl < T(0.35)) {
                 if (d(4, 10) < T(0.35)) return "S";
                 return "E"; 
             }
             return "O";
        }

        // A Check
        if (d(4, 5) > T(0.5) && k[4].y < k[5].y) return "A";
        
        // M, N, T Logic
        const dIndex = d(4, 5);  
        const dMiddle = d(4, 9); 
        const dRing = d(4, 13);  
        const dPinky = d(4, 17); 

        if (dRing < T(0.3) || dPinky < T(0.35)) return "M";
        if (dMiddle < T(0.3)) return "N";
        if (dIndex < T(0.35)) return "T";

        // Fallback for tight E detection
        if (indexCurl < T(0.35) && d(4, 13) < T(0.5)) return "E";

        return "S"; 
    }

    return "🖐️";
  }, []);

  // --- DETECTION LOOP ---
  const detect = useCallback(async () => {
    if (!detector || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    
    // Set Dimensions
    canvas.width = dim.w;
    canvas.height = dim.h;
    ctx.clearRect(0, 0, dim.w, dim.h);

    const hands = await detector.estimateHands(video, { flipHorizontal: true });

    if (hands.length > 0) {
      wasHandPresent.current = true;
      const hand = hands[0];
      const rawGesture = recognizeGesture(hand);
      
      // Smoothing Buffer
      gestureBuffer.current.push(rawGesture);
      if (gestureBuffer.current.length > BUFFER_SIZE) gestureBuffer.current.shift();
      const counts = {};
      let maxCount = 0;
      let smoothedGesture = rawGesture;
      gestureBuffer.current.forEach((g) => {
        counts[g] = (counts[g] || 0) + 1;
        if (counts[g] > maxCount) {
          maxCount = counts[g];
          smoothedGesture = g;
        }
      });
      setGesture(smoothedGesture);

      // Stability Check
      if (smoothedGesture === stableGesture.current) {
        stableCount.current += 1;
      } else {
        stableGesture.current = smoothedGesture;
        stableCount.current = 1;
      }

      // Typing Logic
      if (stableCount.current >= STABLE_THRESHOLD) {
        if (smoothedGesture !== "🖐️" && smoothedGesture !== "👀 Show Hand") {
          if (smoothedGesture !== lastWrittenLetter.current) {
            setText((t) => t + smoothedGesture);
            lastWrittenLetter.current = smoothedGesture;
          }
        } else {
           lastWrittenLetter.current = "";
        }
      }

      // Draw Keypoints
      ctx.fillStyle = THEME_COLOR;
      hand.keypoints.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      // Hand Left Frame -> Add Space
      setGesture("👀 Show Hand");
      stableCount.current = 0;
      lastWrittenLetter.current = "";
      if (wasHandPresent.current) {
        setText((prev) => (prev.length > 0 && !prev.endsWith(" ") ? prev + " " : prev));
        wasHandPresent.current = false;
      }
    }
  }, [detector, recognizeGesture, dim]);

  useEffect(() => {
    if (!detector) return;
    let raf;
    const loop = () => {
      detect();
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [detector, detect]);

  return (
    <div style={{ background: "#ede7e7ff", minHeight: "100vh", color: "#210303ff", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", padding: "15px", boxSizing: "border-box", overflowX: "hidden" }}>
      <h1 style={{ fontSize: "clamp(0.8rem, 4vw, 1rem)", margin: "0 0 10px 0", color: "#2b9308ff" }}>
        Welcome to Signetic - Sign Language to Text Converter
      </h1>
      <h2 style={{ fontSize: "clamp(1rem, 6vw, 1.5rem)", margin: "0 0 10px 0", color: "#272704ff", minHeight: "30px" }}>
        {gesture}
      </h2>
      <div style={{ width: "100%", maxWidth: "480px", padding: "10px", border: `2px solid ${THEME_COLOR}`, borderRadius: 12, minHeight: "50px", fontSize: "1.2rem", background: `rgba(${THEME_RGB}, 0.1)`, wordWrap: "break-word", textAlign: "left", marginBottom: "15px" }}>
        {text || <span style={{ opacity: 0.5 }}>Start signing...</span>}
      </div>
      <div style={{ display: "flex", gap: "15px", marginBottom: "15px" }}>
        <button onClick={() => { setText(""); lastWrittenLetter.current = ""; }} style={{ padding: "8px 16px", background: "red", color: "#fff", border: "none", borderRadius: 8, fontSize: "0.9rem", fontWeight: "bold", cursor: "pointer" }}>CLEAR</button>
        <button onClick={() => { setText((t) => t.slice(0, -1)); lastWrittenLetter.current = ""; }} style={{ padding: "8px 16px", background: "orange", color: "#fff", border: "none", borderRadius: 8, fontSize: "0.9rem", fontWeight: "bold", cursor: "pointer" }}>BACK</button>
        <button onClick={speakText} style={{ padding: "8px 16px", background: "#2b9308ff", color: "#fff", border: "none", borderRadius: 8, fontSize: "0.9rem", fontWeight: "bold", cursor: "pointer" }}>🔊 HEAR</button>
      </div>
      <div style={{ position: "relative", width: dim.w, height: dim.h, border: `4px solid #272704ff`, borderRadius: "20px", overflow: "hidden", backgroundColor: "#000", boxShadow: `0 0 15px rgba(0,0,0,0.3)` }}>
        <Webcam ref={webcamRef} mirrored width={dim.w} height={dim.h} videoConstraints={{ facingMode: "user", aspectRatio: dim.h > dim.w ? 0.75 : 1.333 }} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }} />
      </div>
    </div>
  );
}