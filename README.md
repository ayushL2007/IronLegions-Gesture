<img width="2" height="1" alt="image" src="https://github.com/user-attachments/assets/4d491f80-359d-41ce-ab1c-1f527f13422c" /># Signetic 
🔴 LIVE DEMO  : https://signetic.vercel.app/

## Real-Time Sign Language to Text & Voice Converter

---

## 🌍 The Problem

Communication between sign language users and non-signers remains a major accessibility challenge in everyday digital interactions. Existing solutions typically suffer from one or more of the following limitations:

* Require **specialized hardware** which most don't have access to/
* Depend on **cloud-based inference**, introducing latency and concerning privacy risks
* Perform poorly in **real-world environments** with background clutter and lighting variations
* Lack **accessibility tools** such as speech output
* Are **not scalable**, expensive to deploy, or difficult to integrate

As a result, sign language users are often excluded from seamless communication in web-based systems.

There is a strong need for a **lightweight, real-time, privacy-preserving, and accurate sign language translation system** that runs directly in the browser and works reliably across devices.

---

## 💡 Our Solution – Signetic

**Signetic** is a fully responsive **browser-based real-time ASL fingerspelling to text translator** that runs **entirely on the client side**.

Instead of relying on cloud APIs or black-box classifiers, Signetic uses a **pose-based, explainable computer vision pipeline** that converts hand gestures into readable text with minimal latency.

### Core Principles

* 🧠 **Pose-based ML** instead of raw image classification
* 🔐 **Zero backend** – complete user privacy
* ⚡ **Low latency** via WebGL-accelerated inference
* ♿ **Accessibility-first** with speech output
* 🧩 **Can be used by anyone instantly , as it comes with no hardware restrictions and is a fast single page web application**
* 📂 **History Saving** – previous words saved via browser cookies
* 🎨 **Theme Switcher** – toggle between light and dark modes
* ➕ **Added features like Clear all, backspace, and text to speech for more accessibility**

---

## 🧠 How Signetc Works 

```
Webcam Feed
   ↓
YOLOv8 Hand Detection (Bounding Box + Confidence)
   ↓
MediaPipe Hands (21 × 3D Landmarks)
   ↓
Geometric Feature Engineering
   ↓
Rule-Based Gesture Classification
   ↓
Temporal Stabilization Engine
   ↓
Text Output + Text-to-Speech + History saving
```

---

## 🧩 Core Technical Architecture (Detailed)

### 1️⃣ Computer Vision Layer – YOLOv8 Hand Detection

Signetic uses a **custom-trained YOLOv8 model (trained through Duality AI )** to detect hand presence in real time.

#### What YOLOv8 Does

* Detects whether a hand is present in the frame
* Produces a **stable bounding box** around the hand
* Outputs a **confidence score** to filter false detections

#### Why YOLOv8 Is Necessary

* Prevents MediaPipe from hallucinating landmarks when no hand exists
* Filters background noise before pose estimation
* Stabilizes the UI overlay (green bounding box)
* Improves robustness in cluttered or dynamic scenes

YOLOv8 acts as a **spatial gatekeeper** for the entire pipeline.

---

## 🤖 YOLOv8 + Duality AI (Synthetic Data Training)

The YOLOv8 model used in Signetic is **custom-trained**

### Why Custom Training Was Required

Generic hand detectors often fail due to:

* Limited pose diversity
* Poor generalization to camera angles
* Sensitivity to lighting and occlusion

To overcome this, YOLOv8 was trained using **Duality AI synthetic data**.

### What Is Duality AI?

**Duality AI** provides physics-accurate synthetic environments that enable:

* Controlled camera viewpoints
* Diverse hand orientations
* Variable lighting and backgrounds
* Perfect ground-truth annotations

### How Duality AI Is Used in Signetic

* Synthetic hand scenes are generated using Duality AI
* These scenes are used to train YOLOv8 for hand localization
* The resulting model generalizes better to real-world webcam input

### Why Synthetic Data Matters

| Real-World Data   | Synthetic Data (Duality AI) |
| ----------------- | --------------------------- |
| Limited diversity | Infinite pose variation     |
| Annotation noise  | Perfect labels              |
| Privacy concerns  | Privacy-safe                |
| Expensive         | Scalable                    |

Synthetic data significantly improves **robustness and generalization**, which is critical for browser-based computer vision systems.

---

### 2️⃣ Pose Estimation Layer – Google MediaPipe Hands

Once a hand is detected, Signetic uses **MediaPipe Hands** to extract pose information.

* Extracts **21 normalized 3D landmarks** per hand
* Tracks finger joints, tips, palm center, and orientation
* GPU-accelerated using **WASM + WebGL**

#### Why Pose-Based ML Over Image-Based ML

* Computationally efficient
* Robust to lighting and background changes
* Works consistently across different users

---

### 3️⃣ Geometric Feature Engineering (Key Innovation)

Instead of using a black-box neural classifier, Signetic computes **explicit geometric features** from landmarks:

* Finger extension states
* Joint-to-joint distances
* Thumb relative positioning
* Inter-finger gaps
* Hand orientation vectors
* Vertical / horizontal / inverted pose detection

This approach makes the system:

* Deterministic
* Debuggable
* Transparent
* Improvable 


---

### 4️⃣ Gesture Classification Logic

Each ASL alphabet is identified using **rule-based geometric constraints**:

for example:
* **A / S/ E** → Thumb placement along other fingers
* **G / H** → Horizontal index alignment
* **R** → Crossed index and middle fingers
* **B/F/D** → Position of fingers in a palm

This avoids heavy training and ensures consistent output along with efficiency

---

### 5️⃣ Temporal Stabilization Engine

Live video is inherently noisy. Signetic stabilizes predictions using:

* Frame buffers
* Majority voting
* Stable gesture thresholds
* Duplicate character prevention

Only **consistent gestures across multiple frames** are converted into text.

---

## 🗣️ Accessibility Layer

* **Text-to-Speech** using the Web Speech API
* Converts translated text into audible speech
* Saves the text output via browser cookies
* Enables two-way communication between signers and non-signers

---

## 💾 History Saving & Theme Switcher

* **Previous Words Record**: Signetic saves translated words using **browser cookies**  
  * Users can view and manage their history  
  * Option to **delete logs** and clear previous entries  

* **Theme Switcher**: Toggle between **light** and **dark** modes for user comfort  

---


## 🧪 Why Dual-Model Architecture (YOLO + MediaPipe)?

| Challenge        | Single Model      | Signetic Dual-Model        |
| ---------------- | ----------------- | -------------------------- |
| False positives  | High              | Reduced via YOLO filtering |
| Background noise | Affects landmarks | Isolated hand ROI          |
| Visual stability | Jittery           | Frozen bounding logic      |
| Accuracy         | Inconsistent      | Confidence-aware pipeline  |

This dual-model design significantly improves **real-world reliability**.

---

## 🧬 Google Tech Stack Used 

Signetic is deeply built on Google’s Web-ML ecosystem:

* **TensorFlow.js** – Client-side ML inference engine
* **MediaPipe Hands** – Real-time hand pose estimation
* **WebGL** – GPU-accelerated computation
* **WASM (WebAssembly)** – High-performance execution
* **Web Speech API** – Text-to-Speech accessibility

These technologies enable:

* Browser-native execution
* Zero backend dependency
* Privacy-preserving ML
* Cross-platform scalability

---

## 🛠️ Full Tech Stack

### Frontend

* React.js (Functional Components)
* HTML5 Canvas
* CSS3

### Machine Learning & Vision

* TensorFlow.js
* YOLOv8 (Custom-trained, TFJS GraphModel)
* MediaPipe Tasks Vision

### Runtime

* WebGL Backend
* WASM Execution

---

## ✨ Extra Features

* Live gesture overlay with bounding box
* Manual controls (space, backspace, clear)
* Real-time feedback UI
* Mobile-responsive camera handling

---



## 🔮 Future Scope

* Dynamic gesture recognition 
* Full word-level sign recognition
* Multi-language sign language support
* Offline-first PWA deployment
* Personalized gesture calibration
* Sign-to-Sign translation
* Integration on large platforms
* **Enhanced UX**: Advanced history management, customizable themes, and accessibility personalization 

---

## 🏁 Conclusion

**Signetic** is not just a demo — it is a **production-grade, browser-native computer vision system** that demonstrates how modern ML, accessibility, and web technologies can converge to solve real human problems.

It stands at the intersection of:

* Computer Vision
* Human-Computer Interaction
* Responsible AI
* Web-based ML Deployment

---

> "Accessibility should not be limited by permission, hardware, or compromise — Signetic proves that."






Acknowledgement: [Rion Dsilvia](https://github.com/RionDsilvaCS/yolo-hand-pose) for her yolo v8 model

Made with ❤️ by Ayush nd Anurag


