# 🌱 Crop Disease Finder

> **AI-Powered Agricultural Prototype for Leaf Disease Classification & Agronomic Diagnosis**  
> Built with pure HTML5, CSS3, and JavaScript — featuring a dual-engine architecture with **Offline K-Nearest Neighbors (KNN)** and **Google Gemini Multimodal Vision AI**.

---

## 📌 Overview

**Crop Disease Finder** is an interactive, lightweight web application designed to help farmers, students, and agricultural researchers quickly identify crop pathologies from leaf photographs.

The project features a **Dual-Engine AI architecture**:
1. **🌿 Offline KNN Classifier (College Viva / Demo Mode)**: Runs completely in-browser without requiring an internet connection or backend server. Uses HTML5 Canvas for computer vision feature extraction (chlorophyll ratio, chlorosis index, lesion variance) and classifies samples using weighted Euclidean distance ($k = 3$).
2. **✨ Google Gemini Multimodal Vision AI (Cloud Production Mode)**: Sends leaf imagery directly to Google's state-of-the-art `gemini-1.5-flash` / `gemini-2.5-flash` neural network for deep pathological analysis and structured JSON diagnosis.

---

## ✨ Key Features

- **Dual AI Engines**: Switch seamlessly between offline algorithmic classification (KNN) and cloud-based deep learning vision (Google Gemini).
- **Zero Dependencies / Frameworks**: Built using pure vanilla JavaScript, semantic HTML5, and modern CSS3 (glassmorphism design system).
- **Offline Synthetic Sample Generator**: Includes 6 built-in canvas leaf presets (Healthy Leaf, Corn Rust, Rice Blast, Wheat Mildew, Tomato Blight, Potato Blight) for instant demonstration during lab evaluations and viva presentations without needing external image uploads.
- **Drag-and-Drop Image Upload**: Supports drag-and-drop and standard file selection for any JPG, PNG, or WEBP leaf image.
- **Comprehensive Diagnostic Output**:
  - Identified Crop species
  - Detected Pathology / Disease
  - Dynamic Confidence Score Bar
  - Visible Symptoms breakdown
  - Actionable Agricultural & Chemical Treatment Recommendations
- **Viva Metric Inspector**: Exposes raw mathematical features calculated during KNN analysis:
  - Green Chlorophyll %
  - Brown/Yellow Lesion %
  - Spot/Texture Variance
  - Nearest Neighbor Reference Match
- **Fully Responsive & Modern UI**: Tailored with Plus Jakarta Sans typography, animated status indicators, accessible cards, and smooth CSS transitions.

---

## 🌾 Supported Crops & Diseases (14 Categories)

| Crop | Disease Classes Diagnosed |
| :--- | :--- |
| 🌽 **Corn (Maize)** | Common Rust (`Puccinia sorghi`), Northern Leaf Blight (`Exserohilum turcicum`), Gray Leaf Spot (`Cercospora zeae-maydis`) |
| 🌾 **Paddy (Rice)** | Rice Blast (`Magnaporthe oryzae`), Brown Spot (`Bipolaris oryzae`), Bacterial Leaf Blight (`Xanthomonas oryzae`) |
| 🌾 **Wheat** | Leaf Rust (`Puccinia triticina`), Stripe Rust (`Puccinia striiformis`), Powdery Mildew (`Blumeria graminis`) |
| 🍅 **Tomato** | Early Blight (`Alternaria solani`), Late Blight (`Phytophthora infestans`) |
| 🥔 **Potato** | Early Blight (`Alternaria solani`), Late Blight (`Phytophthora infestans`) |
| 🌿 **Healthy Plants** | Normal leaf with vibrant chlorophyll, intact venation, and absence of lesions |

---

## 🧠 How the AI Works

### 1. Offline K-Nearest Neighbors (KNN) Engine
1. **Pixel Extraction**: When an image is uploaded, an in-memory HTML5 Canvas samples the image pixels.
2. **Feature Vector Representation**: The application computes a 7-dimensional botanical feature vector:
   $$\vec{X} = [\text{avgR}, \text{avgG}, \text{avgB}, \text{greenRatio}, \text{brownYellowRatio}, \text{spotVariance}, \text{paleRatio}]$$
3. **Weighted Euclidean Distance**:
   $$d(\vec{X}, \vec{Y}) = \sqrt{\sum_{i=1}^{n} w_i \cdot (X_i - Y_i)^2}$$
   - Distinct weights ($w_i$) emphasize yellow-brown necrosis ($w=2.2$) and chlorophyll health ($w=2.0$).
4. **Majority Voting ($k=3$)**: The algorithm computes inverse-distance weighted votes across the closest botanical dataset references to determine the most probable disease.

### 2. Google Gemini Vision Engine
- Images are encoded to base64 JPEG directly in the browser.
- Queries Google's Gemini Vision API with structured system prompts.
- Forces an enforced JSON schema output containing the crop, disease, confidence, symptoms, and agronomic management advice.

---

## 🚀 Quick Start & Installation

### Option 1: Direct Browser Launch (No installation required)
Since this project is built with vanilla HTML/JS:
1. Clone or download this repository.
2. Double-click [index.html](file:///p:/AI%20prototype/crop-disease-finder/index.html) to open it in any web browser (Chrome, Edge, Firefox, Safari).

### Option 2: Run with a Local Static Server
To avoid browser local file protocol (`file://`) restrictions on some mobile devices:

```bash
# Using Python
python -m http.server 8000

# Using Node.js (npx)
npx serve .

# Or open in VS Code with the "Live Server" extension
```
Visit `http://localhost:8000` in your web browser.

---

## 🔑 Configuring Gemini API Key (Optional)

The offline KNN engine works 100% without an API key.  
To use the **Google Gemini Vision AI** engine:
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Open [script.js](file:///p:/AI%20prototype/crop-disease-finder/script.js) and locate line 136:
   ```javascript
   const GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE";
   ```
3. Replace the placeholder with your key and save the file.

---

## 📂 Project Structure

```text
crop-disease-finder/
├── index.html       # Application layout, semantic markup, and disease encyclopedia modal
├── style.css        # Glassmorphic styling, animations, responsive grid system
├── script.js       # Dual AI engines (KNN + Gemini), canvas feature extractor, synthetic demo generator
├── LICENSE          # MIT License
└── README.md        # Comprehensive documentation
```

---

## 🎓 Viva / Academic Presentation Notes

- **Q: Why use KNN alongside Deep Learning?**  
  *A:* KNN provides complete offline operability, instant inference without network latency, and transparent mathematical explainability (white-box model), whereas Gemini offers open-domain generalization and deep multimodal reasoning.
- **Q: How does the model distinguish between healthy leaves and diseased leaves?**  
  *A:* By comparing normalized chlorophyll chromaticity ($G > R \text{ and } G > B$) against chlorotic and necrotic pixel ratios ($R > B \text{ and } G > B$).
- **Q: What happens if an unseen disease is photographed?**  
  *A:* KNN finds the mathematically closest known profile. Gemini uses deep generative semantic knowledge to identify unlisted crops and rare pathogens.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).  
Copyright (c) 2026 Balamaheshchandar.
