<h1 align="center">🛠️ Predictive Maintenance AI</h1>
<p align="center">
⚙️ An Arduino-powered system that predicts machine failure using ML logic trained offline.<br>
🎯 Combines real-time hardware simulation + AI-powered prediction in a smart, compact prototype.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-Arduino_Uno-blue?logo=arduino" />
  <img src="https://img.shields.io/badge/Model-Trained_Offline-green?logo=machine-learning" />
  <img src="https://img.shields.io/badge/Status-Working_Prototype-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 🚀 Features

- 🔧 Real-time simulation of motor and behavior input
- 💡 LED indicators for predicted status (🟥 Fault / 🟩 Normal)
- 🧠 ML logic trained offline on sensor-style data (XGBoost)
- 🔌 Fully integrated Arduino circuit
- 🎓 Built for academic demo & real-world applicability

---

## 🧰 Tech Stack

| Layer        | Tools Used                              |
|--------------|------------------------------------------|
| **Hardware** | Arduino Uno, L298N, DC Motor, Potentiometer, LEDs |
| **ML Model** | XGBoost (trained offline)                |
| **Simulation** | Analog Input Variations + Threshold Mapping |
| **Demo Mode** | LED Feedback (Red = Fail, Green = OK)   |

---

<details>
<summary>📦 <strong>Hardware Components</strong></summary>

- Arduino Uno  
- L298N Motor Driver  
- DC Motor (6V or 12V)  
- Potentiometer (10KΩ)  
- Breadboard + Jumper Wires  
- LEDs (Red & Green)  
- Resistors (220Ω or 1kΩ)  
- Power Supply (12V battery pack or adapter)

</details>

---

<details>
<summary>🧠 <strong>ML Model Summary</strong></summary>

- Model: XGBoost Classifier  
- Trained on: Simulated vibration/speed data  
- Output: Binary classification → *Failure* or *No Failure*  
- Deployment: Inference logic mapped into Arduino decision rules (approximation)  

</details>

---

<details>
<summary>📸 <strong>Screenshots</strong></summary>

<p float="left">
  <img src="![setup](https://github.com/user-attachments/assets/7f1550b4-7ca7-4788-91b0-bf8a4333c75b)" width="300"/>
  
</p>

</details>

---

## 🏆 Achievements

| 🏅 | Achievement |
|----|-------------|
| 🥇 | **1st Prize** – National Science Day Project Presentation (CIT) |
| 🧠 | Demonstrated real-time failure prediction using embedded AI |
| 🔗 | Integrated physical hardware + ML decision logic |
