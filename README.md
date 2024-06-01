# Deep Learning Strategies
This repository includes the code that was developed for my master's thesis, _**Novel Deep Learning Strategies for Time Series Forecasting**_, in which I explore how novel deep-learning strategies can be used for time series forecasting.

The work consisted of understanding state-of-the-art models and developing and testing novel strategies throughout four case studies. The case studies performed during this work are the following:
- 〽️ **Lotka Volterra system (Applied Mathematics):** Synthetic data from an LV system is often used to analyse nonlinear, dynamic systems.
- 🐇 **Hare and Lynx system (Ecology):** Real-world data from the Hare and Lynx ecosystem recorded through 80 consecutive years. 
- 💵 **Nasdaq Composite (Finance):** Real-world historical data of the Nasdaq Composite.
- 🏎️ **F1 Telemetry (Engineering):** Real-world data from a high-performance engineering system - specifically, the telemetry from a Formula One race

## 🚀 Developed Strategies
- Augmented Neural Ordinary Differential Equations using Expanding Horizon (ANODE-EH)
- Augmented Neural Ordinary Differential Equations using Multiple Shooting (ANODE-MS I + II)
- Modified Neural Ordinary Differential Equations using Multiple Shooting (MNODE-MS)
- Neural Prediction Error Method (NPEM)
  
## 📄 Structure of the repository
The folder structure of the repository is as follows: 

- [Models](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/models)
  - [ANODE-EH](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/models/ANODE-EH)
  - [ANODE-MS](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/models/ANODE-MS)
  - [MNODE-MS](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/models/MNODE-MS)
  - [NPEM](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/models/NPEM)
- [Case Studies](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies)
  - [Lotka-Volterra](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Lotka-Volterra%20(LV))
    - [Results](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Lotka-Volterra%20(LV)/results)
  - [Hare and Lynx](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Hare%20and%20Lynx%20(HL))
    - [Data](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Hare%20and%20Lynx%20(HL)/data)
    - [Result](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Hare%20and%20Lynx%20(HL)/results)
  - [Nasdaq Composite](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Nasdaq%20Composite%20(IXIC))
    - [Data](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Nasdaq%20Composite%20(IXIC)/data)
    - [Results](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/Nasdaq%20Composite%20(IXIC)/results)
  - [F1 Telemetry](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/F1%20Telemetry%20(F1))
    - [Data](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/F1%20Telemetry%20(F1)/data)
    - [Results](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/case_studies/F1%20Telemetry%20(F1)/results)
- [Comparisons](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/comparisons)
  - [Hare and Lynx (HL)](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/comparisons/Hare%20and%20Lynx%20(HL))
  - [Lotka-Volterra (LV)](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/comparisons/Lotka-Volterra%20(LV))
- [Scripts](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/scripts)
- [Additional Data](https://github.com/junetroan/Deep-Learning-Strategies/tree/main/additional_data)
