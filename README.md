# Deep Learning Strategies
This repository is the working repository for my master's thesis, where I deep dive into how novel deep learning strategies can be used for time series forecasting. The developed strategies, testing strategies and status of results are presented in this README. 

The work has consisted of understanding current strategies, developing novel strategies, and testing both existing and novel strategies throughout four case studies. The case studies performed during this work are the following:
- **Lotka Volterra system (Ecology/Applied Mathematics):** Synthetic data from an LV system, which is often used in cases where nonlinear, dynamic systems are analysed.
- **Hare and Lynx system (Ecology):** Real-world data from the Hare and Lynx ecosystem recorded through 80 consecutive years.
- **NASDAQ Composite (Finance):** Real-world historical data of the NASDAQ Composite.
- **F1 Telemetry (Engineering):** Real-world data from high-performance engineering system. 

## 🚀 Developed Strategies
- Augmented Neural Ordinary Differential Equations using Multiple Shooting (ANODE-MS)
- Modified Neural Ordinary Differential Equations using Multiple Shooting (MNODE-MS)
- Neural Prediction Error Method (NPEM)

## 🧐 Testing Strategies
- Augmented Neural Ordinary Differential Equations using Expanding Horizon (ANODE-EH)
- Neural Ordinary Differential Equations using Multiple Shooting (origin: DiffEqFlux.jl - multipleshoot) (NODE-MS)

## 📝 Status of Results
_**Note! Loss Function, Current Status and Results need to be updated according to results and planned work.**_

**Status**:
- To be developed ✨
- In development ⏳
- Ready for simulation 🔜
- Ready to run 🏃🏼‍♀️
- Results obtained ✅
- Results analysed 📈

| Model | Case Study | Loss Function | Current Status | Results |
|----------|----------|----------|----------|----------|
|ANODE-MS | Lotka Volterra | Total Trajectory Loss | Results obtained ✅ | |
|         | Hare Lynx      | Total Trajectory Loss | Ready for simulation 🔜 | |
|         | NASDAQ Composite      | Total Trajectory Loss | Ready to run 🏃🏼‍♀️ | |
|         | F1 telemetry      | Total Trajectory Loss | Ready to run 🏃🏼‍♀️ | |
|ANODE-EH | Lotka Volterra | Continuity + Prediction Error | Results obtained ✅ | |
|         | Lotka Volterra | Total Trajectory Loss | To be developed ✨ | |
|MNODE-MS  | Lotka Volterra | Total Trajectory Loss   | To be developed ✨ | |
|         | Hare Lynx      | Total Trajectory Loss | To be developed ✨ | |
|         | NASDAQ Composite      | Total Trajectory Loss | To be developed ✨ | |
|         | F1 telemetry      | Total Trajectory Loss | To be developed ✨ | |
|NODE-MS  | Lotka Volterra |  Total Trajectory Loss  | Results obtained ✅ | |
|         | Hare Lynx      | Total Trajectory Loss |  Ready for simulation 🔜 | |
|         | NASDAQ Composite      | Total Trajectory Loss | To be developed ✨ | |
|         | F1 telemetry      | Total Trajectory Loss | To be developed ✨ | |
|NPEM      | Lotka Volterra | MSE  | Results obtained ✅ | |
|         | Hare Lynx      | MSE  | Ready to run 🏃🏼‍♀️| |
|         | NASDAQ Composite      | MSE | To be developed ✨ | |
|         | F1 telemetry      | MSE | Ready to run 🏃🏼‍♀️| |
