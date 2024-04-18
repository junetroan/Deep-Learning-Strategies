# Deep Learning Strategies
This repository is the working repository for my master's thesis, where I deep dive into how novel deep learning strategies can be used for time series forecasting. The developed strategies, testing strategies and status of results are presented in this README. 

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
- Results obtained ✅


| Model | Case Study | Loss Function | Current Status | Results |
|----------|----------|----------|----------|----------|
|ANODE-MS | Lotka Volterra | Continuity + Prediction Error | | |
|         | Lotka Volterra | Total Trajectory Loss |  | |
|         | Hare Lynx      | Continuity + Prediction Error  |  | |
|         | Hare Lynx      | Total Trajectory Loss | | |
|         | NASDAQ Composite      | Total Trajectory Loss | | |
|         | F1 telemetry      | Total Trajectory Loss | | |
|ANODE-EH | Lotka Volterra | Continuity + Prediction Error | | |
|         | Lotka Volterra | Total Trajectory Loss |  | |
|MNODE-MS  | Lotka Volterra | Continuity + Prediction Error  | | |
|         | Lotka Volterra | Total Trajectory Loss | | |
|         | Hare Lynx      | Continuity + Prediction Error  | | |
|         | Hare Lynx      | Total Trajectory Loss | | |
|         | NASDAQ Composite      | Total Trajectory Loss | | |
|         | F1 telemetry      | Total Trajectory Loss | | |
|NODE-MS  | Lotka Volterra | Continuity + Prediction Error  | | |
|         | Lotka Volterra | Total Trajectory Loss |  | |
|         | Hare Lynx      | Continuity + Prediction Error  | | |
|         | Hare Lynx      | Total Trajectory Loss | | |
|         | NASDAQ Composite      | Total Trajectory Loss | | |
|         | F1 telemetry      | Total Trajectory Loss | | |
|NPEM      | Lotka Volterra | MSE  | | |
|         | Hare Lynx      | MSE  | | |
|         | NASDAQ Composite      | MSE |  | |
|         | F1 telemetry      | MSE | | |
