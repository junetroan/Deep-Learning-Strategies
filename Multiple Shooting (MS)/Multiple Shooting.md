# Multiple Shooting #

# ANODE-MS 
This project is developed through my project and master's thesis as the final installation of my master's degree in Chemical Engineering and Biotechnology at the Norwegian University of Science and Technology (NTNU). 
The project runs from August 2023 until early June 2024 and all work published in this repository is to be considered my work. This project is part of a collaboration between an interdisciplinary team of researchers at NTNU, 
the National Commission of Banks and Insurance Companies of Honduras (CNBS), The University of Austin at Texas (UATX) and the Massachusetts Institute of Technology (MIT). 

**Purpose of the work** \
The work consists of developing an improved model for training stability of neural differential equations for time series forecasting. During the project thesis (autumn 2023) an Augmented Neural Ordinary Differential Equation Model utilising Data-Driven Multiple Shooting (ANODE-MS) for training of neural networks was developed. 
Preliminary tests of the proposed ANODE-MS model perform better than the NODE-MS model developed based on a [Multiple Shooting example](https://github.com/SciML/DiffEqFlux.jl/blob/master/test/multiple_shoot.jl) in the DiffEqFlux.jl documentation.
Currently, the model is being tested using different loss functions, randomised initialisation of the neural networks, and a set of experimental data (Hare-Lynx data) to assess the model's true performance.

## Simulation
### Design of Simulations

| Model | Data Utilised | Loss Function | Current status | 
|----------|----------|----------|----------|
|ANODE-MS | Lotka Volterra | Continuity + Prediction Error | Ready for simulation 🔜|
|         | Lotka Volterra | Total Trajectory Loss | Ready for simulation 🔜 |
|         | Hare Lynx      | Continuity + Prediction Error  | Ready for simulation 🔜 |
|         | Hare Lynx      | Total Trajectory Loss | Ready for simulation 🔜|
|NODE-MS  | Lotka Volterra | Continuity + Prediction Error  | Ready for simulation 🔜|
|         | Lotka Volterra | Total Trajectory Loss | Ready for simulation 🔜|
|         | Hare Lynx      | Continuity + Prediction Error  | In development ⏳|
|         | Hare Lynx      | Total Trajectory Loss | Ready for simulation 🔜|



### Simulation models
🔜 **sim-LV-ANODE-SS.jl [Ready for simulation]**
- File used for simulation of the ANODE-MS model that uses a loss function that minimises the loss at each prediction and enforces continuity. This code uses synthetically generated data from Lotka-Volterra dynamics.

🔜 **sim-LV-ANODE-MS.jl [Ready for simulation]**
- File used for simulation of the ANODE-MS model that uses a loss function that minimises the total trajectory loss. This code uses synthetically generated data from Lotka-Volterra dynamics. 

🔜  **sim-HL-ANODE-SS.jl [Ready for simulation]**
- File used for simulation of the ANODE-MS model that uses a loss function that minimises the loss at each prediction and enforces continuity. This code is using experimental Hare and Lynx data from the [FitODE repository](https://github.com/evolbio/FitODE/blob/unified/input/lynx_hare_data.csv?source=post_page-----9ec91fd838d7--------------------------------)

🔜 **sim-HL-ANODE-MS.jl [Ready for simulation]**
- File used for simulation of the ANODE-MS model that uses a loss function that minimises the total trajectory loss. This code is using experimental Hare and Lynx data from the [FitODE repository](https://github.com/evolbio/FitODE/blob/unified/input/lynx_hare_data.csv?source=post_page-----9ec91fd838d7--------------------------------)

🔜 **sim-LV-NODE-SS.jl [Ready for simulation]**
- File used for simulation of the NODE-MS model that uses a loss function that minimises the loss at each prediction and enforces continuity. This code uses synthetically generated data from Lotka-Volterra dynamics. Method inspired by a [Multiple Shooting example](https://github.com/SciML/DiffEqFlux.jl/blob/master/test/multiple_shoot.jl) in the DiffEqFlux.jl documentation.

🔜 **sim-LV-NODE-MS.jl [Ready for simulation]**
- File used for simulation of the NODE-MS model that uses a loss function that minimises the total trajectory loss. This code uses synthetically generated data from Lotka-Volterra dynamics. Method inspired by a [Multiple Shooting example](https://github.com/SciML/DiffEqFlux.jl/blob/master/test/multiple_shoot.jl) in the DiffEqFlux.jl documentation.

⏳  **sim-HL-NODE-SS.jl [In development]**
- File used for simulation of the NODE-MS model that uses a loss function that minimises the loss at each prediction and enforces continuity. This code uses experimental Hare and Lynx data from the [FitODE repository](https://github.com/evolbio/FitODE/blob/unified/input/lynx_hare_data.csv?source=post_page-----9ec91fd838d7--------------------------------). Method inspired by a [Multiple Shooting example](https://github.com/SciML/DiffEqFlux.jl/blob/master/test/multiple_shoot.jl) in the DiffEqFlux.jl documentation.
  
🔜 **sim-HL-NODE-MS.jl [Ready for simulation]**
- File used for simulation of the NODE-MS model that uses a loss function that minimises the total trajectory loss. This code uses experimental Hare and Lynx data from the [FitODE repository](https://github.com/evolbio/FitODE/blob/unified/input/lynx_hare_data.csv?source=post_page-----9ec91fd838d7--------------------------------). Method inspired by a [Multiple Shooting example](https://github.com/SciML/DiffEqFlux.jl/blob/master/test/multiple_shoot.jl) in the DiffEqFlux.jl documentation.



Note: \
The files in this folder are files taken from my repository "ANODE-MS", which was used until the 5th of February.
Due to changes in my master's thesis and technical problems with the previous repository, this repository was made
to include all code for my master's. 