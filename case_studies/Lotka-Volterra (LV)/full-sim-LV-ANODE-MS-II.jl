#=

Simulation file for the Full Simulation of the ANODE-MS II Model on the Lotka-Volterra data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-24

=#

# Loading libraries
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
gr()

# Function for the Lotka-Volterra model
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Generating synthetic Lotka-Volterra data
tspan = (0.0f0, 10.0f0)
rng = StableRNG(1111)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

# Adding noise to the synthetic data
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
u0 = [Xₙ[1,1], 0]

# Experimental parameters
groupsize = 5 # number of points in each shooting segment   
predsize = 5 # number of points to predict per shooting segment
iters = 2 # number of iterations of simulation
state = 2 # number of state variables in the neural network

@time begin
    for i in 1:iters
        
        println("Simulation $i")

        #Random numbers
        rng1 = StableRNG(i)
        rng2 = StableRNG(i+2)
        rng3 = StableRNG(i+3)

        # Simple neural network to predict Lotka-Volterra dynamics
        U = Lux.Chain(Lux.Dense(state, 30, tanh),
                  Lux.Dense(30, state))
        
        # Get the initial parameters and state variables of the model
        p, st = Lux.setup(rng1, U)
    
        # Simple neural network to predict initial points for use in multiple-shooting training
        U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30, state - 1))
        p0, st0 = Lux.setup(rng2, U0_nn)
    
        # Define the hybrid model
        function ude_dynamics!(du, u, p, t, p_true)
            û = U(u, p.vector_field_model, st)[1] # Network prediction
            du[1:end] = û[1:end]
        end
    
        # Closure with the known parameter
        nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
    
        # Construct ODE Problem
        augmented_u0 = vcat(u0[1], randn(rng3, Float32, state - 1))
        params = ComponentVector{Float32}(vector_field_model = p, initial_condition_model = p0)
        prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan, params, saveat = 0.25f0)
    
        # Grouping the data into trajectories for multiple shooting
        function group_x(X::Vector, groupsize, predictsize)
            parent = [X[i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(X) - max(groupsize, predictsize) + 1]
            parent = reduce(hcat, parent)
            targets = parent[1:groupsize,:]
            nn_predictors = parent[1:predictsize,:]
            u0 = parent[1, :]
            return parent, targets, nn_predictors, u0
        end
    
        pas, targets, nn_predictors, u0_vec = group_x(Xₙ[1,:], groupsize, predsize)
    
        # Prediction function utilising the multiple shooting method for training the model
        function predict(θ)
            function prob_func(prob, i, repeat)
                u0_nn = U0_nn(nn_predictors[:, i], θ.initial_condition_model, st0)[1]
                u0_all = vcat(u0_vec[i], u0_nn)
                remake(prob, u0 = u0_all, tspan = (t[1], t[groupsize]))
        end
            sensealg = ReverseDiffAdjoint()
            shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
            Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
            p=θ, trajectories = length(u0_vec), sensealg = sensealg))
        end
    
        # Loss function for the prediction function
        function loss(θ)
            X̂ = predict(θ)
            continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
            prediction_error = mean(abs2, targets .- X̂[1,:,:])
            prediction_error + continuity*10f0
        end

        # Prediction function for the final model
        function predict_final(θ)
            predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
            u0_all = vcat(u0_vec[1], predicted_u0_nn)
            prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
            X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), 
            abstol = 1f-6, reltol = 1f-6, 
            saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
            X̂
        end

        # Loss function for the final prediction function
        function final_loss(θ)
            X̂ = predict_final(θ)
            prediction_error = mean(abs2, Xₙ[1,:] .- X̂[1, :])
            prediction_error
        end

        # Defining an array to store loss data for each iteration
        losses = Float32[]

        # Callback that prints the current loss every 50 iterations
        callback = function (θ,l)

            push!(losses, final_loss(θ))

            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end
        
        # Pre-training the model
        adtype = Optimization.AutoZygote()  
        optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, params)
        res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)
    
        # Saving the loss data
        losses_df = DataFrame(loss = losses)
        CSV.write("full-sim-LV-ANODE-MS-II/Loss Data/Losses $i.csv", losses_df, writeheader = false)

        # Gathering the predictions from the pre-trained model and calculating the loss
        full_traj = predict_final(res_ms.u)
        full_traj_loss = final_loss(res_ms.u)

        # Training the model
        optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
        optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
        @time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

        # Gathering the predictions from the pre-trained model and calculating the loss
        full_traj2 = predict_final(res_final.u)
        actual_loss = Xₙ[1,:] - full_traj2[1,:]
        total_loss = abs(sum(actual_loss))

        # Plotting the training results
        function plot_results(t, real, pred)
            plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised ANODE-MS II Model", xlabel = "Time", ylabel = "Population")
            scatter!(t, real[1,:], label = "Training Data")
            plot!(legend=:topright)
            savefig("full-sim-LV-ANODE-MS-II/Plots/Simulation $i.png")
        end

        plot_results(t, Xₙ, full_traj2)

        # Printing when the simulation is finished to make it easier to track the progress
        if i==iters
            println("Simulation finished")
            break 
        end
   
    end
end
