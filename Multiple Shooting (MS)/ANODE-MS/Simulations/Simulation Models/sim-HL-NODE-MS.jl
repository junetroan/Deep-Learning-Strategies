# sim-HL-NODE-MS.jl
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, OptimizationPolyalgorithms, Plots
using DiffEqFlux
using DiffEqFlux: group_ranges
using StableRNGs
using Random
using CSV, Tables, DataFrames
using StatsBase

function trueODEfunc(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end
data_path = "Data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)
rng = StableRNG(1111)
#Train/test Splits
split_ration = 0.75
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
hare_data = data[:, 2]
lynx_data = data[:, 3]

transformer = fit(ZScoreTransform, hare_data)
X_train = Float32.(StatsBase.transform(transformer, train[:, 2]))
X_test = Float32.(StatsBase.transform(transformer, test[:, 2]))
X_new = reshape(X_train, 1, :)
unknown_X = zeros(Float32, 1, 68)
X = vcat(X_new, unknown_X)



t = Float32.(collect(1:size(data, 1)))
t_train = Float32.(collect(1:Int(round(split_ration*size(data, 1)))))
t_test = Float32.(collect(Int(round(split_ration*size(data, 1))):size(data, 1)))

# Define the experimental parameter

group_size = 2
state = 2
iters = 2
tspan = (Float32(minimum(t_train)), Float32(maximum(t_train)))
datasize = length(X_train)
tsteps = range(tspan[1], tspan[2], length = datasize)
continuity_term = 10.0f0
u0 = Float32[X_train[1], 0.0f0]

fulltraj_losses = Float32[]

@time begin
    for i in 1:iters
        
        println("Simulation $i")

        # Random numbers
        rng_i = StableRNG(i)

        # Define the Neural Network
        nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
        p_init, st = Lux.setup(rng_i, nn)

        neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), abstol = 1f-6, reltol = 1f-6)
        prob_node = ODEProblem((u,p,t) -> nn(u,p,st)[1], [[u0[1];zeros(state-1)]], tspan, ComponentArray(p_init))

        function loss_function(data, pred)
            return mean(abs2, data - pred)
        end

        function continuity_loss_function(u_end, u_0)
            return mean(abs2, u_end - u_0)
        end
        
        function loss_multiple_shooting(p)
            return multiple_shoot(p, X, tsteps, prob_node, loss_function,  continuity_loss_function, AutoTsit5(Rosenbrock23(autodiff=false)),
                                  group_size; continuity_term, abstol = 1f-6, reltol = 1f-6)
        end
        
        function predict_final(θ)
            return Array(neuralode([u0[1]; zeros(state -1)], θ, st)[1])
        end
        
        function final_loss(θ)
            X̂ = predict_final(θ)
            prediction_error = mean(abs2, X[1,:] .- X̂[1, :])
            prediction_error
        end
        
        function predict_single_shooting(p)
            return Array(neuralode([[u0[1];zeros(state-1)]], p, st)[1])
        end
        

        losses = Float32[]
        callback = function (p, l, preds; doplot = false)
            push!(losses, final_loss(p))
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
        
            end
            return false
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
        @time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000; callback = callback)

        losses_df = DataFrame(loss = losses)
        CSV.write("Simulations/Results/sim-HL-NODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

        # Evaluate Single Shooting
        function loss_single_shooting(p)
            pred = predict_single_shooting(p)
            l = loss_function(Xₙ, pred)
            return l, pred
        end

        full_traj = predict_final(res_ms.u)
        full_traj_loss = final_loss(res_ms.u)
        push!(fulltraj_losses, full_traj_loss)

        function plot_results(real, pred)
            plot(t_train, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised NODE-MS Model", xlabel="Time", ylabel="Population")
            scatter!(t_train, real[1,:], label = "Training Data")
            plot!(legend=:topright)
            savefig("Simulations/Results/sim-HL-NODE-MS/Plots/Simulation $i.png")
        end

        plot_results(X, full_traj)

        if i==iters
            println("Simulation finished")
            break 
        end
    end
end
