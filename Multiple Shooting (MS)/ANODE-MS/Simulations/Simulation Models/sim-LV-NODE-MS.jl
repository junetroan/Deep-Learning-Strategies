using Pkg
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

function trueODEfunc(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

rng = StableRNG(1111)
group_size = 5
tsteps = 0.25f0
datasize = 41
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
noise_magnitude = 10f-3
continuity_term = 10.0f0
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob_trueode = ODEProblem(trueODEfunc, u0, tspan, p_)
solution = solve(prob_trueode, AutoVern7(KenCarp4()), abstol = 1e-10, reltol = 1e-8, saveat = 0.25e0)
state = 2

X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

fulltraj_losses = Float32[]


iters = 2
##################################################
@time begin
    for i in 1:iters

        println("Simulation $i")

        # Random numbers
        rng_1 = StableRNG(i)
        rng_2 = StableRNG(i + 1)

        # Define the Neural Network
        nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
        p_init, st = Lux.setup(rng_1, nn)

        params = ComponentVector{Float32}(ANN = p_init, u0_states = 0.5*randn(rng_2, Float32, state - 1))

        neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps,
         abstol = 1f-6, reltol = 1f-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        
        function rhs(du, u, p, t)
            du[1:end] = nn(u, p.ANN, st)[1]
        end
        
        prob_node = ODEProblem(rhs, [u0[1]; params.u0_states], tspan, params, saveat = tsteps)

        solve(prob_node, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6)

        function loss_function(data, pred)
            return mean(abs2, data[1, :] - pred[1, :])
        end

        function continuity_loss_function(u_end, u_0)
            return mean(abs2, u_end - u_0)
        end

        function loss_multiple_shooting(p)
            new_prob = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
            return multiple_shoot(p, Xₙ, tsteps, new_prob, loss_function, continuity_loss_function, AutoTsit5(Rosenbrock23(autodiff = false)),
                                group_size; continuity_term, abstol = 1f-6, reltol = 1f-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        end
        
        function predict_final(p)
            prob_new = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
            return Array(solve(prob_new, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6))[1, :]
        end

        function final_loss(θ)
            X̂ = predict_final(θ)
            prediction_error = mean(abs2, Xₙ[1, :] .- X̂)
            prediction_error
        end


        function predict_single_shooting(p)
            prob_new = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
            return Array(solve(prob_new, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6))[1, :]
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
        optprob = Optimization.OptimizationProblem(optf, params)
        @time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000; callback = callback)

        losses_df = DataFrame(loss = losses)
        #CSV.write("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-LV-NODE-MS.jl/Loss Data/Losses $i.csv", losses_df, writeheader = false)

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
            plot(t, pred, label = "Training Prediction", title="Iteration $i of Randomised NODE-MS Model", xlabel="Time", ylabel="Population")
            scatter!(t, real[1,:], label = "Training Data")
            plot!(legend=:topright)
            #savefig("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-LV-NODE-MS.jl/Plots/Simulation $i.png")
        end

        plot_results(Xₙ, full_traj)

        if i==iters
            println("Simulation finished")
            break 
        end

    end
end