# Ran on the 16th of February 
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
using DataInterpolations
#using OrdinaryDiffEq
using OptimizationPolyalgorithms
using DiffEqFlux
using Plots
gr()

# Producing data
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

rng = StableRNG(1)
u0_lv = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
tspan = (0.0f0, 10.0f0)
prob = ODEProblem(lotka!, u0_lv, tspan, p_)
solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
X = Array(solution)
t = solution.t
tspan = (t[1], t[end])
tsteps = range(tspan[1], tspan[2], length = length(X[1,:]))

# Adding noise
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
y = Xₙ[1,:]

# Interpolation of given data
y_zoh = LinearInterpolation(y, tsteps) #changed as it is way faster
iters = 1
state = 2

fulltraj_losses = Float32[]

@time begin
    for i in 1:iters

        println("Simulation $i")

        # Random initialisation
        rng1 = StableRNG(i+1)
        rng2 = StableRNG(i+2)
        u0 = [y[1], rand(rng1, Float32, state-1)...]
        K = rand(rng2, Float32, state)

        # Neural network definition
        U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
        p, st = Lux.setup(rng1, U)

        # Definition of the model
        function predictor!(du,u,p,t)
            û = U(u, p.vector_field_model, st)[1]
            yt = y_zoh(t)
            e = yt .- û[1]
            du[1:end] =  û[1:end] .+ abs.(p.K) .* e
        end
        
        params = ComponentVector{Float32}(vector_field_model = p, K = K)
        prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat=tsteps)
        soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat = 0.25f0))
        
        function prediction(p)
            _prob = remake(prob_nn, u0 = u0, p = p)
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
            Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-6, reltol = 1e-6, saveat = 0.25f0, sensealg = sensealg))
        end

        function predloss(p)
            yh = prediction(p)
            e2 = mean(abs2, y .- yh[1,:])
            return e2
        end

        losses = Float32[]
        Ks = []
        
        callback = function (θ, l)
            push!(losses, predloss(θ))
            push!(Ks, θ.K[1:end])

            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, params)
        res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback)

        losses_df = DataFrame(losses = losses)
        Ks_mat =  mapreduce(permutedims, vcat, Ks)
        Ks_df = DataFrame(Ks_mat, :auto)
        
        #CSV.write("Loss Data Ks/Ks $i.csv", Ks_df, writeheader = false)
        #CSV.write("Loss Data Ks/Losses $i.csv", losses_df, writeheader = false)

        res_ms.u.K = zeros(Float32, state)
        full_traj = prediction(res_ms.u)
        full_traj_loss = predloss(res_ms.u)
        push!(fulltraj_losses, full_traj_loss)

        function plot_results(t, real, pred)
            plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised PEM Model", xlabel = "Time", ylabel = "Population")
            scatter!(t, real[1,:], label = "Training Data")
            plot!(legend=:topright)
            #savefig("Plots Ks/Simulation $i.png")
        end

        plot_results(t, Xₙ, full_traj)

        if i==iters
            println("Simulation finished")
            break 
        end


    end

end