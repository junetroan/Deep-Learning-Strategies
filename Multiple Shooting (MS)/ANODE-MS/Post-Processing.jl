using CSV, FilePathsBase, DataFrames, Plots, Statistics
#plotly()
gr()

# Reading the loss data into a DataFrame
folder_path_ANODE = "Multiple Shooting (MS)/ANODE-MS/Case Studies/Loss Data/ANODE-MS"
file_names_ANODE = readdir(folder_path_ANODE)

missing_files = []
for i in 1:500
    file_name = "Losses $i.csv"
    if !(file_name in file_names_ANODE)
        push!(missing_files, file_name)
    end
end

if isempty(missing_files)
    println("No files are missing.")
else
    println("The following files are missing:")
    println(missing_files)
end

df_ANODE = DataFrame()

folder_path_NODE = "Multiple Shooting (MS)/ANODE-MS/Case Studies/Loss Data/PEM"
file_names_NODE = readdir(folder_path_NODE)
df_NODE = DataFrame()

function reader(file_names, folder_path, df)  
    for file in file_names
        full_path = joinpath(folder_path, file)
        col_data = CSV.read(full_path, DataFrame)
        col_name = basename(file)
        df[!, Symbol(col_name)] = col_data[!, :1] #Note! All the files need to have the same amount of data points
    end
end

reader(file_names_ANODE, folder_path_ANODE, df_ANODE)
reader(file_names_NODE, folder_path_NODE, df_NODE)

# Plotting the loss evolution
function plotter(df)
    p = plot(xlabel="Iterations", ylabel="Loss", title="Loss Evolution")
    
    simulation_counter = 1

    for col_name = names(df)
        label_name = "Simulation " *string(simulation_counter)
        plot!(p, df[!, Symbol(col_name)], label = label_name)

        simulation_counter += 1
    end

    display(p)
end

plotter(df_ANODE)
plotter(df_NODE)


# Averaging the losses
function averager(df)
    row_means = mean.(eachrow(df))
    row_stds = std.(eachrow(df))    
    p = plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title="Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
    #savefig(p2, "Simulations/ANODE-MS/Results/ANODE-MS Loss Evolution (LR1e-4 8000iter).png")
    display(p)
end

averager(df_ANODE)
averager(df_NODE)

function compare(df1, df2)

    row_means_1 = mean.(eachrow(df1))
    row_means_2 = mean.(eachrow(df2))

    row_stds_1 = std.(eachrow(df1))    
    row_stds_2 = std.(eachrow(df2))

    log_means_1 = log.(row_means_1)
    log_means_2 = log.(row_means_2)

    log_stds_1 = log.(row_stds_1)
    log_stds_2 = log.(row_stds_2)

    p = plot(1:nrow(df1), row_means_1, ribbon=row_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
        plot!(p, 1:nrow(df2), row_means_2, ribbon=row_stds_2, label="NODE-MS")
    
    #p2 = plot(1:nrow(df1), log_means_1, ribbon=log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
    p3 = plot(1:nrow(df1), row_means_1, ribbon = log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10)
    plot!(p3, 1:nrow(df2), row_means_2, ribbon = log_stds_2, label="PEM")
    #plot!(p2, 1:nrow(df2), log_means_2, ribbon=log_stds_2, label="PEM")
    
    #display(p)
    #display(p2)
    #display(p3)
    
    #savefig(p, "Multiple Shooting (MS)/ANODE-MS/Simulations/Comparisons/LV-MS/Comparison ANODE-MS vs. PEM.png")
    #p3 = plot(1:nrow(df1), row_means_1, ribbon = log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10, size=(800,600))
    #plot!(p3, 1:nrow(df2), row_means_2, ribbon = log_stds_2, label="PEM")
    
    savefig(p, "Multiple Shooting (MS)/ANODE-MS/Case Studies/Comparison ANODE-MS vs. PEM (FinCompSys).png")
end

compare(df_ANODE, df_NODE)
