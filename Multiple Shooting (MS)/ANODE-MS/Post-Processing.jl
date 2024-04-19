using CSV, FilePathsBase, DataFrames, Plots, Statistics, PlotlyBase, PlotlyKaleido
plotly()
#gr()

# Reading the loss data into a DataFrame
#folder_path_ANODE = "Multiple Shooting (MS)/ANODE-MS/Case Studies/Loss Data/ANODE-MS"
folder_path_ANODE = "/Users/junetroan/Desktop/Thesis/Archive/ANODE_NODE 2/Loss Data"
file_names_ANODE = readdir(folder_path_ANODE)
file_names_ANODE = file_names_ANODE[1:500]

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
end#

df_ANODE = DataFrame()

folder_path_NODE = "/Users/junetroan/Desktop/Thesis/Archive/ANODE_NODE 2/Loss Data2"
file_names_NODE = readdir(folder_path_NODE)
df_NODE = DataFrame()


folder_path_PEM = "/Users/junetroan/Desktop/Results/files Ks/Loss Data Ks"
file_names_PEM = readdir(folder_path_PEM)
PEM_files = file_names_PEM[501:1000]
df_PEM = DataFrame()

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
reader(PEM_files, folder_path_PEM, df_PEM)

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
plotter(df_PEM)

# Averaging the losses
function averager(df)
    row_means = mean.(eachrow(df))
    row_stds = std.(eachrow(df))    
    p = plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title="Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
    p2 = plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title="Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10)
    #savefig(p, "Multiple Shooting (MS)/ANODE-MS/Simulations ANODE-MS Loss Evolution.png")
    #savefig(p2, "Multiple Shooting (MS)/ANODE-MS/Simulations ANODE-MS Loss Evolution (Log Scale).png")
    savefig(p, "Multiple Shooting (MS)/ANODE-MS/ANODE-MS Loss Evolution.png")
end

averager(df_ANODE)
averager(df_NODE)
averager(df_PEM)

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
        plot!(p, 1:nrow(df2), row_means_2, ribbon=row_stds_2, label="NPEM")
    
    #p2 = plot(1:nrow(df1), log_means_1, ribbon=log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss")
    p3 = plot(1:nrow(df1), row_means_1, ribbon = log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10)
    plot!(p3, 1:nrow(df2), row_means_2, ribbon = log_stds_2, label="NPEM")
    #plot!(p2, 1:nrow(df2), log_means_2, ribbon=log_stds_2, label="PEM")
    
    #display(p)
    #display(p2)
    #display(p3)
    
    #savefig(p, "Multiple Shooting (MS)/ANODE-MS/Simulations/Comparisons/LV-MS/Comparison ANODE-MS vs. PEM.png")
    #p3 = plot(1:nrow(df1), row_means_1, ribbon = log_stds_1, label="ANODE-MS", title="Comparison of Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10, size=(800,600))
    #plot!(p3, 1:nrow(df2), row_means_2, ribbon = log_stds_2, label="PEM")
    
    savefig(p, "Prediction Error Methods (PEM)/Comparison ANODE-MS vs. NPEM LV.png")
end

compare(df_ANODE, df_NODE)


path_Ks = "/Users/junetroan/Desktop/Results/files Ks/Loss Data Ks"
file_names_Ks = readdir(path_Ks)
files = file_names_Ks[1:500]
df_Ks = DataFrame()

reader(files, path_Ks, df_Ks)

matrix_Ks = Matrix(df_Ks)


heat = heatmap(matrix_Ks, 
        title = "Heatmap of K values over 500 Simulations",
        xlabel = "Iterations",
        ylabel = "Simulations",
        color = :plasma,
        aspect_ratio = :auto)

savefig(heat, "Multiple Shooting (MS)/ANODE-MS/Case Studies/Heatmap of K values.png")

final_Ks = matrix_Ks[500, :]

hist = histogram(final_Ks, 
    title = "Histogram of K values after 500 Simulations",
    xlabel = "K values",
    ylabel = "Frequency",
    color = :orange,
    legend = false,
    bins = 20)


savefig(hist, "Multiple Shooting (MS)/ANODE-MS/Case Studies/Histogram of K values.png")