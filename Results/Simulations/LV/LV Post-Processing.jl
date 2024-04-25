using CSV, FilePathsBase, DataFrames, Plots, Statistics
#plotly()
gr()


################################################################################################################################################

# Reading the loss data into a DataFrame

################################################################################################################################################

# ANODE-MS
folder_path_ANODE = "/Users/junetroan/Desktop/data-ANODEMS-LV-correct/Loss Data"
file_names_ANODE = readdir(folder_path_ANODE)
file_names_ANODE = file_names_ANODE[1:500]

missing_files_ANODE = []
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

# MNODE - MS
folder_path_NODE = "/Users/junetroan/Desktop/data-NODEMS-LV/Loss Data2"
file_names_NODE = readdir(folder_path_NODE)

missing_files_NODE = []
for i in 1:500
    file_name = "Losses $i.csv"
    if !(file_name in file_names_NODE)
        push!(missing_files, file_name)
    end
end

if isempty(missing_files_NODE)
    println("No files are missing.")
else
    println("The following files are missing:")
    println(missing_files_NODE)
end

df_NODE = DataFrame()

# NPEM
folder_path_PEM = "/Users/junetroan/Desktop/Results/files Ks/Loss Data Ks"
file_names_PEM = readdir(folder_path_PEM)
file_names_PEM = file_names_PEM[501:1000]


missing_files_PEM = []
for i in 1:500
    file_name = "Losses $i.csv"
    if !(file_name in file_names_PEM)
        push!(missing_files, file_name)
    end
end

if isempty(missing_files_PEM)
    println("No files are missing.")
else
    println("The following files are missing:")
    println(missing_files_PEM)
end

df_PEM = DataFrame()

################################################################################################################################################

# Functions

################################################################################################################################################
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
reader(file_names_PEM, folder_path_PEM, df_PEM)

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

    #p = plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title="Loss Evolution of ANODE-MS Model", xlabel="Iteration", ylabel="Average Loss")
    p1 = plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title="Loss Evolution", xlabel="Iteration", ylabel="Average Loss", yscale=:log10)
    #savefig(p, "Results/Simulations/LV/Simulations ANODE-MS Loss Evolution.png")
    #savefig(p1, "Results/Simulations/LV/Simulations ANODE-MS Loss Evolution (Log Scale).png")
    #savefig(p, "Results/Simulations/LV/Simulations MNODE-MS Loss Evolution.png")
    savefig(p1, "Results/Simulations/LV/Simulation of MNODE-MS Loss Evolution (Log Scale).png")
    #savefig(p, "Results/Simulations/LV/Simulation of NPEM Loss Evolution")
    #savefig(p1, "Results/Simulations/LV/Simulation of NPEM Loss Evolution (Log Scale)")
end

#averager(df_ANODE)
averager(df_NODE)
#averager(df_PEM)

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