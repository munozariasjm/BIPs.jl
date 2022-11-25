using Pkg
Pkg.activate(".")
Pkg.add("HDF5")
using BIPs
using HDF5
using Statistics
using Pkg.Artifacts

dataset_path = "../../DataLake/raw"
# dataset_path = "/User/home/josemm/MyDocs/personal_BIP/BIPs.jl/exampless/ortner/datasets/toptagging"

train_data_path = dataset_path * "/train.h5"
val_data_path = dataset_path * "/val.h5"

println("Loading data train...")
train_jets, train_labels = BIPs.read_data("TQ", train_data_path)
train_labels = [reinterpret(Bool, b == 1.0) for b in train_labels]
print("Number of entries in the training data: ", length(train_jets))

val_jets, val_labels = BIPs.read_data("TQ", val_data_path)
val_labels = [reinterpret(Bool, b == 1.0) for b in val_labels]
print("Number of entries in the validation data: ", length(val_jets))

train_transf_jets = data2basis(train_jets; basis="hyp")
val_transf_jets = data2basis(val_jets; basis="hyp")
println("Transformed jets")

println("Generating basis...")
f_bip, order, a_basis = build_ip(order=4, levels=7)

function bip_data(dataset_jets)
    storage = zeros(length(dataset_jets), length(order))
    for i = 1:length(dataset_jets)
        storage[i, :] = f_bip(dataset_jets[i])
    end
    storage[:, 2:end]
end


train_embedded_jets = bip_data(train_transf_jets)
println("Embedded train jets correclty")
val_embedded_jets = bip_data(val_transf_jets)
println("Embedded test jets correclty")


h5write("../train_basis_4_7.csv", "table", train_embedded_jets)

h5write("../val_basis_4_7.csv", "table", val_embedded_jets)

test_data_path = "../../DataLake/raw/test.h5"
test_jets, test_labels = BIPs.read_data("TQ", test_data_path)
test_labels = [reinterpret(Bool, b == 1.0) for b in test_labels]
test_transf_jets = data2basis(test_jets; basis="hyp")
test_embedded_jets = bip_data(test_transf_jets)
print("Embedded test jets correclty")
test_jets, test_labels = BIPs.read_data("TQ", test_data_path)
test_labels = [reinterpret(Bool, b == 1.0) for b in test_labels]
test_transf_jets = data2basis(test_jets; basis="hyp")
test_embedded_jets = bip_data(test_transf_jets)
print("Embedded test jets correclty")
h5write("../test_basis_4_7.csv", "table", test_embedded_jets)

h5write("../train_labels.csv", train_labels)
h5write("../val_labels.csv", val_labels)
h5write("../test_labels.csv", test_labels)

println("Done")
