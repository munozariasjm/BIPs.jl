
module DatasetGenerator

include("./reader.jl")
include("./transforms.jl")


using Statistics, LinearAlgebra, StaticArrays

using .DataReader, .DataTransformer

function read_data(dataset::String, filename::String; n_limit::Int=-1)::Tuple
    if dataset == "TQ"
        four_momentum, labels = read_TQ(filename; n_limit=n_limit)
    elseif dataset == "RecoTQ"
        four_momentum, labels = read_RecoTQ(filename; n_limit=n_limit)
    else
        error("Dataset not found")
    end
    four_momentum, labels
end


export read_data, transform2hyp, data2basis

end