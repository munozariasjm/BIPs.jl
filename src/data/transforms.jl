module DataTransformer
using Statistics, LinearAlgebra, StaticArrays


function transform2hyp!(tjet, jet::Vector{<:SVector}; ϵ=1e-10)

    # Begin the projection to the jet axis
    pbar4 = mean(jet)
    pbar3 = SVector(pbar4[2], pbar4[3], pbar4[4])
    pbar_norm = pbar3 / norm(pbar3)
    v = pbar_norm - SVector(0.0, 0.0, 1.0)
    norm_v = norm(v)
    if !isapprox(norm_v, 0.0)
        v = v / norm_v
    end
    function transform(Ep, v)
        E = Ep[1] # Energy
        p = SVector(Ep[2], Ep[3], Ep[4]) # Cartesian Momentum in the lab frame
        xyz = p - 2 * dot(p, v) * v # Projection
        z = xyz[3] # Parallel momentum in jet axis
        tM = log(E^2 - z^2 + 1)
        y = 0.5 * log((E + z + ϵ) / (E - z + ϵ)) # Rapidity
        r = sqrt(xyz[1]^2 + xyz[2]^2) + ϵ # Transverse momentum
        cosθ, sinθ = xyz[1] / r, xyz[2] / r # Angle in arbitrary plane
        return SVector(r, cosθ, sinθ, y, tM) # Return the transformed particle
    end
    for i = 1:length(jet)
        @inbounds tjet[i] = transform(jet[i], v)
    end
    sum_pt = sum(x[1] for x in tjet)
    # this is a little hack to make the SVectors writable
    Vec_tjet = reinterpret(Float64, tjet)
    for i in 1:5:length(Vec_tjet)
        @inbounds Vec_tjet[i] /= sum_pt
    end
    return tjet
end

transform2hyp(jet::Vector{<:SVector}; ϵ=1e-4) =
    transform2hyp!(Vector{SVector{5,Float64}}(undef, length(jet)),
        jet; ϵ=ϵ)


function transform2lab!(tjet, jet::Vector{<:SVector}; ϵ=1e-10)

    # We have the four-momentum of the jet in the lab frame
    # pbar4 = mean(jet)

    function transform_to_lab(p4)
        E = p4[1] # Energy
        pz = p4[4] # Parallel momentum in jet axis
        tM = log(E^2 - pz^2 + 1)
        y = 0.5 * log((E + pz + ϵ) / (E - pz + ϵ)) # Rapidity
        pt = sqrt(p4[2]^2 + p4[3]^2) + ϵ # Transverse momentum
        cosθ, sinθ = p4[2] / pt, p4[3] / pt # Angle in arbitrary plane
        return SVector(pt, cosθ, sinθ, y, tM) # Return the transformed particle
    end
    for i = 1:length(jet)
        @inbounds tjet[i] = transform_to_lab(jet[i])
    end
    sum_pt = sum(x[1] for x in tjet)
    # this is a little hack to make the SVectors writable
    Vec_tjet = reinterpret(Float64, tjet)
    for i in 1:5:length(Vec_tjet)
        @inbounds Vec_tjet[i] /= sum_pt
    end
    return tjet
end


transform2lab(jet::Vector{<:SVector}; ϵ=1e-4) =
    transform2lab!(Vector{SVector{5,Float64}}(undef, length(jet)),
        jet; ϵ=ϵ)
"""
`data2basis`: Transforms the readed dataset into the jet coordinates.
Args:
* dataset_jets (Vector{Vector{<:SVector}}):  Datat readded from a file using `read_data`.
Returns:
* Vector{Vector{<:SVector}}:  The transformed dataset.

"""
function data2basis(dataset_jets; basis="hyp")
    storage = Vector{SVector}[]
    if basis == "hyp"
        for i = eachindex(dataset_jets)
            push!(storage, transform2hyp(dataset_jets[i]))
        end
    else # basis == "lab"
        for i = eachindex(dataset_jets)
            push!(storage, transform2lab(dataset_jets[i]))
        end
    end

    storage
end

export transform2hyp, data2basis


end