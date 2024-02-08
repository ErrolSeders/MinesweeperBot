import Base: push!


mutable struct Experience
    state::AbstractArray
    action::CartesianIndex{2}
    reward::Float64
    next_state::AbstractArray
    done::Bool
end

mutable struct ReplayBuffer
    buffer::Vector{Experience}
    capacity::Int
    ReplayBuffer(capacity) = new(Experience[], capacity)
end

function push!(rb::ReplayBuffer, exp::Experience)
    if length(rb.buffer) < rb.capacity
        Base.push!(rb.buffer, exp)
    else 
        rb.buffer[1:end-1] .= rb.buffer[2:end]
        rb.buffer[end] = exp
    end
end

function samplebatch(rb::ReplayBuffer, batch_size::Int)
    idxs = rand(1:length(rb.buffer), batch_size)
    return rb.buffer[idxs]
    
    # (state = [exp.state for exp ∈ batch], 
    #         action = [exp.action for exp ∈ batch], 
    #         reward = [exp.reward for exp ∈ batch], 
    #         next_state = [exp.next_state for exp ∈ batch], 
    #         done = [exp.done for exp ∈ batch])
end
