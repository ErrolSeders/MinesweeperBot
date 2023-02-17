using Minesweeper
using Parameters: @with_kw
using Flux 
using BSON
using Dates

@with_kw mutable struct HyperParams
    episodes::Integer = 10000
    batchsize::Integer = 64 
    replaybuffercapacity::Integer = 10000
    updatefreq::Integer = 1000
    ϵ::Real = 0.95 
    ϵ_min::Real = 0.001
    ϵ_decay::Real = 0.99975
    γ::Real = 0.1
    α::Real = 0.01
    α_min::Real = 0.001
    α_decay::Real = 0.99975
    rewards::Dict{String, Real} = Dict("Loss" => -1.0, "Guess" => -0.3, "Progress" => 0.3, "Win" => 1.0)
end

decayparam = (p,pd,pm) -> p > pm ? p = p*pd : Nothing

@with_kw struct GameParams 
    board_dim::Tuple{Integer, Integer} = (16,16)
    mines::Integer = 40
end

struct ModelParams 
    board_dim::Tuple{Integer, Integer}
    droprate::Real     
end

function q_net(mp::ModelParams)

    total_dim = foldl(*, mp.board_dim)

    layers = [
        Conv((3,3), 1 => 32, pad = (1,1), relu),
        Conv((3,3), 32 => 64, pad = (1,1), relu),
        Conv((3,3), 64 => 64, pad = (1,1), relu),
        Flux.flatten,
        Dense((total_dim * 64), 512, relu),
        Dense(512, total_dim)
    ]

    return Chain(layers...)

end

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

function sample(rb::ReplayBuffer, batch_size::Int)
    idxs = rand(1:length(rb.buffer), batch_size)
    return rb.buffer[idxs]
    
    # (state = [exp.state for exp ∈ batch], 
    #         action = [exp.action for exp ∈ batch], 
    #         reward = [exp.reward for exp ∈ batch], 
    #         next_state = [exp.next_state for exp ∈ batch], 
    #         done = [exp.done for exp ∈ batch])
end

function getboardstate(minegame::Game)
    revealed = [cell.revealed for cell ∈ minegame.cells]
    mine_counts = [cell.mine_count for cell ∈ minegame.cells]
    board_state = zeros(size(mine_counts)) .|> Int
    for (i,x) ∈ enumerate(revealed)
        board_state[i] = revealed[i] == 1 ? mine_counts[i] : -1
    end

    return board_state
end

function checkguess(minegame::Game, action::CartesianIndex)::Bool
    neighbors = get_neighbors(minegame,action)
    return foldl( | , [cell.revealed for cell ∈ neighbors])
end

function takeaction(minegame::Game, action::CartesianIndex)
    select_cell!(minegame,(action |> tuple)...)
    
    if game_over(minegame)
        reward = "Loss" 
        done = true
    else
        done = false
        if checkguess(minegame, action)
            reward = "Guess"
        end
        reward = "Progress"
    end
    return reward, getboardstate(minegame), done

end

function train_loop(mp::ModelParams, hp::HyperParams, gp::GameParams)
    q_online = q_net(mp)
    q_target = q_net(mp)

    num_actions = foldl(*, gp.board_dim)

    
    opt_state = Flux.setup(ADAM(hp.α), q_online)
    lossfn = Flux.mse

    rb = ReplayBuffer(hp.replaybuffercapacity)

    for episode ∈ 1:hp.episodes
        
        minegame = Game(dims=gp.board_dim, n_mines=gp.mines)

        total_reward = 0.0
        steps = 0

        playing = true

        while playing

            #prepare boardstate
            board_state = getboardstate(minegame)

            unsolved = [i for (i,x) ∈ pairs(board_state) if x == -1]
        
            if length(unsolved) == gp.mines

                reward = "Win"
                playing = false
                done = true
                next_state = board_stat

            else

                if rand() < 0
                    action = (rand(1:mp.board_dim[1]), rand(1:mp.board_dim[2])) |> CartesianIndex
                else
                    q_vals = q_online(reshape(board_state,(mp.board_dim...,1,1)))
                    action = reshape(q_vals, (mp.board_dim...)) |> argmax
                end

                reward, next_state, done = takeaction(minegame, action)

            reward_val = hp.rewards[reward]
            exp = Experience(board_state, action, reward_val, next_state, done)

            push!(rb, exp)

            total_reward += reward_val
            steps += 1

            batch = sample(rb, hp.batchsize)
            
            # states = batch.state
            # println(size(states))
            # action = batch.action
            # rewards = batch.reward 
            # next_states = batch.next_state
            # dones = batch.done

            for exp ∈ batch
            
                state = exp.state
                state = reshape(state,(size(state)...,1,1))
                
                next_state = exp.next_state
                next_state = reshape(next_state,(size(next_state)...,1,1))
                
                action = exp.action
                reward = exp.reward
                done = exp.done
                
                next_q_vals = q_target(next_state)
                TD_targets = (reward + hp.γ * (1 - done)) .* next_q_vals
                
                grads = gradient(q_online) do q
                    q_vals = q(state)
                    loss = lossfn(q_vals, TD_targets)
                end
                
                Flux.update!(opt_state, q_online, grads[1])
            
            end

            #update target Q-network
            if steps % hp.updatefreq == 0
                Flux.loadparams!(q_target, Flux.params(q_online))
            end

            #Decay ϵ and α 
            decayparam(hp.ϵ,hp.ϵ_decay,hp.ϵ_min)
            decayparam(hp.α,hp.α_decay,hp.α_min)
            
            Flux.adjust!(opt_state, hp.α)

            end
        end
        println("Episode $episode : Total Reward $total_reward : Exploration Rate $(hp.ϵ)")
    end

    #save models after training
    now = Dates.now()
    fn_online = "q_online_$(Dates.format(now, "yyyymmdd_HHMMSS")).bson"
    fn_target = "q_target_$(Dates.format(now, "yyyymmdd_HHMMSS")).bson"

    current_online = Flux.params(q_online)
    current_target = Flux.params(q_target)

    BSON.@save fn_online current_online
    BSON.@save fn_target current_target

end