using Parameters: @with_kw
using Flux 
using BSON
using Dates
using Logging

include("ReplayBuffer.jl")
include("MinesweeperEnv.jl")
include("Model.jl")

@with_kw mutable struct HyperParams
    episodes::Integer = 1000
    batchsize::Integer = 32
    replaybuffercapacity::Integer = 10000
    updatefreq::Integer = 100
    ϵ::Real = 0.95 
    ϵ_min::Real = 0.001
    ϵ_decay::Real = 0.99975
    γ::Real = 0.1
    α::Real = 0.01
    α_min::Real = 0.001
    α_decay::Real = 0.99975
    rewards::Dict{String, Real} = Dict("Loss" => -1.0, "Progress" => 0.3, "Win" => 1.0)
    unrevealed_value::Int = 9
end

decayparam = (p,pd,pm) -> p > pm ? p*pd : p

extend_dims(x::AbstractArray) = reshape(x, (size(x)...,1,1))

@with_kw struct GameParams 
    board_dim::Tuple{Integer, Integer} = (16,16)
    mines::Integer = 40
end

function ϵ_greedy(model, state, ϵ)
    valid = findall(x -> x == 9, state)
    if rand() < ϵ
        return valid[rand(1:end)]
    else
        q_vals = model(state |> extend_dims)
        q_vals[:,:,1,1][valid] .= -Inf32
        return argmax(q_vals[:,:,1,1])
    end
end

function get_state(game::MinesweeperGame, unrevealed_value::Int)
    return (game.board .* game.revealed_mask) .+ (.~game.revealed_mask .* unrevealed_value)
end

function takeaction!(first_move::Bool, game::MinesweeperGame, action::CartesianIndex)
    game.game_over = first_move ? first_move!(game, action) : move!(game, action)
    if check_win(game) 
        reward = hp.rewards["Win"]
        game.game_over = true
    else
        reward = game.game_over ? hp.rewards["Progress"] : hp.rewards["Loss"]
    end
    return reward
end

function train_loop(hp:: HyperParams, gp:: GameParams)
    rb = ReplayBuffer(hp.replaybuffercapacity)
    q_online = QNetworkFCN()
    q_target = QNetworkFCN()

    try 
        Flux.loadparams!(q_online, "model.bson")
        Flux.loadparams!(q_target, "model.bson")
    catch e
        @info e.msg
        @info "Unable to find model, training from scratch"
    end

    opt_state = Flux.setup(Adam(hp.α), q_online)

    for episode in 1:hp.episodes

        game = MinesweeperGame(gp.board_dim, gp.mines)
        first_move = true
        total_reward = 0.0
        steps = 0 

        while !game.game_over

            state = get_state(game, hp.unrevealed_value)
            action = ϵ_greedy(q_online, state, hp.ϵ)
            reward = takeaction!(first_move, game, action)
            next_state = get_state(game, hp.unrevealed_value)
            exp = Experience(state, action, reward, next_state, game.game_over)

            push!(rb, exp)
            total_reward += reward
            steps += 1

            if length(rb.buffer) > hp.batchsize

                batch = samplebatch(rb, hp.batchsize)
                for exp ∈ batch 

                    state = exp.state |> extend_dims
                    action = exp.action
                    reward = exp.reward
                    next_state = exp.next_state |> extend_dims
                    done = exp.done

                    if episode % hp.updatefreq == 0
                        @info state
                        @info next_state
                        @info reward
                    end

                    next_qvals = q_target(next_state)
                    TD_targets = (reward + hp.γ * (1-done)) .* next_qvals

                    grads = gradient(q_online) do q
                        qvals = q(state)
                        Flux.mse(qvals, TD_targets) # loss
                    end

                    Flux.update!(opt_state, q_online, grads[1])
                end
            end
        end

        if episode % hp.updatefreq == 0
            Flux.loadparams!(q_target, params(q_online))
        end

        hp.ϵ = decayparam(hp.ϵ, hp.ϵ_decay, hp.ϵ_min)
        hp.α = decayparam(hp.α, hp.α_decay, hp.α_min)
        Flux.adjust!(opt_state, hp.α)
        @info "Episode $episode : Total Reward $total_reward : Exploration Rate $(hp.ϵ)"
    end

    now = Dates.now()
    fn_online = "q_online_$(Dates.format(now, "yyyymmdd_HHMMSS")).bson"
    fn_target = "q_target_$(Dates.format(now, "yyyymmdd_HHMMSS")).bson"

    current_online = Flux.params(q_online)
    current_target = Flux.params(q_target)

    BSON.@save fn_online current_online
    BSON.@save fn_target current_target
end

begin 
    hp = HyperParams()
    gp = GameParams()
    train_loop(hp, gp)
end