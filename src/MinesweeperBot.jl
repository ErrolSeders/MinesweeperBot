module MinesweeperBot
    include("DDQNAgent.jl")
    export train_loop, HyperParams, GameParams, decayparam, extend_dims, ϵ_greedy, get_state, takeaction!
end