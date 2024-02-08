
using Pipe: @pipe
using StatsBase: sample

mutable struct MinesweeperGame 
    board::Array{Int,2}
    revealed_mask::Array{Bool,2}
    dims::Tuple{Int,Int}
    n_mines::Int
    n_revealed::Int
    game_over::Bool

    function MinesweeperGame(dims::Tuple{Int,Int}, n_mines::Int)
        board = zeros(Int, dims...)
        revealed_mask = falses(dims...)
        n_mines_flagged = 0
        n_revealed = 0
        game_over = false
        return new(board, revealed_mask, dims, n_mines, n_revealed, game_over)
    end

end

function place_mines!(game::MinesweeperGame, pos::CartesianIndex)::Vector{CartesianIndex}
    positions = CartesianIndices(game.board)
    positions = filter(x -> x != pos, positions)
    mine_idxs = sample(positions, game.n_mines, replace=false)
    game.board[mine_idxs] .= -1
    return mine_idxs
end

function cell_counts!(game::MinesweeperGame, mine_idxs)
    xr, yr = game.dims
    for mine_idx ∈ mine_idxs
        r, c = mine_idx |> Tuple
        for i in r-1:r+1
            for j in c-1:c+1
                if i>=1 && j>=1 && i<=xr && j<=yr 
                    if game.board[i,j] != -1
                        game.board[i,j] += 1
                    end
                end
            end
        end
    end
end

function first_move!(game::MinesweeperGame, pos::CartesianIndex)::Bool
    @pipe place_mines!(game, pos) |> cell_counts!(game, _)
    update_reveal!(game, pos)
    return game.game_over
end

function move!(game::MinesweeperGame, pos::CartesianIndex)::Bool
    game.board[pos] == -1 ? game.game_over = true : update_reveal!(game, pos)
    return game.game_over
end

check_win(game::MinesweeperGame)::Bool = game.n_revealed + game.n_mines == foldl(*,game.dims)

function update_reveal!(game::MinesweeperGame, pos::CartesianIndex)
    xr, yr = game.dims
    queue = Vector{Tuple}()
    push!(queue, pos |> Tuple)
    while !isempty(queue)
        clx, cly = pop!(queue)
        for r in clx-1:clx+1
            for c in cly-1:cly+1
                if r>=1 && c>=1 && r<=xr && c<=yr
                    if !game.revealed_mask[r,c] && game.board[r,c] != -1
                        game.revealed_mask[r,c] = true
                        game.n_revealed += 1
                        if game.board[r,c] == 0
                            push!(queue, (r,c))
                        end
                    end
                end
            end
        end
    end
end
