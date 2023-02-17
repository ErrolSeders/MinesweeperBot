using Pkg

Pkg.activate(".")

using MinesweeperBot

begin
    hp = HyperParams()
    gp = GameParams()
    mp = ModelParams(gp.board_dim, 0.1)

    train_loop(mp,hp,gp)
end

