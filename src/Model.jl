using Flux

function QNetworkFCN()
    return Chain(
        Conv((3,3), 1 => 32, pad = SamePad(), relu),
        Conv((3,3), 32 => 64, pad = SamePad(), relu),
        Conv((3,3), 64 => 64, pad = SamePad(), relu),
        Conv((3,3), 64 => 128, pad = SamePad(), relu),         
        Conv((3,3), 128 => 1, pad = SamePad(), relu),
    )
end
