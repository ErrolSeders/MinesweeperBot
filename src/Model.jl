using BSON 
using Dates
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

function save_model(model, model_name)
    now = Dates.now()
    filename = "$(model_name)_$(Dates.format(now, "yyyymmdd_HHMMSS")).bson"
    BSON.@save filename model
end