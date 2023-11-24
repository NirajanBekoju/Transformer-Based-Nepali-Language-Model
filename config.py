def getConfig():
    model_config = {
        "emsize" : 300, 
        "d_hid" : 1024,
        "nlayers" : 6,
        "nhead" : 6, 
        "dropout" : 0.2,
        "bptt" : 64
    }

    app_config = {
        "logs" : "tensorboard_logs",
        "epochs" : 25,
    }

    return model_config, app_config
