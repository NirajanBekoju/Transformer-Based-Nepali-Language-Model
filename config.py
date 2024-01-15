def getConfig(small = False):
    if not small:
        model_config = {
            "emsize" : 300, 
            "d_hid" : 1024,
            "nlayers" : 6,
            "nhead" : 6, 
            "dropout" : 0.2,
            "bptt" : 64
        }
        
    else:
        model_config = {
            "emsize" : 300, 
            "d_hid" : 800,
            "nlayers" : 4,
            "nhead" : 4, 
            "dropout" : 0.05,
            "bptt" : 16
        }

    app_config = {
        "logs" : "tensorboard_logs",
        "epochs" : 25 if not small else 10,
    }
    
 

    return model_config, app_config
