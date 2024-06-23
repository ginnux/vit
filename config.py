def get_config():
    config = {}
    config["learning_rate"] = 3e-5
    config["batch_size"] = 64
    config["num_workers"] = 8
    config["pretrained_model"] = "WinKawaks/vit-small-patch16-224"
    config["load_last_checkpoint"] = False
    config["epochs"] = 5

    return config
