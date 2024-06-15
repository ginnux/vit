def get_config():
    config = {}
    config["learning_rate"] = 1e-5
    config["batch_size"] = 64
    config["num_workers"] = 8
    config["pretrained_model"] = "vit-base-patch16-224-in21k"
    config["load_last_checkpoint"] = True
    config["epochs"] = 3

    return config
