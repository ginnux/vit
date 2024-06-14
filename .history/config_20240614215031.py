def get_config():
    config = {}
    config["learning_rate"] = 3e-5
    config["batch_size"] = 64
    config["num_workers"] = 8
    config["pretrained_model"] = "vit-base-patch16-224-in21k"

    return config
