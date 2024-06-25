# 配置文件，包含了训练的一些参数，如学习率、batch_size等，训练前请根据实际情况修改配置文件。
def get_config():
    config = {}
    config["learning_rate"] = 3e-5
    config["batch_size"] = 64
    config["num_workers"] = 8
    config["pretrained_model"] = "google/vit-base-patch16-224"
    config["load_last_checkpoint"] = False
    config["epochs"] = 5

    return config
