class RunSummary:
    def __init__(self, model_type:str|None = None, dataset_type:str|None = None, batch_size:int|None = None,
                 target_T: int|None = None, random_crop:bool|None = None, epoch:int|None = None,
                 lr:float|None = None, optim_type:str|None = None, weight_decay:float|None = None,
                 seed:int|None = None, use_augment:bool=False, val_training:bool|None = None,
                 scheduler:bool=False):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.target_T = target_T
        self.random_crop = random_crop
        self.epoch = epoch
        self.lr = lr
        self.optim_type = optim_type
        self.weight_decay = weight_decay
        self.seed = seed
        self.test_results = 0
        self.use_augment = use_augment
        self.name = None
        self.val_training=val_training
        self.scheduler=scheduler

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "dataset_type": self.dataset_type,
            "batch_size": self.batch_size,
            "target_T": self.target_T,
            "random_crop": self.random_crop,
            "epoch": self.epoch,
            "lr": self.lr,
            "optim_type": self.optim_type,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "use_augment":self.use_augment,
            "final_result": self.test_results,
            "name": self.name,
            "val_training":self.val_training,
            "scheduler":self.scheduler
        }

    def load_data(self, data:dict):
        self.model_type = data.get("model_type", "SmallCNN" )
        self.dataset_type = data.get("dataset_type", "MelNpyDataset")
        self.batch_size = data.get("batch_size", 32)
        self.target_T = data.get("target_T", 128)
        self.random_crop = data.get("random_crop", True )
        self.epoch = data.get("epoch", 20)
        self.lr = data.get("lr", 1e-3 )
        self.optim_type = data.get("optim_type", "Adam")
        self.weight_decay = data.get("weight_decay", None)
        self.seed = data.get("seed", 1234)
        self.use_augment = data.get("use_augment", False)
        self.name = data.get("name", "best_model.pt")
        self.val_training = data.get("val_training", False)
        self.scheduler = data.get("scheduler", True)
        self.use_augment = data.get("use_augment", False)
