# Toward Compositional Generalization in Object-Oriented World Modeling

[//]: # (This is an active repository maintaining a set of environments and algorithms in object-based reinforcement learning.)

**Note: The files for benchmarking other models are moved to the `dev_benchmark` branch.**

## Install required packages (Python 3.6) 
    pip install -r requirements.txt


## Generate data
The script and default config file are located in `./gen_data` folder. Run the following commands under root folder to generate data for basic `Shapes` environment (with all shapes and absolute-orientation actions: north, south, west, and east) and `Rush Hour` environment (with only triangles relative-and orientation actions: forward, backward, left, right). 


### Basic Shapes environment (saving pickle file for persistent object library)

    python -m gen_data.run_data_gen \
    gen_env='Shapes' \
    gen_mode='joint' \
    config_shapes.data_folder='./datasets' \
    config_shapes.data_prefix='shapes_library_100train1eval' \
    config_shapes.shapes='all' \
    config_shapes.cascade_gen=True \
    config_shapes.num_objects_total_list='[5, 10, 15, 20, 30, 40, 50]' \
    config_shapes.num_objects_scene=5 \
    config_shapes.shuffle_color=True \
    config_shapes.num_episodes.train=1000 \
    config_shapes.num_episodes.eval=10000 \
    config_shapes.num_config_eval=1 \
    config_shapes.num_config_train=100 \
    config_shapes.pickle_library=True \
    config_shapes.load_pickle_library=False \
    config_shapes.load_pickle_file=None

### Rush Hour environment

    python -m gen_data.run_data_gen \
    gen_env='Shapes' \
    gen_mode='joint' \
    config_shapes.data_folder='./datasets' \
    config_shapes.data_prefix='rush-hour_library_100train1eval' \
    config_shapes.shapes='rush_hour' \
    config_shapes.cascade_gen=True \
    config_shapes.num_objects_total_list='[5, 10, 15, 20, 30, 40, 50]' \
    config_shapes.num_objects_scene=5 \
    config_shapes.shuffle_color=True \
    config_shapes.num_episodes.train=1000 \
    config_shapes.num_episodes.eval=10000 \
    config_shapes.num_config_eval=1 \
    config_shapes.num_config_train=100 \
    config_shapes.pickle_library=True \
    config_shapes.load_pickle_library=False \
    config_shapes.load_pickle_file=None

## Demo Run - to update

    python -m scripts.run -p -l INFO train with config_model \
    model_train.dataset='datasets/shapes_library_n10k5_train_debug_config_unlimited.h5' \
    model_eval.dataset='datasets/shapes_library_n10k5_eval_debug_config_unlimited.h5' \
    model_train.num_objects=5 \
    model_train.num_objects_total=10 \
    model_train.epochs=1 \
    save_folder='checkpoints/homo-slot-attention-experiments' \
    model_train.homo_slot_att=True \
    model_train.decoupled_homo_att=False \
    model_train.batch_size=1024 \
    model_train.learning_rate=3e-3 \
    model_train.encoder_type=specific \
    model_train.embedding_dim=4 \
    model_train.num_iterations=3 \
    num_training_episodes=1000 \
    data_config=config_unlimited_neg0.5diff \
    model_train.same_config_ratio=0.5 \
    stats_collection='homo_slot_att_debug' \
    description='debug - config + recon'


## Structure (to update)

- Environments (Static/Actionable) - `envs/`
- Data Generation - `gen_data/`
- Algorithms - `algorithms/`
- Utilities - `util`
- Running entrance scripts - `runs/`
- Configuration files - `Config/`
