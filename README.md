# Federated Learning for LibSignal (`federated`)

This module adds **Federated Learning (FL)** to LibSignal. Each traffic intersection acts as an independent **Client** that trains its own RL agent locally and periodically shares its model weights with a central **Trainer** (server).

---

## 1. Folder Structure

```text
federated/
├── agent/          # Base Model Implementation
├── profile/        # System Settings for every Clients
├── compression/    # Weight Compression Module
├── client/         # Combines (Agent + Profile + Compressor) into one unit.
├── selection/      # Client Selection Module
├── aggregation/    # Model Aggregation Module
├── trainer/        # Runs the simulation and federated rounds.
└── configs/        # YAML configuration files.
```

### How the Federated Loop Works

```text
[1. Select]       Trainer picks clients for the round using Selection.
     │
     ▼
[2. Local Train]  Selected Clients train their local RL models for N episodes
     │            (All intersections still control traffic in the simulation).
     ▼
[3. Upload]       Selected Clients compress and send their trained weights
     │            (Profile can randomly drop packets to simulate bad network).
     ▼
[4. Aggregate]    Trainer decompresses weights and merges them via Aggregation.
     │
     ▼
[5. Broadcast]    Trainer sends the new global weights back to all Clients.
```

---

## 2. How to Run

Run [`federated_run.py`](../federated_run.py) from the `LibSignal/` root folder:

```bash
python federated_run.py
```

### Changing Settings (`federated/configs/fl_tsc/fl_dqn.yml`)
You can swap strategies or tweak hardware settings directly in the YAML file without changing code:

```yaml
trainer:
  aggregate_interval: 5   # Sync weights every 5 episodes

  selection:
    type: "all"           # "all" or "random" (with fraction: 0.5)

  aggregation:
    type: "fedavg"
    weighted: False       # True = busy intersections get more weight

  compression:
    type: "none"

  client_profiles:
    type: "simple"
    default:
      buffer_size: 5000   # Max replay buffer memory
      drop_rate: 0.0      # Probability (0.0 - 1.0) that a client's update is lost
```

---

## 3. How to Add New Algorithms

### A. Add a New Aggregator, Selector, Compressor, or Profile
All four strategy folders follow the exact same 3 steps:

1. **Create your file** in the matching folder:
   - **Aggregation** (`federated/aggregation/`): Inherit [`BaseAggregator`](aggregation/base.py) and implement `aggregate(self, client_weights, sample_counts=None, **kwargs)`.
   - **Selection** (`federated/selection/`): Inherit [`BaseClientSelector`](selection/base.py) and implement `select(self, clients, **kwargs)`.
   - **Compression** (`federated/compression/`): Inherit [`BaseCompressor`](compression/base.py) and implement `compress(self, weights, **kwargs)` and `decompress(self, compressed_weights, **kwargs)`.
   - **Profile** (`federated/profile/`): Inherit [`BaseProfile`](profile/base.py) and implement `is_dropped(self)` and `get_latency(self)`.
2. **Register it** in that folder's `__init__.py` by adding `"your_name": YourClass` to the `..._REGISTRY` dictionary (`AGGREGATOR_REGISTRY`, `SELECTOR_REGISTRY`, `COMPRESSOR_REGISTRY`, or `PROFILE_REGISTRY`).
3. **Enable it** in your YAML config under `trainer:` by setting `type: "your_name"` (any extra YAML keys in that block are automatically passed to your class's `__init__`).

---

### B. Add a New RL Agent (`federated/agent/`)
1. **Create your file** in `federated/agent/` (e.g., `federated/agent/my_agent.py`):
   - Inherit from a LibSignal agent (e.g., `DQNAgent`, `FRAP_DQNAgent`, `PressLightAgent`) and [`BaseFLAgent`](agent/base.py).
   - Decorate the class with `@Registry.register_model('your_agent_name')`.
   - Implement `get_weights(self)` and `set_weights(self, weights)`.
   - *(Optional)* If your algorithm modifies the local loss (e.g., **FedProx**), inherit from [`FL_DQNAgent`](agent/dqn.py) and override `regularization_loss(self)`.
2. **Import it** in [`federated/agent/__init__.py`](agent/__init__.py) so the `@Registry` decorator runs on startup.
3. **Create a config** at `federated/configs/fl_tsc/your_agent_name.yml` (with `model.name: your_agent_name`) and run with `-a your_agent_name`.

---

### C. Add a New Trainer / Communication Schedule (`federated/trainer/`)
1. **Create your file** in `federated/trainer/` (e.g., `federated/trainer/async_trainer.py`):
   - Inherit from [`BaseFLTrainer`](trainer/base.py) (or [`SyncFLTrainer`](trainer/sync.py)).
   - Decorate the class with `@Registry.register_trainer('your_task_name')`.
   - Implement `federated_round(self)` (how weights are collected and aggregated) and `train(self)` (when rounds are triggered across `self.train_episode(e)`).
   - *(Optional)* Override `on_step_end(self)` if you want to trigger communication after individual simulation steps inside an episode (e.g., for Async FL).
2. **Import it** in [`federated/trainer/__init__.py`](trainer/__init__.py).
3. **Register a matching task** in [`federated/task/task.py`](task/task.py) with `@Registry.register_task('your_task_name')` and run with `-t your_task_name`.
