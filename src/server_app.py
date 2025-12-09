import torch
from src.task import Net
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, ConfigRecord, Context

def get_server(server_rounds: int = 3, fraction_train: float = 1.0, lr: float = 0.005) -> ServerApp:
    # Create ServerApp
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # Read run config
        #fraction_train: float = context.run_config["fraction-train"]
        num_rounds: int = server_rounds
        #lr: float = context.run_config["lr"]

        # Load global model
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # Initialize FedAvg strategy
        strategy = FedAvg(fraction_train=fraction_train)

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

    return app