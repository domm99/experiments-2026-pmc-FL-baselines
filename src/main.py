from flwr.simulation import run_simulation
from src.server_app import get_server
from src.client_app import get_client

if __name__ == '__main__':

    print("Starting Simulation...")

    run_config = {
        "local-epochs": 2,
        "lr": 0.005,
        "num-rounds": 3,
    }

    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0,
        }
    }

    server_app = get_server()
    client_app = get_client(10, 2)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=10,
        backend_config=backend_config,
    )