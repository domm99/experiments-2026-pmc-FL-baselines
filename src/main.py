from flwr.simulation import run_simulation
from src.server_app import get_server
from src.client_app import get_client
from src.task import load_data

if __name__ == '__main__':

    print("Starting Simulation...")

    sparsity_levels = [0.5] # TODO - add all levels
    number_of_regions = [5] # TODO - add all
    partitioning_methods = ['Hard'] # TODO - add all
    seeds = range(1) # TODO - increase

    config = {
        "local_epochs": 2,
        "lr": 0.005,
        "global_rounds": 5, # TODO - increase
        "num_clients": 50,
    }

    for seed in seeds:
        for partitioning in partitioning_methods:
            for regions in number_of_regions:

                load_data(
                    dataset_name = 'EMNIST',
                    number_subregions=regions,
                    number_of_devices_per_subregion=config['num_clients']/regions,
                    partitioning_method=partitioning,
                    seed=seed,
                )

                for sparsity_level in sparsity_levels:

                    server_app = get_server(sparsity_level=sparsity_level)
                    client_app = get_client(config['local_epochs'])

                    run_simulation(
                        server_app=server_app,
                        client_app=client_app,
                        num_supernodes=config['num_clients'],
                    )